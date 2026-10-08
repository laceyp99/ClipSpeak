import threading
import time

import pytest

from clipspeak.clipboard import ClipboardResult, ClipboardStatus
from clipspeak.controller import QueueController, SubmissionStatus


def wait_until(predicate, timeout=2):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.005)
    raise AssertionError("controller did not reach the expected state")


def test_fifo_snapshots_duplicates_and_waits_for_playback_end():
    release = threading.Event()
    started = threading.Event()
    calls = []
    voices = []

    def voice_factory():
        voice = object()
        voices.append(voice)
        return voice

    def playback(voice, text, *, speed, volume, control):
        calls.append((voice, text, speed, volume))
        started.set()
        release.wait(2)

    controller = QueueController(voice_factory=voice_factory, playback=playback)
    try:
        clipboard = ["same"]
        reader = lambda: ClipboardResult(ClipboardStatus.OK, clipboard[0])
        assert controller.submit_clipboard(reader=reader, speed=1.0).status == SubmissionStatus.ACCEPTED
        assert started.wait(2)
        assert controller.submit_clipboard(reader=reader, speed=1.5, volume=0.5).status == SubmissionStatus.ACCEPTED
        clipboard[0] = "changed"
        assert controller.submit_clipboard(reader=reader, speed=2.0).status == SubmissionStatus.ACCEPTED
        assert controller.snapshot().pending_items == 2
        assert len(calls) == 1
        release.set()
        assert controller.wait_idle(2)
        assert [(text, speed, volume) for _, text, speed, volume in calls] == [
            ("same", 1.0, 1.0), ("same", 1.5, 0.5), ("changed", 2.0, 1.0)
        ]
        assert all(voice is voices[0] for voice, *_ in calls)
        assert len(voices) == 1
        assert controller.snapshot().total_chars == 0
    finally:
        release.set()
        controller.close()
        assert controller.join(2)


def test_limits_include_active_and_preserve_pending_on_error():
    started = threading.Event()
    fail = threading.Event()

    def playback(_voice, _text, **_kwargs):
        started.set()
        fail.wait(2)
        raise RuntimeError("private text must not appear in status")

    controller = QueueController(voice_factory=object, playback=playback)
    try:
        assert controller.submit("a" * 100_000).status == SubmissionStatus.ACCEPTED
        assert started.wait(2)
        for _ in range(20):
            assert controller.submit("b" * 20_000).status == SubmissionStatus.ACCEPTED
        snap = controller.snapshot()
        assert (snap.pending_items, snap.total_chars, snap.has_active) == (20, 500_000, True)
        assert controller.submit("c").status == SubmissionStatus.QUEUE_FULL
        assert controller.submit("x" * 100_001).status == SubmissionStatus.OVERSIZED
        fail.set()
        wait_until(lambda: controller.snapshot().state == "error")
        snap = controller.snapshot()
        assert (snap.pending_items, snap.total_chars, snap.has_active, snap.error_type) == (
            20, 500_000, True, "RuntimeError"
        )
        assert "private" not in repr(snap)
        assert not controller.wait_idle(0.01)
    finally:
        fail.set()
        controller.close()
        assert controller.join(2)


def test_invalid_and_unavailable_inputs_are_not_queued():
    controller = QueueController(voice_factory=object, playback=lambda *_a, **_kw: None)
    try:
        assert controller.submit("  ").status == SubmissionStatus.EMPTY
        assert controller.submit("x", speed=float("nan")).status == SubmissionStatus.INVALID_SETTINGS
        assert controller.submit("x", volume=2).status == SubmissionStatus.INVALID_SETTINGS
        for status in (ClipboardStatus.EMPTY, ClipboardStatus.NON_TEXT, ClipboardStatus.UNAVAILABLE,
                       ClipboardStatus.OVERSIZED):
            result = controller.submit_clipboard(reader=lambda: ClipboardResult(status))
            assert result.status.value == status.value
        assert controller.snapshot().pending_items == 0
    finally:
        controller.close()
        assert controller.join(2)
    assert controller.submit("x").status == SubmissionStatus.CLOSED


def test_clipboard_reads_and_submissions_are_serialized():
    reading = threading.Event()
    release = threading.Event()
    calls = []

    def slow_reader():
        reading.set()
        release.wait(2)
        return ClipboardResult(ClipboardStatus.OK, "first")

    controller = QueueController(voice_factory=object, playback=lambda _v, text, **_kw: calls.append(text))
    try:
        first = threading.Thread(target=lambda: controller.submit_clipboard(reader=slow_reader))
        second = threading.Thread(target=lambda: controller.submit("second"))
        first.start()
        assert reading.wait(2)
        second.start()
        release.set()
        first.join(2)
        second.join(2)
        assert controller.wait_idle(2)
        assert calls == ["first", "second"]
    finally:
        release.set()
        controller.close()
        assert controller.join(2)


def test_close_during_voice_load_prevents_playback():
    loading = threading.Event()
    release = threading.Event()
    calls = []

    def voice_factory():
        loading.set()
        release.wait(2)
        return object()

    controller = QueueController(voice_factory=voice_factory, playback=lambda *_a, **_kw: calls.append(1))
    assert controller.submit("one").status == SubmissionStatus.ACCEPTED
    assert loading.wait(2)
    controller.close()
    release.set()
    assert controller.join(2)
    assert calls == []


@pytest.mark.parametrize("pending_count,pending_chars", [(20, 1), (4, 100_000)])
def test_pending_count_and_total_character_limits_independently(pending_count, pending_chars):
    started = threading.Event()
    release = threading.Event()
    def playback(*_args, **_kwargs):
        started.set()
        release.wait(2)
    controller = QueueController(voice_factory=object, playback=playback)
    try:
        assert controller.submit("a" * 100_000).status is SubmissionStatus.ACCEPTED
        assert started.wait(2)
        for _ in range(pending_count):
            assert controller.submit("b" * pending_chars).status is SubmissionStatus.ACCEPTED
        before = controller.snapshot()
        assert controller.submit("c").status is SubmissionStatus.QUEUE_FULL
        assert controller.snapshot() == before
    finally:
        release.set()
        controller.close()
        assert controller.join(2)


def test_voice_loads_at_startup_and_load_failure_blocks_reading():
    loading = threading.Event()
    release = threading.Event()
    calls = []
    def failing_factory():
        loading.set()
        release.wait(2)
        raise FileNotFoundError("private path")
    controller = QueueController(voice_factory=failing_factory,
                                 playback=lambda *_a, **_kw: calls.append(1))
    try:
        assert loading.wait(2)
        assert controller.submit("retained").status is SubmissionStatus.ACCEPTED
        release.set()
        assert not controller.wait_idle(2)
        snap = controller.snapshot()
        assert (snap.state, snap.error_type, snap.pending_items, snap.total_chars) == (
            "error", "FileNotFoundError", 1, 8)
        assert not snap.has_active and calls == []
        assert "private" not in repr(snap)
    finally:
        release.set()
        controller.close()
        assert controller.join(2)
        assert controller.snapshot().total_chars == 0
