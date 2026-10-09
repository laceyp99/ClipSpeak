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
        def unavailable():
            raise RuntimeError("copied private text")
        result = controller.submit_clipboard(reader=unavailable)
        assert result.status is SubmissionStatus.UNAVAILABLE
        assert "private" not in repr(result)
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


def test_pause_resume_preserves_active_and_blocks_queue_advance():
    started = threading.Event()
    release = threading.Event()
    calls = []

    def playback(_voice, text, *, control, **_settings):
        calls.append(text)
        started.set()
        release.wait(2)

    controller = QueueController(voice_factory=object, playback=playback)
    try:
        controller.pause()
        assert controller.submit("one").status is SubmissionStatus.ACCEPTED
        assert controller.snapshot().state == "paused"
        assert not started.wait(0.05)
        controller.resume()
        assert started.wait(2)
        controller.pause()
        controller.submit("two")
        release.set()
        wait_until(lambda: controller.snapshot().state == "paused" and not controller.snapshot().has_active)
        assert calls == ["one"]
        controller.resume()
        assert controller.wait_idle(2)
        assert calls == ["one", "two"]
    finally:
        release.set()
        controller.close()
        assert controller.join(2)


@pytest.mark.parametrize("late_error", [False, True])
def test_stop_discards_late_completion_and_serializes_fresh_item(late_error):
    started = threading.Event()
    release = threading.Event()
    calls = []

    def playback(_voice, text, *, control, **_settings):
        calls.append(text)
        if text == "old":
            started.set()
            release.wait(2)
            if late_error:
                raise RuntimeError("old private text")

    controller = QueueController(voice_factory=object, playback=playback)
    try:
        controller.submit("old")
        assert started.wait(2)
        controller.pause()
        controller.submit("pending")
        controller.stop()
        snap = controller.snapshot()
        assert (snap.total_chars, snap.pending_items, snap.has_active, snap.paused) == (0, 0, False, False)
        assert controller.submit("fresh").status is SubmissionStatus.ACCEPTED
        assert calls == ["old"]
        release.set()
        assert controller.wait_idle(2)
        assert calls == ["old", "fresh"]
        assert controller.snapshot().error_type is None
    finally:
        release.set()
        controller.close()
        assert controller.join(2)


def test_clear_queue_preserves_active_and_error_retry_restarts_with_settings():
    started = threading.Event()
    release = threading.Event()
    calls = []

    def playback(_voice, text, *, speed, volume, control):
        calls.append((text, speed, volume))
        if len(calls) == 1:
            started.set()
            release.wait(2)
            raise RuntimeError("secret audio text")

    controller = QueueController(voice_factory=object, playback=playback)
    try:
        controller.submit("secret", speed=2, volume=0.4)
        assert started.wait(2)
        controller.submit("discard")
        controller.clear_queue()
        assert (controller.snapshot().total_chars, controller.snapshot().has_active) == (6, True)
        controller.submit("later")
        release.set()
        wait_until(lambda: controller.snapshot().state == "error")
        snap = controller.snapshot()
        assert (snap.total_chars, snap.pending_items, snap.error_type) == (11, 1, "RuntimeError")
        assert "secret" not in repr(snap)
        assert "beginning" in snap.error_message
        controller.resume()
        assert controller.wait_idle(2)
        assert calls == [("secret", 2, 0.4), ("secret", 2, 0.4), ("later", 1.5, 1)]
    finally:
        release.set()
        controller.close()
        assert controller.join(2)


def test_resume_retries_missing_voice_after_startup_failure():
    attempts = []
    calls = []

    def factory():
        attempts.append(1)
        if len(attempts) == 1:
            raise FileNotFoundError("secret model path")
        return object()

    controller = QueueController(voice_factory=factory,
                                 playback=lambda _voice, text, **_kw: calls.append(text))
    try:
        controller.submit("retained")
        wait_until(lambda: controller.snapshot().state == "error")
        assert "secret" not in repr(controller.snapshot())
        assert "model files" in controller.snapshot().error_message
        controller.resume()
        assert controller.wait_idle(2)
        assert calls == ["retained"] and len(attempts) == 2
    finally:
        controller.close()
        assert controller.join(2)


def test_resume_retries_voice_without_a_queued_item():
    attempts = []

    def factory():
        attempts.append(1)
        if len(attempts) == 1:
            raise FileNotFoundError("model")
        return object()

    controller = QueueController(voice_factory=factory, playback=lambda *_a, **_kw: None)
    try:
        wait_until(lambda: controller.snapshot().state == "error")
        controller.resume()
        wait_until(lambda: len(attempts) == 2)
        assert controller.snapshot().state == "idle"
        assert controller.wait_idle(2)
    finally:
        controller.close()
        assert controller.join(2)


def test_stop_during_voice_load_ignores_stale_error_and_retries_fresh():
    loading = threading.Event()
    release = threading.Event()
    attempts = []
    calls = []

    def factory():
        attempts.append(1)
        if len(attempts) == 1:
            loading.set()
            release.wait(2)
            raise RuntimeError("stale load")
        return object()

    controller = QueueController(voice_factory=factory,
                                 playback=lambda _voice, text, **_kw: calls.append(text))
    try:
        assert loading.wait(2)
        controller.submit("old")
        controller.stop()
        controller.submit("fresh")
        release.set()
        assert controller.wait_idle(2)
        assert calls == ["fresh"] and controller.snapshot().error_type is None
    finally:
        release.set()
        controller.close()
        assert controller.join(2)


def test_slow_clipboard_read_does_not_block_stop_and_is_cancelled():
    reading = threading.Event()
    release = threading.Event()
    results = []
    calls = []

    def reader():
        reading.set()
        release.wait(2)
        return ClipboardResult(ClipboardStatus.OK, "stale")

    controller = QueueController(voice_factory=object,
                                 playback=lambda _voice, text, **_kw: calls.append(text))
    thread = threading.Thread(target=lambda: results.append(controller.submit_clipboard(reader=reader)))
    try:
        thread.start()
        assert reading.wait(2)
        start = time.monotonic()
        controller.stop()
        assert time.monotonic() - start < 0.1
        release.set()
        thread.join(2)
        assert results[0].status is SubmissionStatus.CANCELLED
        assert controller.wait_idle(2)
        assert calls == []
    finally:
        release.set()
        thread.join(2)
        controller.close()
        assert controller.join(2)


@pytest.mark.parametrize("paused", [False, True])
@pytest.mark.parametrize("late_error", [False, True])
def test_close_clears_active_and_pending_and_rejects_new_work(paused, late_error):
    started = threading.Event()
    release = threading.Event()

    def playback(_voice, _text, **_kw):
        started.set()
        release.wait(2)
        if late_error:
            raise RuntimeError("cancelled private text")

    controller = QueueController(voice_factory=object, playback=playback)
    try:
        controller.submit("active")
        assert started.wait(2)
        controller.submit("pending")
        if paused:
            controller.pause()
        controller.quit()
        controller.close()
        snap = controller.snapshot()
        assert (snap.state, snap.total_chars, snap.has_active, snap.pending_items) == ("closing", 0, False, 0)
        assert controller.submit("new").status is SubmissionStatus.CLOSED
    finally:
        release.set()
        assert controller.join(2)


@pytest.mark.parametrize("clipboard", [False, True])
def test_stop_cancels_submissions_waiting_behind_a_clipboard_read(clipboard):
    reading = threading.Event()
    release = threading.Event()
    waiting = threading.Event()
    results = []
    class SubmissionLock:
        def __init__(self):
            self.lock = threading.Lock()
        def __enter__(self):
            if threading.current_thread().name == "waiting submission":
                waiting.set()
            self.lock.acquire()
        def __exit__(self, *_):
            self.lock.release()
    def reader():
        reading.set()
        release.wait(2)
        return ClipboardResult(ClipboardStatus.OK, "old")
    controller = QueueController(voice_factory=object, playback=lambda *_a, **_kw: None)
    controller._submission_lock = SubmissionLock()
    first = threading.Thread(target=lambda: results.append(controller.submit_clipboard(reader=reader)))
    def queued_submission():
        result = (controller.submit_clipboard(reader=lambda: ClipboardResult(ClipboardStatus.OK, "old"))
                  if clipboard else controller.submit("old"))
        results.append(result)
    second = threading.Thread(target=queued_submission, name="waiting submission")
    try:
        first.start()
        assert reading.wait(2)
        second.start()
        assert waiting.wait(2)
        controller.stop()
        release.set()
        first.join(2)
        second.join(2)
        assert len(results) == 2
        assert all(result.status is SubmissionStatus.CANCELLED for result in results)
        assert controller.snapshot().total_chars == 0
        assert controller.submit("fresh").status is SubmissionStatus.ACCEPTED
        assert controller.wait_idle(2)
    finally:
        release.set()
        first.join(2)
        if second.ident is not None:
            second.join(2)
        controller.close()
        assert controller.join(2)
