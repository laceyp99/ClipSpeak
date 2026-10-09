"""In-memory FIFO ownership for clipboard speech submissions."""

from __future__ import annotations

import math
import threading
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable

from .clipboard import ClipboardResult, ClipboardStatus, read_clipboard_text
from .playback import PlaybackControl, play_text
from .synthesis import DEFAULT_VOICE, ResidentVoice


MAX_ITEM_CHARS = 100_000
MAX_PENDING_ITEMS = 20
MAX_TOTAL_CHARS = 500_000


class SubmissionStatus(str, Enum):
    ACCEPTED = "accepted"
    EMPTY = "empty"
    NON_TEXT = "non_text"
    UNAVAILABLE = "unavailable"
    OVERSIZED = "oversized"
    QUEUE_FULL = "queue_full"
    INVALID_SETTINGS = "invalid_settings"
    CANCELLED = "cancelled"
    CLOSED = "closed"


@dataclass(frozen=True)
class SubmissionResult:
    status: SubmissionStatus
    reason: str = ""


@dataclass(frozen=True)
class QueueItem:
    text: str = field(repr=False)
    speed: float = 1.5
    volume: float = 1.0


@dataclass(frozen=True)
class QueueSnapshot:
    state: str
    pending_items: int
    total_chars: int
    has_active: bool
    paused: bool
    error_type: str | None
    error_message: str | None = None


class QueueController:
    """Serialize submission and play one immutable snapshot at a time."""

    def __init__(
        self,
        *,
        voice_factory: Callable[[], object] = lambda: ResidentVoice(DEFAULT_VOICE),
        playback: Callable = play_text,
    ):
        self._condition = threading.Condition()
        self._submission_lock = threading.Lock()
        self._pending: deque[QueueItem] = deque()
        self._active: QueueItem | None = None
        self._control: PlaybackControl | None = None
        self._total_chars = 0
        self._paused = False
        self._error_type: str | None = None
        self._error_message: str | None = None
        self._closing = False
        self._generation = 0
        self._voice_retry_requested = False
        self._voice_unavailable = False
        self._voice_factory = voice_factory
        self._playback = playback
        self._worker = threading.Thread(target=self._run, name="ClipSpeak queue", daemon=True)
        self._worker.start()

    def submit(self, text: str, *, speed: float = 1.5, volume: float = 1.0) -> SubmissionResult:
        with self._condition:
            generation = self._generation
        with self._submission_lock, self._condition:
            if not self._closing and generation != self._generation:
                return SubmissionResult(SubmissionStatus.CANCELLED, "Submission was cancelled")
            return self._submit_locked(text, speed, volume)

    def submit_clipboard(
        self,
        *,
        speed: float = 1.5,
        volume: float = 1.0,
        reader: Callable = read_clipboard_text,
    ) -> SubmissionResult:
        # Serialize submissions without delaying playback controls during a slow clipboard read.
        with self._condition:
            generation = self._generation
        with self._submission_lock:
            with self._condition:
                if self._closing:
                    return SubmissionResult(SubmissionStatus.CLOSED, "ClipSpeak is closing")
                if generation != self._generation:
                    return SubmissionResult(SubmissionStatus.CANCELLED, "Clipboard read was cancelled")
            try:
                result = reader()
            except Exception:
                result = ClipboardResult(ClipboardStatus.UNAVAILABLE)
            with self._condition:
                if self._closing:
                    return SubmissionResult(SubmissionStatus.CLOSED, "ClipSpeak is closing")
                if generation != self._generation:
                    return SubmissionResult(SubmissionStatus.CANCELLED, "Clipboard read was cancelled")
                if result.status != ClipboardStatus.OK:
                    status = SubmissionStatus(result.status.value)
                    return SubmissionResult(status, _REASONS[status])
                return self._submit_locked(result.text, speed, volume)

    def _submit_locked(self, text: str, speed: float, volume: float) -> SubmissionResult:
        if self._closing:
            return SubmissionResult(SubmissionStatus.CLOSED, "ClipSpeak is closing")
        if not isinstance(text, str) or not text.strip():
            return SubmissionResult(SubmissionStatus.EMPTY, _REASONS[SubmissionStatus.EMPTY])
        if len(text) > MAX_ITEM_CHARS:
            return SubmissionResult(SubmissionStatus.OVERSIZED, _REASONS[SubmissionStatus.OVERSIZED])
        try:
            speed, volume = float(speed), float(volume)
        except (TypeError, ValueError, OverflowError):
            return SubmissionResult(SubmissionStatus.INVALID_SETTINGS, _REASONS[SubmissionStatus.INVALID_SETTINGS])
        if not (math.isfinite(speed) and 1.0 <= speed <= 2.0 and math.isfinite(volume) and 0.0 <= volume <= 1.0):
            return SubmissionResult(SubmissionStatus.INVALID_SETTINGS, _REASONS[SubmissionStatus.INVALID_SETTINGS])
        if len(self._pending) >= MAX_PENDING_ITEMS:
            return SubmissionResult(SubmissionStatus.QUEUE_FULL, "Queue already has 20 pending items")
        if self._total_chars + len(text) > MAX_TOTAL_CHARS:
            return SubmissionResult(SubmissionStatus.QUEUE_FULL, "Queued text would exceed 500,000 characters")
        self._pending.append(QueueItem(text, speed, volume))
        self._total_chars += len(text)
        self._condition.notify_all()
        return SubmissionResult(SubmissionStatus.ACCEPTED)

    def snapshot(self) -> QueueSnapshot:
        with self._condition:
            state = ("closing" if self._closing else "error" if self._error_type else
                     "paused" if self._paused else
                     "playing" if self._active else "queued" if self._pending else "idle")
            return QueueSnapshot(
                state, len(self._pending), self._total_chars,
                self._active is not None, self._paused,
                self._error_type, self._error_message,
            )

    def pause(self) -> None:
        """Hold the active playback cursor and stop queue advancement."""
        with self._condition:
            if not self._closing and not self._error_type:
                self._paused = True
                if self._control is not None:
                    self._control.pause()
                self._condition.notify_all()

    def resume(self) -> None:
        """Resume playback, or retry a failed item from its beginning."""
        with self._condition:
            if self._closing:
                return
            if self._error_type and self._voice_unavailable and self._active is None:
                self._voice_retry_requested = True
            self._error_type = self._error_message = None
            self._paused = False
            if self._control is not None:
                self._control.resume()
            self._condition.notify_all()

    def stop(self) -> None:
        """Discard current and pending speech without waiting for inference."""
        with self._condition:
            if self._closing:
                return
            self._generation += 1
            self._voice_retry_requested = False
            self._pending.clear()
            self._active = None
            self._total_chars = 0
            self._paused = False
            self._error_type = self._error_message = None
            control = self._control
            self._control = None
            self._condition.notify_all()
        if control is not None:
            control.stop()

    def clear_queue(self) -> None:
        """Discard pending submissions while preserving active speech."""
        with self._condition:
            if self._closing:
                return
            self._pending.clear()
            self._total_chars = len(self._active.text) if self._active else 0
            self._condition.notify_all()

    def wait_idle(self, timeout: float | None = None) -> bool:
        """Wait for a drained queue, or return false on error or close."""
        with self._condition:
            finished = self._condition.wait_for(
                lambda: self._closing or self._error_type is not None or
                (self._active is None and not self._pending), timeout,
            )
            return bool(finished and not self._closing and self._error_type is None)

    def close(self) -> None:
        """Request shutdown immediately; join separately outside UI callbacks."""
        with self._condition:
            if self._closing:
                return
            self._closing = True
            self._generation += 1
            self._voice_retry_requested = False
            self._pending.clear()
            self._active = None
            self._total_chars = 0
            self._paused = False
            self._error_type = self._error_message = None
            control = self._control
            self._control = None
            self._condition.notify_all()
        if control is not None:
            control.stop()

    quit = close

    def join(self, timeout: float | None = None) -> bool:
        self._worker.join(timeout)
        return not self._worker.is_alive()

    def _run(self) -> None:
        voice = None
        with self._condition:
            load_generation = self._generation
        try:
            voice = self._voice_factory()
        except Exception as exc:
            with self._condition:
                if not self._closing and load_generation == self._generation:
                    self._voice_unavailable = True
                    self._set_error_locked(exc)
        while True:
            with self._condition:
                self._condition.wait_for(lambda: self._closing or
                    (not self._paused and self._error_type is None and
                     (self._active or self._pending or self._voice_retry_requested)))
                if self._closing:
                    return
                if self._voice_retry_requested and self._active is None and not self._pending:
                    self._voice_retry_requested = False
                    generation = self._generation
                    retry_voice_only = True
                else:
                    self._voice_retry_requested = False
                    retry_voice_only = False
                if retry_voice_only:
                    item = None
                    control = None
                else:
                    if self._active is None:
                        self._active = self._pending.popleft()
                    item = self._active
                    generation = self._generation
                    control = PlaybackControl()
                    self._control = control
            try:
                if voice is None:
                    voice = self._voice_factory()
                with self._condition:
                    if not self._closing and generation == self._generation:
                        self._voice_unavailable = False
                    current = not self._closing and generation == self._generation
                    if current and self._paused and control is not None:
                        control.pause()
                if current and item is not None:
                    self._playback(voice, item.text, speed=item.speed, volume=item.volume, control=control)
            except Exception as exc:
                with self._condition:
                    if not self._closing and generation == self._generation:
                        if voice is None:
                            self._voice_unavailable = True
                        self._set_error_locked(exc)
                    if self._control is control:
                        self._control = None
                    self._condition.notify_all()
                del item, control
                continue
            with self._condition:
                if self._control is control:
                    self._control = None
                if item is not None and not self._closing and generation == self._generation:
                    self._active = None
                    self._total_chars -= len(item.text)
                self._condition.notify_all()
            del item, control

    def _set_error_locked(self, exc: Exception) -> None:
        self._error_type = type(exc).__name__
        if self._voice_unavailable:
            self._error_message = "Voice is unavailable. Check model files and runtime, then Resume to retry."
        else:
            self._error_message = "Speech stopped. Check audio output and voice runtime. Resume retries the current item from the beginning."
        self._condition.notify_all()


_REASONS = {
    SubmissionStatus.EMPTY: "Clipboard has no text",
    SubmissionStatus.NON_TEXT: "Clipboard has no plain text",
    SubmissionStatus.UNAVAILABLE: "Clipboard is unavailable",
    SubmissionStatus.OVERSIZED: "Text exceeds 100,000 characters",
    SubmissionStatus.INVALID_SETTINGS: "Speed or volume is outside the allowed range",
}
