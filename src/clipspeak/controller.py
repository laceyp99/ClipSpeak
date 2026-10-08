"""In-memory FIFO ownership for clipboard speech submissions."""

from __future__ import annotations

import math
import threading
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable

from .clipboard import ClipboardStatus, read_clipboard_text
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


class QueueController:
    """Serialize submission and play one immutable snapshot at a time."""

    def __init__(
        self,
        *,
        voice_factory: Callable[[], object] = lambda: ResidentVoice(DEFAULT_VOICE),
        playback: Callable = play_text,
    ):
        self._condition = threading.Condition()
        self._pending: deque[QueueItem] = deque()
        self._active: QueueItem | None = None
        self._control: PlaybackControl | None = None
        self._total_chars = 0
        self._paused = False
        self._error_type: str | None = None
        self._closing = False
        self._voice_factory = voice_factory
        self._playback = playback
        self._worker = threading.Thread(target=self._run, name="ClipSpeak queue", daemon=True)
        self._worker.start()

    def submit(self, text: str, *, speed: float = 1.5, volume: float = 1.0) -> SubmissionResult:
        with self._condition:
            return self._submit_locked(text, speed, volume)

    def submit_clipboard(
        self,
        *,
        speed: float = 1.5,
        volume: float = 1.0,
        reader: Callable = read_clipboard_text,
    ) -> SubmissionResult:
        # Holding this lock through the bounded read gives concurrent hotkey
        # submissions the same order in which their reads began.
        with self._condition:
            if self._closing:
                return SubmissionResult(SubmissionStatus.CLOSED, "ClipSpeak is closing")
            try:
                result = reader()
            except Exception:
                return SubmissionResult(SubmissionStatus.UNAVAILABLE, "Clipboard is unavailable")
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
                     "playing" if self._active else "queued" if self._pending else "idle")
            return QueueSnapshot(
                state, len(self._pending), self._total_chars,
                self._active is not None, self._paused,
                self._error_type,
            )

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
            self._closing = True
            self._pending.clear()
            self._total_chars = len(self._active.text) if self._active else 0
            control = self._control
            self._condition.notify_all()
        if control is not None:
            control.stop()

    def join(self, timeout: float | None = None) -> bool:
        self._worker.join(timeout)
        return not self._worker.is_alive()

    def _run(self) -> None:
        voice = None
        try:
            voice = self._voice_factory()
        except Exception as exc:
            with self._condition:
                self._error_type = type(exc).__name__
                self._condition.notify_all()
        while True:
            with self._condition:
                self._condition.wait_for(lambda: self._closing or (self._pending and self._error_type is None))
                if self._closing:
                    self._active = None
                    self._control = None
                    self._error_type = None
                    self._total_chars = 0
                    return
                item = self._pending.popleft()
                self._active = item
                control = PlaybackControl()
                self._control = control
            try:
                if voice is None:
                    voice = self._voice_factory()
                with self._condition:
                    closing = self._closing
                if not closing:
                    self._playback(voice, item.text, speed=item.speed, volume=item.volume, control=control)
            except Exception as exc:
                with self._condition:
                    if not self._closing:
                        self._error_type = type(exc).__name__
                    self._control = None
                    self._condition.notify_all()
                continue
            with self._condition:
                self._control = None
                self._active = None
                self._total_chars -= len(item.text)
                self._condition.notify_all()
            del item, control


_REASONS = {
    SubmissionStatus.EMPTY: "Clipboard has no text",
    SubmissionStatus.NON_TEXT: "Clipboard has no plain text",
    SubmissionStatus.UNAVAILABLE: "Clipboard is unavailable",
    SubmissionStatus.OVERSIZED: "Text exceeds 100,000 characters",
    SubmissionStatus.INVALID_SETTINGS: "Speed or volume is outside the allowed range",
}
