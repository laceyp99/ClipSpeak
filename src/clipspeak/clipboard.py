"""Read a bounded, private snapshot of Windows clipboard text."""

from __future__ import annotations

import ctypes
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


CF_UNICODETEXT = 13
MAX_TEXT_CHARS = 100_000


class ClipboardStatus(str, Enum):
    OK = "ok"
    EMPTY = "empty"
    NON_TEXT = "non_text"
    UNAVAILABLE = "unavailable"
    OVERSIZED = "oversized"


@dataclass(frozen=True)
class ClipboardResult:
    status: ClipboardStatus
    text: str | None = field(default=None, repr=False)


def _win32_functions() -> tuple[Any, Any]:
    user32 = ctypes.WinDLL("user32", use_last_error=True)
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    user32.OpenClipboard.argtypes = [ctypes.c_void_p]
    user32.OpenClipboard.restype = ctypes.c_int
    user32.CloseClipboard.argtypes = []
    user32.CloseClipboard.restype = ctypes.c_int
    user32.IsClipboardFormatAvailable.argtypes = [ctypes.c_uint]
    user32.IsClipboardFormatAvailable.restype = ctypes.c_int
    user32.CountClipboardFormats.argtypes = []
    user32.CountClipboardFormats.restype = ctypes.c_int
    user32.GetClipboardData.argtypes = [ctypes.c_uint]
    user32.GetClipboardData.restype = ctypes.c_void_p
    kernel32.GlobalLock.argtypes = [ctypes.c_void_p]
    kernel32.GlobalLock.restype = ctypes.c_void_p
    kernel32.GlobalUnlock.argtypes = [ctypes.c_void_p]
    kernel32.GlobalUnlock.restype = ctypes.c_int
    kernel32.GlobalSize.argtypes = [ctypes.c_void_p]
    kernel32.GlobalSize.restype = ctypes.c_size_t
    return user32, kernel32


def read_clipboard_text(
    *,
    _functions: tuple[Any, Any] | None = None,
    _sleep: Any = time.sleep,
) -> ClipboardResult:
    """Copy CF_UNICODETEXT while open, then release Windows clipboard handles."""
    try:
        user32, kernel32 = _functions or _win32_functions()
    except (AttributeError, OSError):
        return ClipboardResult(ClipboardStatus.UNAVAILABLE)

    for attempt in range(5):
        if user32.OpenClipboard(None):
            break
        if attempt < 4:
            _sleep(0.05)
    else:
        return ClipboardResult(ClipboardStatus.UNAVAILABLE)

    try:
        if not user32.IsClipboardFormatAvailable(CF_UNICODETEXT):
            status = ClipboardStatus.NON_TEXT if user32.CountClipboardFormats() else ClipboardStatus.EMPTY
            return ClipboardResult(status)
        handle = user32.GetClipboardData(CF_UNICODETEXT)
        if not handle:
            return ClipboardResult(ClipboardStatus.UNAVAILABLE)
        pointer = kernel32.GlobalLock(handle)
        if not pointer:
            return ClipboardResult(ClipboardStatus.UNAVAILABLE)
        try:
            byte_count = kernel32.GlobalSize(handle)
            if byte_count < 2:
                return ClipboardResult(ClipboardStatus.UNAVAILABLE)
            # A Python character can occupy two UTF-16 code units. Read at
            # most the largest valid text plus its terminating zero unit.
            read_bytes = min(byte_count // 2, MAX_TEXT_CHARS * 2 + 1) * 2
            raw = ctypes.string_at(pointer, read_bytes)
            terminator = next(
                (i for i in range(0, len(raw), 2) if raw[i : i + 2] == b"\x00\x00"),
                None,
            )
            if terminator is None:
                status = ClipboardStatus.OVERSIZED if byte_count > read_bytes else ClipboardStatus.UNAVAILABLE
                return ClipboardResult(status)
            try:
                text = raw[:terminator].decode("utf-16-le")
            except UnicodeDecodeError:
                return ClipboardResult(ClipboardStatus.UNAVAILABLE)
            if len(text) > MAX_TEXT_CHARS:
                return ClipboardResult(ClipboardStatus.OVERSIZED)
            if not text.strip():
                return ClipboardResult(ClipboardStatus.EMPTY)
            return ClipboardResult(ClipboardStatus.OK, text)
        finally:
            kernel32.GlobalUnlock(handle)
    except (OSError, ValueError):
        return ClipboardResult(ClipboardStatus.UNAVAILABLE)
    finally:
        user32.CloseClipboard()
