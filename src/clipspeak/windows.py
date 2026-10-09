"""Windows single-instance and global shortcut primitives for the tray app."""

from __future__ import annotations

import ctypes
import threading
from collections import deque
from ctypes import wintypes
from dataclasses import dataclass
from typing import Callable


MOD_ALT = 0x0001
MOD_CONTROL = 0x0002
MOD_SHIFT = 0x0004
MOD_WIN = 0x0008
MOD_NOREPEAT = 0x4000
WM_HOTKEY = 0x0312
WM_APP_COMMAND = 0x8001
ERROR_ALREADY_EXISTS = 183

_MODIFIERS = {"ctrl": MOD_CONTROL, "alt": MOD_ALT, "shift": MOD_SHIFT, "win": MOD_WIN}
_MODIFIER_LABELS = ((MOD_CONTROL, "Ctrl"), (MOD_ALT, "Alt"), (MOD_SHIFT, "Shift"), (MOD_WIN, "Win"))


@dataclass(frozen=True)
class Shortcut:
    modifiers: int
    virtual_key: int

    def __post_init__(self) -> None:
        if not self.modifiers or self.modifiers & ~(MOD_ALT | MOD_CONTROL | MOD_SHIFT | MOD_WIN):
            raise ValueError("Shortcut needs at least one supported modifier")
        if not _key_label(self.virtual_key):
            raise ValueError("Shortcut needs a letter, digit, or F1 through F24")


def _key_label(key: int) -> str:
    if 0x41 <= key <= 0x5A or 0x30 <= key <= 0x39:
        return chr(key)
    if 0x70 <= key <= 0x87:
        return f"F{key - 0x6F}"
    return ""


def parse_shortcut(text: str) -> Shortcut:
    """Parse a user-entered shortcut into Win32 modifiers and virtual key."""
    if not isinstance(text, str):
        raise ValueError("Enter a shortcut such as Ctrl+Alt+A")
    parts = [part.strip().lower() for part in text.split("+")]
    if len(parts) < 2 or any(not part for part in parts):
        raise ValueError("Enter a shortcut such as Ctrl+Alt+A")
    modifiers = 0
    for part in parts[:-1]:
        flag = _MODIFIERS.get(part)
        if flag is None or modifiers & flag:
            raise ValueError("Use each supported modifier at most once")
        modifiers |= flag
    key_name = parts[-1].upper()
    if len(key_name) == 1 and ("A" <= key_name <= "Z" or "0" <= key_name <= "9"):
        key = ord(key_name)
    elif key_name.startswith("F") and key_name[1:].isdigit() and 1 <= int(key_name[1:]) <= 24:
        key = 0x6F + int(key_name[1:])
    else:
        raise ValueError("Use a letter, digit, or F1 through F24 as the key")
    return Shortcut(modifiers, key)


def format_shortcut(shortcut: Shortcut) -> str:
    return "+".join([*(label for flag, label in _MODIFIER_LABELS if shortcut.modifiers & flag), _key_label(shortcut.virtual_key)])


def _functions():
    user32 = ctypes.WinDLL("user32", use_last_error=True)
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    user32.RegisterHotKey.argtypes = [wintypes.HWND, ctypes.c_int, ctypes.c_uint, ctypes.c_uint]
    user32.RegisterHotKey.restype = wintypes.BOOL
    user32.UnregisterHotKey.argtypes = [wintypes.HWND, ctypes.c_int]
    user32.UnregisterHotKey.restype = wintypes.BOOL
    user32.PeekMessageW.argtypes = [ctypes.POINTER(wintypes.MSG), wintypes.HWND, ctypes.c_uint, ctypes.c_uint, ctypes.c_uint]
    user32.PeekMessageW.restype = wintypes.BOOL
    user32.GetMessageW.argtypes = [ctypes.POINTER(wintypes.MSG), wintypes.HWND, ctypes.c_uint, ctypes.c_uint]
    user32.GetMessageW.restype = wintypes.BOOL
    user32.PostThreadMessageW.argtypes = [wintypes.DWORD, ctypes.c_uint, wintypes.WPARAM, wintypes.LPARAM]
    user32.PostThreadMessageW.restype = wintypes.BOOL
    kernel32.GetCurrentThreadId.argtypes = []
    kernel32.GetCurrentThreadId.restype = wintypes.DWORD
    kernel32.CreateMutexW.argtypes = [ctypes.c_void_p, wintypes.BOOL, wintypes.LPCWSTR]
    kernel32.CreateMutexW.restype = wintypes.HANDLE
    kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
    kernel32.CloseHandle.restype = wintypes.BOOL
    return user32, kernel32


class SingleInstance:
    """Hold a per-session named mutex until tray shutdown finishes."""

    def __init__(self, name: str = r"Local\ClipSpeak", *, _kernel32=None, _last_error=ctypes.get_last_error):
        self.name = name
        self._kernel32 = _kernel32
        self._last_error = _last_error
        self._handle = None

    def acquire(self) -> bool:
        if self._handle:
            return True
        kernel32 = self._kernel32 or _functions()[1]
        handle = kernel32.CreateMutexW(None, False, self.name)
        if not handle:
            raise OSError(self._last_error(), "Could not create ClipSpeak instance mutex")
        if self._last_error() == ERROR_ALREADY_EXISTS:
            kernel32.CloseHandle(handle)
            return False
        self._kernel32 = kernel32
        self._handle = handle
        return True

    def close(self) -> None:
        handle, self._handle = self._handle, None
        if handle:
            self._kernel32.CloseHandle(handle)


class HotkeyConflictError(RuntimeError):
    """The selected global shortcut is reserved or owned by another app."""


class HotkeyThread:
    """Receive a global shortcut on a dedicated Win32 message-loop thread."""

    def __init__(
        self,
        callback: Callable[[], None],
        *,
        error_callback: Callable[[Exception], None] | None = None,
        _user32=None,
        _kernel32=None,
    ):
        self._callback = callback
        self._error_callback = error_callback
        self._user32 = _user32
        self._kernel32 = _kernel32
        self._commands: deque[tuple[str, Shortcut | None, threading.Event, list]] = deque()
        self._commands_lock = threading.Lock()
        self._ready = threading.Event()
        self._thread: threading.Thread | None = None
        self._thread_id: int | None = None
        self._shortcut: Shortcut | None = None
        self._hotkey_id = 0
        self._lock = threading.Lock()
        self._closed = False
        self._closing = False
        self._loop_error: Exception | None = None

    def start(self, shortcut: Shortcut) -> None:
        with self._lock:
            if self._closed:
                raise RuntimeError("Hotkey listener is closed")
            if self._thread is None:
                if self._user32 is None or self._kernel32 is None:
                    self._user32, self._kernel32 = _functions()
                self._thread = threading.Thread(target=self._run, name="ClipSpeakHotkey", daemon=True)
                self._thread.start()
        if not self._ready.wait(timeout=5):
            raise RuntimeError("Hotkey message loop did not start")
        if self._loop_error is not None:
            raise RuntimeError("Hotkey message loop failed to start") from self._loop_error
        self.replace(shortcut)

    def replace(self, shortcut: Shortcut) -> None:
        if not isinstance(shortcut, Shortcut):
            raise TypeError("shortcut must be a Shortcut")
        if not self._request("replace", shortcut):
            raise HotkeyConflictError(f"{format_shortcut(shortcut)} is already in use or reserved by Windows")

    def _request(self, action: str, shortcut: Shortcut | None = None, *, timeout: float = 5) -> bool:
        with self._lock:
            if self._closed or (self._closing and action != "close") or self._thread is None or self._thread_id is None:
                raise RuntimeError("Hotkey listener is not running") from self._loop_error
            done = threading.Event()
            result: list = []
            command = (action, shortcut, done, result)
            with self._commands_lock:
                self._commands.append(command)
                if not self._user32.PostThreadMessageW(self._thread_id, WM_APP_COMMAND, 0, 0):
                    self._commands.pop()
                    raise OSError(ctypes.get_last_error(), "Could not signal hotkey thread")
        if not done.wait(timeout=timeout):
            with self._commands_lock:
                if command in self._commands:
                    self._commands.remove(command)
                    raise RuntimeError("Hotkey message loop did not respond")
            # The native call has started. Wait for its actual result so a
            # replacement cannot take effect after its caller reports failure.
            done.wait()
        if isinstance(result[0], BaseException):
            raise result[0]
        return bool(result[0])

    def _replace_on_thread(self, shortcut: Shortcut) -> bool:
        if shortcut == self._shortcut:
            return True
        new_id = 2 if self._hotkey_id == 1 else 1
        if not self._user32.RegisterHotKey(None, new_id, shortcut.modifiers | MOD_NOREPEAT, shortcut.virtual_key):
            return False
        if self._hotkey_id:
            if not self._user32.UnregisterHotKey(None, self._hotkey_id):
                self._user32.UnregisterHotKey(None, new_id)
                raise OSError(ctypes.get_last_error(), "Could not release previous shortcut")
        self._shortcut = shortcut
        self._hotkey_id = new_id
        return True

    def _run(self) -> None:
        try:
            self._thread_id = int(self._kernel32.GetCurrentThreadId())
            message = wintypes.MSG()
            self._user32.PeekMessageW(ctypes.byref(message), None, 0, 0, 0)
            self._ready.set()
            while True:
                status = self._user32.GetMessageW(ctypes.byref(message), None, 0, 0)
                if status <= 0:
                    if status < 0:
                        raise OSError(ctypes.get_last_error(), "Hotkey message loop failed")
                    raise RuntimeError("Hotkey message loop stopped unexpectedly")
                if message.message == WM_HOTKEY and message.wParam == self._hotkey_id:
                    try:
                        self._callback()
                    except Exception as exc:
                        if self._error_callback is not None:
                            try:
                                self._error_callback(exc)
                            except Exception:
                                pass
                elif message.message == WM_APP_COMMAND:
                    while True:
                        with self._commands_lock:
                            if not self._commands:
                                break
                            action, shortcut, done, result = self._commands.popleft()
                        try:
                            result.append(self._replace_on_thread(shortcut) if action == "replace" else True)
                            if action == "close":
                                return
                        except Exception as exc:
                            result.append(exc)
                        finally:
                            done.set()
        except Exception as exc:
            self._loop_error = exc
            if self._error_callback is not None:
                try:
                    self._error_callback(exc)
                except Exception:
                    pass
        finally:
            self._ready.set()
            if self._hotkey_id:
                if not self._user32.UnregisterHotKey(None, self._hotkey_id) and self._error_callback is not None:
                    try:
                        self._error_callback(OSError(ctypes.get_last_error(), "Could not release shortcut"))
                    except Exception:
                        pass
                self._hotkey_id = 0
                self._shortcut = None
            with self._commands_lock:
                while self._commands:
                    _, _, done, result = self._commands.popleft()
                    result.append(self._loop_error or RuntimeError("Hotkey listener stopped"))
                    done.set()
            self._thread_id = None

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            if self._closing:
                return
            self._closing = True
        if self._thread is not None and self._thread.is_alive():
            try:
                self._request("close")
                self._thread.join(timeout=5)
                if self._thread.is_alive():
                    raise RuntimeError("Hotkey message loop did not stop")
            except Exception:
                with self._lock:
                    self._closing = False
                raise
        with self._lock:
            self._closed = True


HotkeyListener = HotkeyThread
