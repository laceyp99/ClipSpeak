"""Global shortcut and instance lifetime checks without pressing real keys."""

import ctypes
import queue
import threading
from ctypes import wintypes

import pytest

from clipspeak.windows import (
    ERROR_ALREADY_EXISTS,
    MOD_NOREPEAT,
    WM_HOTKEY,
    HotkeyConflictError,
    HotkeyThread,
    SingleInstance,
    format_shortcut,
    parse_shortcut,
)


@pytest.mark.parametrize("entry,expected", [
    ("ctrl+alt+a", "Ctrl+Alt+A"),
    (" Shift + Ctrl + F24 ", "Ctrl+Shift+F24"),
    ("Win+0", "Win+0"),
])
def test_shortcut_canonicalization(entry, expected):
    assert format_shortcut(parse_shortcut(entry)) == expected


@pytest.mark.parametrize("entry", ["A", "Ctrl+Ctrl+A", "Ctrl+", "Ctrl+F25", "Ctrl+Space", "Ctrl+Alt", "Ctrl++A"])
def test_shortcut_rejects_invalid_input(entry):
    with pytest.raises(ValueError):
        parse_shortcut(entry)


class FakeUser32:
    def __init__(self):
        self.messages = queue.Queue()
        self.registrations = {}
        self.register_calls = []
        self.unregister_calls = []
        self.blocked = set()
        self.thread_ids = []
        self.fail_post = False
        self.fail_unregister_id = None

    def PeekMessageW(self, pointer, *_):
        return 0

    def PostThreadMessageW(self, thread_id, message, wparam, lparam):
        if self.fail_post:
            return 0
        self.messages.put((message, wparam))
        return 1

    def GetMessageW(self, pointer, *_):
        message, wparam = self.messages.get(timeout=2)
        if message == -1:
            return -1
        msg = ctypes.cast(pointer, ctypes.POINTER(wintypes.MSG)).contents
        msg.message = message
        msg.wParam = wparam
        return 1

    def RegisterHotKey(self, window, identifier, modifiers, key):
        self.thread_ids.append(threading.get_ident())
        self.register_calls.append((identifier, modifiers, key))
        if (modifiers, key) in self.blocked:
            return 0
        self.registrations[identifier] = (modifiers, key)
        return 1

    def UnregisterHotKey(self, window, identifier):
        self.unregister_calls.append(identifier)
        if identifier == self.fail_unregister_id:
            return 0
        self.registrations.pop(identifier)
        return 1


class FakeKernel32:
    def GetCurrentThreadId(self):
        return 17


def test_conflicting_replacement_keeps_previous_hotkey_and_close_unregisters():
    user32 = FakeUser32()
    called = threading.Event()
    listener = HotkeyThread(called.set, _user32=user32, _kernel32=FakeKernel32())
    first = parse_shortcut("Ctrl+Alt+A")
    second = parse_shortcut("Ctrl+Shift+F2")
    listener.start(first)
    try:
        first_id = next(iter(user32.registrations))
        assert user32.registrations[first_id] == (first.modifiers | MOD_NOREPEAT, first.virtual_key)
        assert user32.thread_ids == [listener._thread.ident]
        user32.blocked.add((second.modifiers | MOD_NOREPEAT, second.virtual_key))
        with pytest.raises(HotkeyConflictError, match="Ctrl\\+Shift\\+F2"):
            listener.replace(second)
        assert list(user32.registrations) == [first_id]
        user32.messages.put((WM_HOTKEY, first_id))
        assert called.wait(1)
        user32.blocked.clear()
        listener.replace(second)
        assert first_id in user32.unregister_calls
        assert next(iter(user32.registrations.values())) == (second.modifiers | MOD_NOREPEAT, second.virtual_key)
    finally:
        listener.close()
    assert user32.registrations == {}
    listener.close()


def test_start_conflict_can_be_resolved_without_restarting_thread():
    user32 = FakeUser32()
    first = parse_shortcut("Ctrl+Alt+A")
    user32.blocked.add((first.modifiers | MOD_NOREPEAT, first.virtual_key))
    listener = HotkeyThread(lambda: None, _user32=user32, _kernel32=FakeKernel32())
    try:
        with pytest.raises(HotkeyConflictError):
            listener.start(first)
        listener.replace(parse_shortcut("Ctrl+Alt+B"))
        assert len(user32.registrations) == 1
    finally:
        listener.close()


def test_failed_unregistration_rolls_back_new_mapping():
    user32 = FakeUser32()
    listener = HotkeyThread(lambda: None, _user32=user32, _kernel32=FakeKernel32())
    first = parse_shortcut("Ctrl+Alt+A")
    listener.start(first)
    try:
        old_id = next(iter(user32.registrations))
        user32.fail_unregister_id = old_id
        with pytest.raises(OSError, match="previous shortcut"):
            listener.replace(parse_shortcut("Ctrl+Alt+B"))
        assert list(user32.registrations) == [old_id]
        user32.fail_unregister_id = None
    finally:
        user32.fail_unregister_id = None
        listener.close()


def test_callback_error_is_reported_and_loop_continues():
    user32 = FakeUser32()
    errors = []
    delivered = threading.Event()

    def callback():
        delivered.set()
        raise ValueError("submission failed")

    listener = HotkeyThread(callback, error_callback=errors.append, _user32=user32, _kernel32=FakeKernel32())
    listener.start(parse_shortcut("Ctrl+Alt+A"))
    try:
        hotkey_id = next(iter(user32.registrations))
        user32.messages.put((WM_HOTKEY, hotkey_id))
        assert delivered.wait(1)
        listener.replace(parse_shortcut("Ctrl+Alt+B"))
        assert len(errors) == 1
        assert isinstance(errors[0], ValueError)
    finally:
        listener.close()


def test_timed_out_request_cannot_replace_hotkey_later():
    user32 = FakeUser32()
    callback_entered = threading.Event()
    callback_release = threading.Event()

    def callback():
        callback_entered.set()
        assert callback_release.wait(2)

    listener = HotkeyThread(callback, _user32=user32, _kernel32=FakeKernel32())
    first = parse_shortcut("Ctrl+Alt+A")
    listener.start(first)
    try:
        user32.messages.put((WM_HOTKEY, next(iter(user32.registrations))))
        assert callback_entered.wait(1)
        with pytest.raises(RuntimeError, match="did not respond"):
            listener._request("replace", parse_shortcut("Ctrl+Alt+B"), timeout=0.01)
        callback_release.set()
        listener.replace(first)
        assert len(user32.register_calls) == 1
        assert next(iter(user32.registrations.values()))[-1] == first.virtual_key
    finally:
        callback_release.set()
        listener.close()


def test_post_failure_does_not_leave_a_queued_replacement():
    user32 = FakeUser32()
    listener = HotkeyThread(lambda: None, _user32=user32, _kernel32=FakeKernel32())
    first = parse_shortcut("Ctrl+Alt+A")
    listener.start(first)
    try:
        user32.fail_post = True
        with pytest.raises(OSError, match="Could not signal"):
            listener.replace(parse_shortcut("Ctrl+Alt+B"))
        user32.fail_post = False
        listener.replace(first)
        assert len(user32.register_calls) == 1
    finally:
        user32.fail_post = False
        listener.close()


def test_message_loop_failure_is_reported_and_rejects_future_replacement():
    user32 = FakeUser32()
    errors = []
    listener = HotkeyThread(lambda: None, error_callback=errors.append, _user32=user32, _kernel32=FakeKernel32())
    listener.start(parse_shortcut("Ctrl+Alt+A"))
    user32.messages.put((-1, 0))
    listener._thread.join(timeout=1)
    assert not listener._thread.is_alive()
    assert errors and isinstance(errors[0], OSError)
    assert user32.registrations == {}
    with pytest.raises(RuntimeError, match="not running"):
        listener.replace(parse_shortcut("Ctrl+Alt+B"))
    listener.close()


class FakeMutex:
    def __init__(self):
        self.handles = []
        self.closed = []

    def CreateMutexW(self, attributes, owner, name):
        self.handles.append(name)
        return len(self.handles)

    def CloseHandle(self, handle):
        self.closed.append(handle)
        return 1


def test_single_instance_closes_duplicate_handle_and_owned_handle():
    native = FakeMutex()
    first = SingleInstance(_kernel32=native, _last_error=lambda: 0)
    duplicate = SingleInstance(_kernel32=native, _last_error=lambda: ERROR_ALREADY_EXISTS)
    assert first.acquire()
    assert first.acquire()
    assert not duplicate.acquire()
    assert native.closed == [2]
    first.close()
    first.close()
    assert native.closed == [2, 1]
