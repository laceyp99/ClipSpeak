"""Tray-to-controller integration and real Tk settings transactions."""

from dataclasses import replace
import gc
from types import SimpleNamespace

import pytest

from clipspeak import app
from clipspeak.controller import SubmissionResult, SubmissionStatus
from clipspeak.settings import Settings, apply_settings, load_settings
from clipspeak.windows import parse_shortcut


@pytest.fixture(autouse=True)
def collect_ui_objects_on_main_thread():
    yield
    # Destroyed Tk widgets can retain cycles. Collect them before later worker
    # tests trigger garbage collection on another thread.
    gc.collect()


class Root:
    def __init__(self):
        self.destroyed = False
    def after(self, delay, callback):
        pass
    def destroy(self):
        self.destroyed = True


class Icon:
    def __init__(self, name, image, title, menu):
        self.menu = menu
        self.title = title
        self.notifications = []
        self.stopped = 0
    def notify(self, message, title):
        self.notifications.append(message)
    def update_menu(self):
        pass
    def stop(self):
        self.stopped += 1


class Hotkey:
    def __init__(self, callback, **kwargs):
        self.callback = callback
        self.shortcut = parse_shortcut(Settings().shortcut)
        self.reject = None
        self.closed = False
    def replace(self, shortcut):
        if shortcut == self.reject:
            raise RuntimeError("Shortcut is in use")
        self.shortcut = shortcut
    def close(self):
        self.closed = True


class Controller:
    def __init__(self):
        self.calls = []
        self.state = SimpleNamespace(state="idle", paused=False, error_type=None, error_message=None)
        self.finished = False
    def submit_clipboard(self, **kwargs):
        self.calls.append(kwargs)
        return SubmissionResult(SubmissionStatus.ACCEPTED)
    def snapshot(self):
        return self.state
    def pause(self):
        self.calls.append("pause")
    def stop(self):
        self.calls.append("stop")
    def close(self):
        self.calls.append("close")
    def join(self, timeout):
        return self.finished


@pytest.fixture
def build(monkeypatch):
    monkeypatch.setattr(app.pystray, "Icon", Icon)
    monkeypatch.setattr(app, "HotkeyListener", Hotkey)
    monkeypatch.setattr(app, "load_settings", lambda: (Settings(), None))
    def create(root=None):
        instance = app.TrayApp(root or Root(), Controller())
        instance.tray_ready.set()
        return instance
    return create


def test_hotkey_reads_on_command_with_future_settings_and_no_success_notice(build):
    instance = build()
    instance.hotkey.callback()
    instance.settings = replace(instance.settings, speed=2, volume=0.4)
    instance.hotkey.callback()
    instance.poll()
    assert instance.controller.calls == [{"speed": 1.5, "volume": 1}, {"speed": 2, "volume": 0.4}]
    assert not instance.icon.notifications


def test_tray_callbacks_dispatch_on_ui_thread_and_errors_notify_once(build):
    instance = build()
    entries = list(instance.icon.menu.items)
    entries[1](instance.icon)
    assert not instance.controller.calls
    instance.poll()
    assert instance.controller.calls == ["pause"]
    instance.controller.state = SimpleNamespace(state="error", paused=False, error_type="RuntimeError", error_message="Resume retries from the beginning.")
    instance.poll()
    instance.poll()
    assert instance.icon.notifications == ["Resume retries from the beginning."]


def test_quit_waits_for_worker_and_stops_tray_even_before_ready(build):
    instance = build()
    instance.tray_ready.clear()
    instance.quit()
    instance.quit()
    instance.poll()
    assert instance.controller.calls == ["close"]
    assert instance.hotkey.closed
    assert not instance.root.destroyed
    instance.controller.finished = True
    instance.poll()
    assert instance.root.destroyed
    assert instance.icon.stopped >= 2


def test_tray_startup_failure_is_not_blocked_by_earlier_notice(build, monkeypatch):
    instance = build()
    instance.tray_ready.clear()
    messages = []
    monkeypatch.setattr(app.messagebox, "showerror", lambda title, text, **kwargs: messages.append(text))
    instance.commands.put(("notify", "Clipboard unavailable"))
    instance.commands.put(("fatal", "Tray failed"))
    instance.poll()
    assert messages == ["Tray failed"]
    assert instance.closing
    assert instance.controller.calls == ["close"]


def test_cleanup_failure_still_releases_worker_root_and_instance(monkeypatch):
    calls = []
    class Guard:
        def acquire(self):
            return True
        def close(self):
            calls.append("mutex")
    class UI:
        def withdraw(self):
            pass
        def mainloop(self):
            pass
        def destroy(self):
            calls.append("root")
    class Worker:
        def __init__(self, **kwargs):
            pass
        def close(self):
            calls.append("worker stop")
        def join(self):
            calls.append("worker join")
    def failed_hotkey_close():
        raise OSError("native cleanup failed")
    class App:
        def __init__(self, root, controller):
            self.hotkey = SimpleNamespace(close=failed_hotkey_close)
            self.icon = SimpleNamespace(stop=lambda: calls.append("tray"))
            self.tray_thread = SimpleNamespace(ident=None)
        def start(self):
            pass
        def cancel_timers(self):
            pass
    monkeypatch.setattr(app, "SingleInstance", Guard)
    monkeypatch.setattr(app.tk, "Tk", UI)
    monkeypatch.setattr(app, "QueueController", Worker)
    monkeypatch.setattr(app, "TrayApp", App)
    with pytest.raises(RuntimeError, match="all cleanup steps"):
        app.run_app(lambda: object())
    assert calls == ["worker stop", "tray", "worker join", "root", "mutex"]


def test_settings_dialog_conflict_preserves_disk_then_success_survives_reload(build, tmp_path, monkeypatch):
    root = app.tk.Tk()
    root.withdraw()
    instance = build(root)
    path = tmp_path / "settings.json"
    monkeypatch.setattr(app, "apply_settings", lambda current, proposed, **kwargs: apply_settings(current, proposed, path=path, **kwargs))
    try:
        instance.show_settings()
        frame = instance.dialog.winfo_children()[0]
        entries = [widget for widget in frame.winfo_children() if isinstance(widget, app.ttk.Entry)]
        buttons = [widget for widget in frame.winfo_children() if isinstance(widget, app.ttk.Frame)][0]
        save = buttons.winfo_children()[1]
        entries[0].delete(0, "end")
        entries[0].insert(0, "Ctrl+Shift+F8")
        entries[1].delete(0, "end")
        entries[1].insert(0, "1.75")
        instance.hotkey.reject = parse_shortcut("Ctrl+Shift+F8")
        save.invoke()
        assert instance.dialog.winfo_exists()
        assert instance.settings == Settings()
        assert not path.exists()
        instance.hotkey.reject = None
        save.invoke()
        assert instance.dialog is None
        assert load_settings(path) == (Settings("Ctrl+Shift+F8", 1.75), None)
        instance.hotkey.callback()
        assert instance.controller.calls[-1] == {"speed": 1.75, "volume": 1}
    finally:
        root.destroy()
