"""Windows tray lifecycle with Tk settings on the main thread."""

from __future__ import annotations

import queue
import threading
import tkinter as tk
from tkinter import messagebox, ttk

import pystray
from PIL import Image, ImageDraw

from .controller import QueueController, SubmissionStatus
from .settings import Settings, apply_settings, load_settings
from .windows import HotkeyListener, SingleInstance, parse_shortcut


def tray_image(state: str) -> Image.Image:
    image = Image.new("RGBA", (64, 64))
    draw = ImageDraw.Draw(image)
    color = {"playing": "#39b86b", "paused": "#edbd45", "error": "#e45a5a"}.get(state, "#6ba9e8")
    draw.rounded_rectangle((2, 2, 62, 62), radius=14, fill=color)
    draw.polygon(((12, 25), (23, 25), (36, 14), (36, 50), (23, 39), (12, 39)), fill="white")
    draw.arc((28, 18, 52, 46), -60, 60, fill="white", width=4)
    return image


class TrayApp:
    """Bridge native read commands and tray controls to one controller."""

    def __init__(self, root: tk.Tk, controller: QueueController):
        self.root = root
        self.controller = controller
        self.settings, self.settings_warning = load_settings()
        self.commands: queue.SimpleQueue = queue.SimpleQueue()
        self.closing = False
        self.dialog: tk.Toplevel | None = None
        self.last_state = None
        self.last_error = None
        self.hotkey_error: str | None = None
        self.pending_notice: str | None = None
        self.timers: set[str] = set()
        self.tray_ready = threading.Event()
        self.hotkey = HotkeyListener(self.read, error_callback=lambda exc: self.commands.put(("notify", f"Read shortcut failed ({type(exc).__name__}).")))
        self.icon = pystray.Icon("ClipSpeak", tray_image("idle"), "ClipSpeak: idle", self._menu())
        self.tray_thread = threading.Thread(target=self._run_tray, name="ClipSpeak tray", daemon=True)

    def _menu(self):
        item = pystray.MenuItem
        def command(name):
            return lambda icon, entry: self.commands.put((name, None))
        return pystray.Menu(
            item(lambda entry: "ClipSpeak: " + ("error" if self.hotkey_error else self.controller.snapshot().state), None, enabled=False),
            item("Pause", command("pause"), enabled=lambda entry: not self.controller.snapshot().paused),
            item("Resume", command("resume"), enabled=lambda entry: self.controller.snapshot().paused or bool(self.controller.snapshot().error_type)),
            item("Stop", command("stop")),
            item("Clear Queue", command("clear_queue")),
            pystray.Menu.SEPARATOR,
            item("Settings", command("settings"), default=True),
            item("Quit", command("quit")),
        )

    def later(self, delay, callback):
        timer = None
        def invoke():
            self.timers.discard(timer)
            callback()
        timer = self.root.after(delay, invoke)
        if timer is not None:
            self.timers.add(timer)

    def cancel_timers(self):
        for timer in self.timers:
            self.root.after_cancel(timer)
        self.timers.clear()

    def _run_tray(self):
        def ready(icon):
            icon.visible = True
            self.tray_ready.set()
        try:
            self.icon.run(setup=ready)
        except Exception as exc:
            self.commands.put(("fatal", f"The tray could not start ({type(exc).__name__})."))

    def start(self):
        try:
            self.hotkey.start(parse_shortcut(self.settings.shortcut))
        except Exception as exc:
            self.hotkey_error = str(exc)
        self.tray_thread.start()
        self.later(50, self.poll)
        if self.settings_warning:
            self.later(100, lambda: messagebox.showwarning("ClipSpeak settings", self.settings_warning, parent=self.root))
        if self.hotkey_error:
            self.later(150, lambda: messagebox.showerror("ClipSpeak shortcut", self.hotkey_error + "\nOpen Settings to choose an available shortcut.", parent=self.root))

    def read(self):
        # Immutable settings are captured before the bounded clipboard read.
        settings = self.settings
        result = self.controller.submit_clipboard(speed=settings.speed, volume=settings.volume)
        if result.status not in (SubmissionStatus.ACCEPTED, SubmissionStatus.CLOSED, SubmissionStatus.CANCELLED):
            self.commands.put(("notify", result.reason))

    def poll(self):
        if self.closing:
            # Quit can arrive before pystray creates its native window.
            self.icon.stop()
            if self.controller.join(0) and not self.tray_thread.is_alive():
                self.cancel_timers()
                self.root.destroy()
            else:
                self.later(50, self.poll)
            return
        for _ in range(100):
            try:
                name, value = self.commands.get_nowait()
            except queue.Empty:
                break
            if name == "quit":
                self.quit()
                return
            if name == "settings":
                self.show_settings()
            elif name == "notify":
                if not self.tray_ready.is_set():
                    self.pending_notice = value
                else:
                    self.icon.notify(value, "ClipSpeak")
            elif name == "fatal":
                messagebox.showerror("ClipSpeak", value, parent=self.root)
                self.quit()
                return
            else:
                getattr(self.controller, name)()
        snapshot = self.controller.snapshot()
        if not self.tray_ready.is_set():
            self.later(50, self.poll)
            return
        if self.pending_notice:
            self.icon.notify(self.pending_notice, "ClipSpeak")
            self.pending_notice = None
        state = "error" if self.hotkey_error else snapshot.state
        if state != self.last_state:
            self.last_state = state
            self.icon.title = "ClipSpeak: " + state
            self.icon.icon = tray_image(state)
            self.icon.update_menu()
        error = (snapshot.error_type, snapshot.error_message)
        if snapshot.error_type and error != self.last_error:
            self.icon.notify(snapshot.error_message or "Reading failed. Use Resume to retry.", "ClipSpeak")
        self.last_error = error
        self.later(50, self.poll)

    def show_settings(self):
        if self.dialog is not None and self.dialog.winfo_exists():
            self.dialog.lift()
            self.dialog.focus_force()
            return
        dialog = self.dialog = tk.Toplevel(self.root)
        dialog.title("ClipSpeak Settings")
        dialog.resizable(False, False)
        frame = ttk.Frame(dialog, padding=18)
        frame.grid()
        shortcut = tk.StringVar(value=self.settings.shortcut)
        speed = tk.StringVar(value=str(self.settings.speed))
        volume = tk.StringVar(value=str(round(self.settings.volume * 100, 3)))
        startup = tk.BooleanVar(value=self.settings.startup)
        for row, (label, variable) in enumerate((("Read shortcut", shortcut), ("Speed (1x to 2x)", speed), ("Volume (0 to 100%)", volume))):
            ttk.Label(frame, text=label).grid(row=row, column=0, sticky="w", pady=6)
            ttk.Entry(frame, textvariable=variable, width=25).grid(row=row, column=1, padx=(16, 0), pady=6)
        ttk.Label(frame, text="Example: Ctrl+Alt+A or Ctrl+Shift+F8").grid(row=3, columnspan=2, sticky="w", pady=(0, 10))
        ttk.Checkbutton(frame, text="Start with Windows (preference only)", variable=startup).grid(row=4, columnspan=2, sticky="w")
        ttk.Label(frame, text="Windows startup integration arrives in Phase 5.\nChanges apply to future submissions.").grid(row=5, columnspan=2, sticky="w", pady=10)
        error = tk.StringVar()
        ttk.Label(frame, textvariable=error, foreground="#ad2222", wraplength=390).grid(row=6, columnspan=2, sticky="w", pady=6)

        def save():
            try:
                proposed = Settings(shortcut=shortcut.get(), speed=float(speed.get()), volume=float(volume.get()) / 100, startup=startup.get())
                self.settings = apply_settings(self.settings, proposed, hotkey=self.hotkey)
            except Exception as exc:
                error.set(str(exc))
                return
            self.hotkey_error = None
            dialog.destroy()
            self.dialog = None

        buttons = ttk.Frame(frame)
        buttons.grid(row=7, columnspan=2, sticky="e", pady=(10, 0))
        ttk.Button(buttons, text="Cancel", command=dialog.destroy).pack(side="left", padx=6)
        ttk.Button(buttons, text="Save", command=save).pack(side="left")
        dialog.bind("<Escape>", lambda event: dialog.destroy())
        dialog.bind("<Return>", lambda event: save())
        dialog.lift()
        dialog.focus_force()

    def quit(self):
        if self.closing:
            return
        self.closing = True
        self.controller.close()
        self.hotkey.close()
        self.icon.stop()
        if self.dialog is not None and self.dialog.winfo_exists():
            self.dialog.destroy()
        self.cancel_timers()
        self.later(50, self.poll)


def run_app(voice_factory) -> int:
    """Hold the instance guard until all app-owned work has shut down."""
    instance = SingleInstance()
    root = None
    controller = None
    app = None
    try:
        if not instance.acquire():
            root = tk.Tk()
            root.withdraw()
            messagebox.showinfo("ClipSpeak", "ClipSpeak is already running. Open its tray menu for Settings or Quit.", parent=root)
            return 0
        root = tk.Tk()
        root.withdraw()
        controller = QueueController(voice_factory=voice_factory)
        app = TrayApp(root, controller)
        app.start()
        root.mainloop()
        return 0
    finally:
        cleanup_errors = []
        def cleanup(action):
            try:
                action()
            except Exception as exc:
                cleanup_errors.append(exc)
        if controller is not None:
            cleanup(controller.close)
        if app is not None:
            cleanup(app.cancel_timers)
            cleanup(app.hotkey.close)
            cleanup(app.icon.stop)
            if app.tray_thread.ident is not None:
                # Stop again once native tray startup has completed.
                app.tray_ready.wait(5)
                cleanup(app.icon.stop)
                cleanup(lambda: app.tray_thread.join(5))
                if app.tray_thread.is_alive():
                    cleanup_errors.append(RuntimeError("Tray did not shut down"))
        if controller is not None:
            cleanup(controller.join)
        if root is not None:
            def destroy_root():
                try:
                    root.destroy()
                except tk.TclError:
                    pass
            cleanup(destroy_root)
        cleanup(instance.close)
        if cleanup_errors:
            raise RuntimeError("ClipSpeak cleanup failed; all cleanup steps were attempted") from cleanup_errors[0]
