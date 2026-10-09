"""Validated per-user settings for the tray application."""

from __future__ import annotations

import json
import math
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from .windows import Shortcut, format_shortcut, parse_shortcut


@dataclass(frozen=True)
class Settings:
    shortcut: str = "Ctrl+Alt+A"
    speed: float = 1.5
    volume: float = 0.225
    startup: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "shortcut", format_shortcut(parse_shortcut(self.shortcut)))
        for name, minimum, maximum in (("speed", 1.0, 2.0), ("volume", 0.0, 1.0)):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"{name} must be a number")
            if not math.isfinite(value) or not minimum <= value <= maximum:
                raise ValueError(f"{name} must be between {minimum:g} and {maximum:g}")
            object.__setattr__(self, name, float(value))
        if not isinstance(self.startup, bool):
            raise ValueError("startup must be true or false")


def settings_path() -> Path:
    """Return ClipSpeak's per-user configuration path on Windows."""
    local_app_data = os.environ.get("LOCALAPPDATA")
    root = Path(local_app_data) if local_app_data else Path.home() / "AppData" / "Local"
    return root / "ClipSpeak" / "settings.json"


def load_settings(path: Path | None = None) -> tuple[Settings, str | None]:
    """Read settings or return defaults and a user-facing warning on failure."""
    target = path if path is not None else settings_path()
    try:
        data = json.loads(target.read_text(encoding="utf-8"))
        if not isinstance(data, dict) or set(data) != {"shortcut", "speed", "volume", "startup"}:
            raise ValueError("settings fields are missing or unknown")
        return Settings(**data), None
    except FileNotFoundError:
        return Settings(), None
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError, TypeError) as exc:
        return Settings(), f"Settings could not be loaded ({type(exc).__name__}). Defaults are in use."


def save_settings(settings: Settings, path: Path | None = None) -> None:
    """Replace the JSON file atomically after writing a complete validated document."""
    if not isinstance(settings, Settings):
        raise TypeError("settings must be a Settings instance")
    # Revalidate in case frozen fields were modified through unsupported reflection.
    settings = Settings(settings.shortcut, settings.speed, settings.volume, settings.startup)
    target = path if path is not None else settings_path()
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=target.parent,
            prefix=".settings-", suffix=".tmp", delete=False,
        ) as stream:
            temporary = Path(stream.name)
            json.dump(
                {"shortcut": settings.shortcut, "speed": settings.speed,
                 "volume": settings.volume, "startup": settings.startup},
                stream, indent=2,
            )
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, target)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


class HotkeyRegistration(Protocol):
    def replace(self, shortcut: Shortcut) -> None: ...


class SettingsApplyError(RuntimeError):
    """A settings save failed and the prior shortcut could not be restored."""


def apply_settings(
    current: Settings,
    proposed: Settings,
    *,
    hotkey: HotkeyRegistration,
    path: Path | None = None,
) -> Settings:
    """Replace the shortcut before saving; restore it if persistence fails."""
    if not isinstance(current, Settings) or not isinstance(proposed, Settings):
        raise TypeError("current and proposed must be Settings instances")
    changed = current.shortcut != proposed.shortcut
    # Also retry an unchanged shortcut after an initial registration conflict.
    hotkey.replace(parse_shortcut(proposed.shortcut))
    try:
        save_settings(proposed, path)
    except Exception as save_error:
        if changed:
            try:
                hotkey.replace(parse_shortcut(current.shortcut))
            except Exception as rollback_error:
                raise SettingsApplyError(
                    "Settings could not be saved, and the previous shortcut could not be restored"
                ) from rollback_error
        raise save_error
    return proposed
