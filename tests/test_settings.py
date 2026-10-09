import json
from dataclasses import replace

import pytest

from clipspeak.settings import Settings, SettingsApplyError, apply_settings, load_settings, save_settings
from clipspeak.windows import parse_shortcut


def test_defaults_and_canonical_shortcut(tmp_path):
    settings, warning = load_settings(tmp_path / "missing.json")
    assert settings == Settings("Ctrl+Alt+A", 1.5, 0.225, False)
    assert warning is None
    assert Settings("shift+ctrl+f12").shortcut == "Ctrl+Shift+F12"


@pytest.mark.parametrize(
    "changes",
    [
        {"shortcut": "A"},
        {"shortcut": "Ctrl+Ctrl+A"},
        {"speed": 0.99},
        {"speed": float("nan")},
        {"speed": True},
        {"volume": 1.01},
        {"volume": float("inf")},
        {"volume": "1"},
        {"startup": 1},
    ],
)
def test_invalid_values_are_rejected(changes):
    with pytest.raises(ValueError):
        replace(Settings(), **changes)


def test_round_trip_and_corrupt_file_preserved(tmp_path):
    path = tmp_path / "ClipSpeak" / "settings.json"
    expected = Settings("Ctrl+Shift+9", 2, 0.25, True)
    save_settings(expected, path)
    loaded, warning = load_settings(path)
    assert loaded == expected
    assert warning is None
    assert json.loads(path.read_text(encoding="utf-8"))["speed"] == 2.0
    path.write_text('{"shortcut": "Ctrl+Alt+A", "speed": "private"}', encoding="utf-8")
    before = path.read_bytes()
    loaded, warning = load_settings(path)
    assert loaded == Settings()
    assert "Defaults" in warning
    assert path.read_bytes() == before


class FakeHotkey:
    def __init__(self):
        self.shortcut = parse_shortcut("Ctrl+Alt+A")
        self.calls = []
        self.reject = None

    def replace(self, shortcut):
        self.calls.append(shortcut)
        if shortcut == self.reject:
            raise RuntimeError("shortcut in use")
        self.shortcut = shortcut


def test_apply_updates_shortcut_and_persists_future_playback_settings(tmp_path):
    path = tmp_path / "settings.json"
    original = Settings()
    hotkey = FakeHotkey()
    proposed = Settings("Alt+Shift+8", 1.75, 0.4, True)
    assert apply_settings(original, proposed, hotkey=hotkey, path=path) == proposed
    assert hotkey.shortcut == parse_shortcut(proposed.shortcut)
    assert load_settings(path) == (proposed, None)


def test_conflicting_shortcut_leaves_registration_and_file_unchanged(tmp_path):
    path = tmp_path / "settings.json"
    original = Settings()
    save_settings(original, path)
    before = path.read_bytes()
    hotkey = FakeHotkey()
    proposed = replace(original, shortcut="Ctrl+Shift+Q", speed=2.0)
    hotkey.reject = parse_shortcut(proposed.shortcut)
    with pytest.raises(RuntimeError, match="shortcut in use"):
        apply_settings(original, proposed, hotkey=hotkey, path=path)
    assert hotkey.shortcut == parse_shortcut(original.shortcut)
    assert path.read_bytes() == before


def test_unchanged_shortcut_conflict_cannot_be_saved_as_working(tmp_path):
    hotkey = FakeHotkey()
    hotkey.reject = parse_shortcut(Settings().shortcut)
    path = tmp_path / "settings.json"
    with pytest.raises(RuntimeError, match="shortcut in use"):
        apply_settings(Settings(), Settings(speed=2), hotkey=hotkey, path=path)
    assert not path.exists()


def test_failed_save_restores_previous_shortcut(tmp_path, monkeypatch):
    path = tmp_path / "settings.json"
    original = Settings()
    save_settings(original, path)
    before = path.read_bytes()
    hotkey = FakeHotkey()
    proposed = replace(original, shortcut="Ctrl+Shift+Q")

    def fail_replace(_source, _target):
        raise OSError("disk failure")

    monkeypatch.setattr("clipspeak.settings.os.replace", fail_replace)
    with pytest.raises(OSError, match="disk failure"):
        apply_settings(original, proposed, hotkey=hotkey, path=path)
    assert hotkey.calls == [parse_shortcut(proposed.shortcut), parse_shortcut(original.shortcut)]
    assert hotkey.shortcut == parse_shortcut(original.shortcut)
    assert path.read_bytes() == before
    assert not list(tmp_path.glob(".settings-*.tmp"))


def test_failed_rollback_is_reported(tmp_path, monkeypatch):
    hotkey = FakeHotkey()
    original = Settings()
    proposed = replace(original, shortcut="Ctrl+Shift+Q")
    hotkey.reject = parse_shortcut(original.shortcut)
    monkeypatch.setattr("clipspeak.settings.os.replace", lambda *_: (_ for _ in ()).throw(OSError()))
    with pytest.raises(SettingsApplyError, match="previous shortcut could not be restored"):
        apply_settings(original, proposed, hotkey=hotkey, path=tmp_path / "settings.json")
