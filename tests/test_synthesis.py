"""Focused checks for resident model and one-way chunk delivery."""

from pathlib import Path
from types import SimpleNamespace

import pytest

from clipspeak.synthesis import ResidentVoice


def _model(tmp_path: Path) -> Path:
    path = tmp_path / "voice.onnx"
    path.touch()
    Path(f"{path}.json").write_text("{}", encoding="utf-8")
    return path


def test_load_once_and_reuse(tmp_path, monkeypatch):
    calls = []
    chunk = object()

    class FakeVoice:
        session = SimpleNamespace(get_providers=lambda: ["CUDAExecutionProvider"])

        def synthesize(self, text):
            yield chunk

    monkeypatch.setattr("clipspeak.synthesis.PiperVoice.load", lambda path, use_cuda: calls.append(use_cuda) or FakeVoice())
    voice = ResidentVoice(_model(tmp_path))
    assert list(voice.synthesize("first")) == [chunk]
    assert list(voice.synthesize("second")) == [chunk]
    assert calls == [True]


def test_silent_cuda_session_falls_back_to_cpu(tmp_path, monkeypatch):
    calls = []

    def load(path, use_cuda):
        calls.append(use_cuda)
        return SimpleNamespace(session=SimpleNamespace(get_providers=lambda: ["CPUExecutionProvider"]), synthesize=lambda text: iter(["cpu"]))

    monkeypatch.setattr("clipspeak.synthesis.PiperVoice.load", load)
    voice = ResidentVoice(_model(tmp_path))
    assert voice.provider == "CPUExecutionProvider"
    assert list(voice.synthesize("text")) == ["cpu"]
    assert calls == [True, False]


def test_inference_failure_before_first_chunk_falls_back_once(tmp_path, monkeypatch):
    calls = []

    def load(path, use_cuda):
        calls.append(use_cuda)
        if use_cuda:
            def fail(text):
                raise RuntimeError("private text")
                yield
            return SimpleNamespace(session=SimpleNamespace(get_providers=lambda: ["CUDAExecutionProvider"]), synthesize=fail)
        return SimpleNamespace(synthesize=lambda text: iter(["cpu"]))

    monkeypatch.setattr("clipspeak.synthesis.PiperVoice.load", load)
    voice = ResidentVoice(_model(tmp_path))
    assert list(voice.synthesize("private text")) == ["cpu"]
    assert voice.provider == "CPUExecutionProvider"
    assert "private text" not in voice.fallback_reason
    assert calls == [True, False]


def test_failure_after_chunk_never_replays(tmp_path, monkeypatch):
    calls = []

    def load(path, use_cuda):
        calls.append(use_cuda)

        def generate(text):
            yield "already delivered"
            raise RuntimeError("late failure")

        return SimpleNamespace(session=SimpleNamespace(get_providers=lambda: ["CUDAExecutionProvider"]), synthesize=generate)

    monkeypatch.setattr("clipspeak.synthesis.PiperVoice.load", load)
    voice = ResidentVoice(_model(tmp_path))
    chunks = voice.synthesize("text")
    assert next(chunks) == "already delivered"
    with pytest.raises(RuntimeError, match="late failure"):
        next(chunks)
    assert calls == [True]


def test_missing_model_is_actionable(tmp_path):
    with pytest.raises(FileNotFoundError, match="Download the Ryan ONNX model"):
        ResidentVoice(tmp_path / "missing.onnx")
