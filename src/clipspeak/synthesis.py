"""Resident Piper voice with a verified CUDA preference and CPU fallback."""

import ctypes
import os
from pathlib import Path
from typing import Iterator

from piper import PiperVoice
from piper.voice import AudioChunk


DEFAULT_VOICE = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local")) / "ClipSpeak" / "voices" / "en_US-ryan-high.onnx"

# Keep Windows loader handles alive while ONNX Runtime uses the libraries.
_dll_handles: list[object] = []
_dll_directories: set[Path] = set()


def add_cuda_dll_directory(path: Path) -> None:
    """Expose an existing CUDA/cuDNN directory to this process's native loader."""
    directory = path.resolve(strict=True)
    if not directory.is_dir():
        raise NotADirectoryError(directory)
    if directory in _dll_directories:
        return
    if os.name != "nt":
        raise RuntimeError("CUDA DLL directory is supported on Windows only")
    # cuDNN loads companion DLLs lazily through PATH. Keep this process-only.
    os.environ["PATH"] = str(directory) + os.pathsep + os.environ.get("PATH", "")
    _dll_handles.append(os.add_dll_directory(str(directory)))
    for name in ("cudart64_12.dll", "cublasLt64_12.dll", "cublas64_12.dll", "cudnn64_9.dll"):
        _dll_handles.append(ctypes.WinDLL(str(directory / name)))
    _dll_directories.add(directory)


class ResidentVoice:
    """Keep one voice resident; yield Piper's normal-speed sentence chunks."""

    def __init__(self, model_path: Path, *, prefer_cuda: bool = True, cuda_dll_dir: Path | None = None) -> None:
        model_path = Path(model_path)
        config_path = Path(f"{model_path}.json")
        for required in (model_path, config_path):
            if not required.is_file():
                raise FileNotFoundError(
                    f"Voice file missing: {required}. Download the Ryan ONNX model and its .onnx.json config into the voices directory."
                )
        self.model_path = model_path
        self.fallback_reason: str | None = None
        self.provider = "CPUExecutionProvider"
        if prefer_cuda:
            try:
                if cuda_dll_dir is not None:
                    add_cuda_dll_directory(cuda_dll_dir)
                self.voice = PiperVoice.load(model_path, use_cuda=True)
                if "CUDAExecutionProvider" not in self.voice.session.get_providers():
                    raise RuntimeError("CUDA provider was not active in the loaded voice session")
                self.provider = "CUDAExecutionProvider"
            except Exception as exc:
                self.fallback_reason = f"CUDA voice load failed ({type(exc).__name__})"
                self.voice = PiperVoice.load(model_path, use_cuda=False)
        else:
            self.voice = PiperVoice.load(model_path, use_cuda=False)

    def synthesize(self, text: str) -> Iterator[AudioChunk]:
        """Retry on CPU only before any chunk escapes to the caller."""
        emitted = False
        try:
            for chunk in self.voice.synthesize(text):
                emitted = True
                yield chunk
        except Exception as exc:
            if emitted or self.provider != "CUDAExecutionProvider":
                raise
            self.fallback_reason = f"CUDA inference failed before first chunk ({type(exc).__name__})"
            self.voice = PiperVoice.load(self.model_path, use_cuda=False)
            self.provider = "CPUExecutionProvider"
            yield from self.voice.synthesize(text)
