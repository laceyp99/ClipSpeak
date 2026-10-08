"""Development diagnostics and resident synthesis benchmark."""

import argparse
import json
import ctypes
import os
import time
from importlib.metadata import version
from pathlib import Path

from .synthesis import DEFAULT_VOICE, ResidentVoice, add_cuda_dll_directory


BENCHMARK_TEXTS = {
    "short": "The quick brown fox jumps over the lazy dog.",
    "long": (
        "Reading and listening together can help me understand a complex passage. "
        "When an unfamiliar term appears, I can slow down and review its meaning. "
        "For familiar material, faster speech helps me move through the text. "
        "This sample measures synthesis without opening an audio device."
    ) * 4,
}


def working_set_bytes() -> int | None:
    """Return Windows process resident memory, if available."""
    if os.name != "nt":
        return None

    class Counters(ctypes.Structure):
        _fields_ = [
            ("cb", ctypes.c_ulong), ("PageFaultCount", ctypes.c_ulong),
            ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
            ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
            ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
            ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t),
        ]

    counters = Counters()
    counters.cb = ctypes.sizeof(counters)
    get_process = ctypes.windll.kernel32.GetCurrentProcess
    get_process.restype = ctypes.c_void_p
    get_memory = ctypes.windll.psapi.GetProcessMemoryInfo
    get_memory.argtypes = [ctypes.c_void_p, ctypes.POINTER(Counters), ctypes.c_ulong]
    get_memory.restype = ctypes.c_int
    process = get_process()
    if not get_memory(process, ctypes.byref(counters), counters.cb):
        raise ctypes.WinError()
    return counters.WorkingSetSize


def benchmark(path: Path, *, prefer_cuda: bool, cuda_dll_dir: Path | None = None) -> dict:
    before_memory = working_set_bytes()
    start = time.perf_counter()
    voice = ResidentVoice(path, prefer_cuda=prefer_cuda, cuda_dll_dir=cuda_dll_dir)
    load_seconds = time.perf_counter() - start
    loaded_memory = working_set_bytes()
    passages = []
    for name, passage in BENCHMARK_TEXTS.items():
        for run in range(2):
            start = time.perf_counter()
            first_chunk_seconds = None
            audio_seconds = 0.0
            chunk_count = 0
            for chunk in voice.synthesize(passage):
                if first_chunk_seconds is None:
                    first_chunk_seconds = time.perf_counter() - start
                audio_seconds += len(chunk.audio_float_array) / chunk.sample_rate
                chunk_count += 1
            generation_seconds = time.perf_counter() - start
            passages.append({
                "passage": name, "run": "first_call" if run == 0 else "repeat",
                "first_chunk_seconds": first_chunk_seconds,
                "generation_seconds": generation_seconds,
                "audio_seconds": audio_seconds,
                "throughput_audio_seconds_per_generation_second": audio_seconds / generation_seconds,
                "chunks": chunk_count,
            })
    return {
        "model": str(path), "provider": voice.provider,
        "fallback_reason": voice.fallback_reason,
        "load_seconds": load_seconds,
        "working_set_bytes": {
            "before_load": before_memory, "after_load": loaded_memory,
            "after_benchmark": working_set_bytes(),
        },
        "passages": passages,
        "note": "Audio generated in memory only; first-chunk time excludes playback and speed processing.",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cuda-dll-dir", type=Path,
        help="Existing directory containing CUDA 12 and cuDNN 9 DLLs",
    )
    parser.add_argument("--benchmark", action="store_true", help="Measure resident synthesis without playback")
    parser.add_argument("--voice", type=Path, default=DEFAULT_VOICE, help="ONNX voice path for benchmark")
    parser.add_argument("--cpu", action="store_true", help="Use CPU for benchmark")
    args = parser.parse_args()
    if args.benchmark:
        print(json.dumps(benchmark(args.voice, prefer_cuda=not args.cpu, cuda_dll_dir=args.cuda_dll_dir), indent=2))
        return 0
    if args.cuda_dll_dir:
        add_cuda_dll_directory(args.cuda_dll_dir)

    import onnxruntime
    import pedalboard
    import piper
    import pystray
    import sounddevice
    import tkinter

    print(json.dumps({
        "status": "Resident synthesis ready; clipboard, buffered playback, and tray pending",
        "versions": {name: version(name) for name in (
            "piper-tts", "onnxruntime-gpu", "numpy", "sounddevice", "pystray", "pedalboard",
        )},
        "onnx_providers": onnxruntime.get_available_providers(),
        "tk_version": tkinter.TkVersion,
        "default_output": sounddevice.query_devices(kind="output")["name"],
        "cuda_dll_directory": str(args.cuda_dll_dir) if args.cuda_dll_dir else None,
        "note": "Provider availability alone does not prove GPU inference.",
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
