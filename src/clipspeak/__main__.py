"""Development diagnostics and resident synthesis benchmark."""

import argparse
import json
import ctypes
import os
import time
import threading
from importlib.metadata import version
from pathlib import Path

from .synthesis import DEFAULT_VOICE, ResidentVoice, add_cuda_dll_directory
from .playback import PlaybackControl, play_text


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


def controls_demo(voice, *, speed: float, volume: float) -> dict:
    """Run timed controls on fixed public samples without clipboard access."""
    control = PlaybackControl()
    done = threading.Event()
    actions = []
    started = time.perf_counter()

    def commands():
        for delay, name in ((3, "pause"), (2, "resume"), (3, "stop")):
            if done.wait(delay):
                return
            before = time.perf_counter()
            getattr(control, name)()
            actions.append({"action": name, "at_seconds": before - started,
                            "call_seconds": time.perf_counter() - before})
            print(name.upper(), flush=True)

    thread = threading.Thread(target=commands, name="clipspeak-demo-controls")
    thread.start()
    try:
        interrupted = play_text(voice, BENCHMARK_TEXTS["long"], speed=speed,
                                volume=volume, control=control)
    finally:
        done.set()
        control.stop()
        thread.join()

    # Cancel a separate request before its first audio, then reuse the resident voice.
    print("STOP BEFORE FIRST AUDIO", flush=True)
    early_control = PlaybackControl()
    timer = threading.Timer(0.03, early_control.stop)
    timer.start()
    try:
        early = play_text(voice, BENCHMARK_TEXTS["short"], speed=speed,
                         volume=volume, control=early_control)
    finally:
        timer.cancel()
        timer.join()
        early_control.stop()
    time.sleep(1)
    print("FRESH SAMPLE", flush=True)
    fresh = play_text(voice, BENCHMARK_TEXTS["short"], speed=speed, volume=volume)
    return {"provider": voice.provider, "actions": actions,
            "pause_resume_stop": interrupted.to_dict(),
            "stop_before_audio": early.to_dict(), "fresh_after_stop": fresh.to_dict()}


def queue_demo(voice_factory) -> dict:
    """Play fixed public submissions with different immutable settings."""
    from .controller import QueueController, SubmissionStatus
    samples = [
        ("one", "This is queue item one, played at normal speed.", 1.0, 1.0),
        ("two", "This is queue item two, played faster and at half volume.", 1.5, 0.5),
        ("three", "This is queue item three, played at double speed.", 2.0, 1.0),
    ]
    labels = {text: name for name, text, _, _ in samples}
    completions = []

    def playback(voice, text, **settings):
        metrics = play_text(voice, text, **settings)
        completions.append({"sample": labels.get(text, "unexpected"),
                            "provider": voice.provider, "playback": metrics.to_dict()})
        return metrics

    controller = QueueController(voice_factory=voice_factory, playback=playback)
    try:
        for _, text, speed, volume in samples:
            result = controller.submit(text, speed=speed, volume=volume)
            if result.status is not SubmissionStatus.ACCEPTED:
                raise RuntimeError(result.reason)
        if not controller.wait_idle(timeout=30):
            state = controller.snapshot()
            raise RuntimeError(f"queue demo did not finish ({state.state}, {state.error_type})")
        state = controller.snapshot()
        return {"completed": completions, "pending_items": state.pending_items,
                "total_chars": state.total_chars}
    finally:
        controller.close()
        if not controller.join(timeout=5):
            raise RuntimeError("queue worker is still finishing synthesis")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cuda-dll-dir", type=Path,
        help="Existing directory containing CUDA 12 and cuDNN 9 DLLs",
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--benchmark", action="store_true", help="Measure resident synthesis without playback")
    mode.add_argument("--controls-demo", action="store_true", help="Demonstrate pause, resume, stop and fresh playback")
    mode.add_argument("--clipboard-check", action="store_true", help="Snapshot clipboard and report status/count without displaying text")
    mode.add_argument("--queue-demo", action="store_true", help="Play three public samples with FIFO settings snapshots")
    parser.add_argument("--voice", type=Path, default=DEFAULT_VOICE, help="ONNX voice path for benchmark")
    parser.add_argument("--cpu", action="store_true", help="Use CPU for benchmark")
    mode.add_argument("--play-sample", choices=BENCHMARK_TEXTS, help="Play a fixed public sample")
    parser.add_argument("--speed", type=float, default=1.5, help="Pitch-preserving playback speed, 1 to 2")
    parser.add_argument("--volume", type=float, default=1.0, help="Playback volume, 0 to 1")
    args = parser.parse_args()
    if args.queue_demo:
        factory = lambda: ResidentVoice(args.voice, prefer_cuda=not args.cpu, cuda_dll_dir=args.cuda_dll_dir)
        print(json.dumps(queue_demo(factory), indent=2))
        return 0
    if args.clipboard_check:
        from .clipboard import read_clipboard_text
        result = read_clipboard_text()
        print(json.dumps({"status": result.status,
                          "characters": len(result.text) if result.text is not None else 0}, indent=2))
        return 0
    if args.controls_demo:
        voice = ResidentVoice(args.voice, prefer_cuda=not args.cpu, cuda_dll_dir=args.cuda_dll_dir)
        print(json.dumps(controls_demo(voice, speed=args.speed, volume=args.volume), indent=2))
        return 0
    if args.benchmark:
        print(json.dumps(benchmark(args.voice, prefer_cuda=not args.cpu, cuda_dll_dir=args.cuda_dll_dir), indent=2))
        return 0
    if args.play_sample:
        voice = ResidentVoice(args.voice, prefer_cuda=not args.cpu, cuda_dll_dir=args.cuda_dll_dir)
        metrics = play_text(voice, BENCHMARK_TEXTS[args.play_sample], speed=args.speed, volume=args.volume)
        print(json.dumps({"provider": voice.provider, "fallback_reason": voice.fallback_reason,
                          "playback": metrics.to_dict()}, indent=2))
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
        "status": "Synthesis, playback, clipboard snapshots, and FIFO queue ready; tray and hotkey pending",
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
