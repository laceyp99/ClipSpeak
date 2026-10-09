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


def controller_demo(voice_factory) -> dict:
    """Exercise real output plus one injected failure on fixed public text."""
    from .controller import QueueController, SubmissionStatus
    active = ("The active reading continues after a pause. Clearing the queue preserves "
              "this reading. Listen for this final sentence after playback resumes.")
    discarded = "A cleared or stopped pending item must never be heard."
    fresh = "This fresh reading follows Stop."
    retry = "The retained item plays again after Resume."
    follower = "The pending item follows the successful retry."
    labels = {active: "active", fresh: "fresh", retry: "retry", follower: "follower"}
    started = threading.Event()
    completions = []
    fail_once = True

    def playback(voice, text, **settings):
        nonlocal fail_once
        started.set()
        if text == retry and fail_once:
            fail_once = False
            raise RuntimeError("injected diagnostic failure")
        metrics = play_text(voice, text, **settings)
        completions.append({"sample": labels.get(text, "unexpected"),
                            "playback": metrics.to_dict()})
        return metrics

    controller = QueueController(voice_factory=voice_factory, playback=playback)
    def submit(text):
        result = controller.submit(text)
        if result.status is not SubmissionStatus.ACCEPTED:
            raise RuntimeError(result.reason)
    def drain():
        if not controller.wait_idle(30):
            raise RuntimeError(f"controller demo did not drain ({controller.snapshot().error_type})")
    def start_active():
        started.clear()
        submit(active)
        if not started.wait(10):
            raise RuntimeError("controller demo did not start")

    try:
        start_active()
        submit(discarded)
        time.sleep(2)
        controller.pause()
        submit(discarded)
        controller.clear_queue()
        paused = controller.snapshot()
        if not paused.has_active or paused.pending_items:
            raise RuntimeError("Clear Queue did not preserve only the active item")
        print("PAUSE AND CLEAR QUEUE", flush=True)
        time.sleep(2)
        controller.resume()
        drain()

        start_active()
        time.sleep(2)
        controller.pause()
        submit(discarded)
        before = time.perf_counter()
        controller.stop()
        stop_seconds = time.perf_counter() - before
        print("STOP WHILE PAUSED, THEN FRESH READING", flush=True)
        submit(fresh)
        drain()

        submit(retry)
        submit(follower)
        if controller.wait_idle(10):
            raise RuntimeError("diagnostic failure was not observed")
        error = controller.snapshot()
        print("INJECTED FAILURE, THEN RESUME", flush=True)
        controller.resume()
        drain()
        controller.pause()
        submit(discarded)
        controller.quit()
        if not controller.join(5):
            raise RuntimeError("controller demo did not quit")
        return {"completed": completions, "stop_call_seconds": stop_seconds,
                "retained_after_error": {"has_active": error.has_active,
                                         "pending_items": error.pending_items,
                                         "message": error.error_message},
                "quit_total_chars": controller.snapshot().total_chars}
    finally:
        controller.close()
        if not controller.join(5):
            raise RuntimeError("controller worker is still finishing inference")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cuda-dll-dir", type=Path,
        help="Existing directory containing CUDA 12 and cuDNN 9 DLLs",
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--tray", action="store_true", help="Run the Windows tray app with saved settings")
    mode.add_argument("--benchmark", action="store_true", help="Measure resident synthesis without playback")
    mode.add_argument("--controls-demo", action="store_true", help="Demonstrate pause, resume, stop and fresh playback")
    mode.add_argument("--clipboard-check", action="store_true", help="Snapshot clipboard and report status/count without displaying text")
    mode.add_argument("--queue-demo", action="store_true", help="Play three public samples with FIFO settings snapshots")
    mode.add_argument("--controller-demo", action="store_true", help="Demonstrate queue controls, cancellation, and recovery")
    parser.add_argument("--voice", type=Path, default=DEFAULT_VOICE, help="ONNX voice path for benchmark")
    parser.add_argument("--cpu", action="store_true", help="Use CPU for benchmark")
    mode.add_argument("--play-sample", choices=BENCHMARK_TEXTS, help="Play a fixed public sample")
    parser.add_argument("--speed", type=float, default=1.5, help="Pitch-preserving playback speed, 1 to 2")
    parser.add_argument("--volume", type=float, default=1.0, help="Playback volume, 0 to 1")
    args = parser.parse_args()
    if args.tray:
        from .app import run_app
        return run_app(lambda: ResidentVoice(args.voice, prefer_cuda=not args.cpu, cuda_dll_dir=args.cuda_dll_dir))
    if args.controller_demo:
        factory = lambda: ResidentVoice(args.voice, prefer_cuda=not args.cpu, cuda_dll_dir=args.cuda_dll_dir)
        print(json.dumps(controller_demo(factory), indent=2))
        return 0
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
        "status": "Runtime ready; use --tray to launch ClipSpeak",
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
