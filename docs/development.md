# Development setup

ClipSpeak currently has development diagnostics and a resident-synthesis
benchmark. It does not yet read the clipboard, play speech, or create a tray icon.

Use the approved Python 3.12.11 x64 interpreter with uv:

```powershell
uv sync --locked
uv run --locked clipspeak
```

The project selects ONNX Runtime GPU 1.20.2 for CUDA 12.x and cuDNN 9 compatibility.
Its wheel also contains the CPU execution provider. Piper's CPU-only
`onnxruntime` dependency is excluded through uv to avoid installing two
distributions that overwrite the same Python package. Use uv for setup, rather
than installing this project with pip.

On Pat's current machine, compatible CUDA 12.6 and cuDNN 9.10 DLLs already exist
in the system Python 3.12 PyTorch installation. Supply that directory explicitly:

```powershell
uv run --locked clipspeak --cuda-dll-dir 'C:\Users\Patrick\AppData\Local\Programs\Python\Python312\Lib\site-packages\torch\lib'
```

This loads native DLLs and prepends their directory to PATH for this process only.
cuDNN needs the process PATH to locate its lazily loaded companion DLLs.
It does not import system PyTorch
into the project environment, copy its DLLs, change global PATH, or install a
driver/toolkit. This machine-specific directory is outside the uv lockfile and
must remain available. Another machine needs its own compatible runtime-library
directory. The synthesis layer tries CUDA first and uses CPU if its session
fails to load, silently chooses CPU, or fails before its first audio chunk.
An error after a delivered chunk propagates so playback cannot replay words.
The fallback reason names the error type only, avoiding copied text in output.

Place Ryan's `en_US-ryan-high.onnx` and `en_US-ryan-high.onnx.json` in
`%LOCALAPPDATA%\ClipSpeak\voices`, then measure resident synthesis:

```powershell
uv run --locked clipspeak --benchmark --cuda-dll-dir 'C:\Users\Patrick\AppData\Local\Programs\Python\Python312\Lib\site-packages\torch\lib'
uv run --locked clipspeak --benchmark --cpu
```

Use `--voice 'C:\path\to\voice.onnx'` to override the model path. The benchmark
loads one voice and uses it for two calls each on fixed short and long public
passages. `first_call` is the first call for that passage; only the short
passage's first call is the first synthesis request after load. It reports
model load time, first generated chunk latency, generated audio duration,
generation time, audio-seconds-per-generation-second, and Windows process
working set before load, after load, and after all calls. Working set is CPU
process memory, not GPU VRAM. It neither plays nor saves speech and does not
measure first audible sound, 1.5x playback, or long-running memory stability.

The default diagnostic reports package imports and default output selection,
not verified GPU inference or audio playback. Run focused synthesis tests with
`uv run --locked pytest -q`. Controller tests will follow when that behavior exists.

On 2026-10-07, the isolated environment passed a tiny ONNX Add inference with
CPU fallback disabled. Profiling confirmed the node ran on CUDAExecutionProvider
and returned the expected result. A cuDNN handle creation check also passed after
the process PATH adjustment. This establishes compatibility for a basic GPU
operation, not yet Ryan synthesis or sustained speech performance. Driver 560.94
and an RTX 4050 Laptop GPU were used; no driver update was required.

## Resident Ryan validation

The next synthesis unit downloaded Ryan to the per-user voices directory.
The model (120,786,792 bytes) and config (4,166 bytes) matched the MD5 digests
published in Piper's voice index. A separate model profile confirmed actual
CUDA node execution, alongside CPU-assigned operations. Generated audio was
finite, non-silent, mono, and 22,050 Hz.

Separate CUDA and CPU benchmark runs used the same resident model for four
requests. CUDA model load took about 1.43 seconds; its first short request took
about 601 ms to generate the first chunk, and the repeat took about 165 ms.
The longer passage generated roughly 58 seconds of normal-speed speech in
4.53 to 5.19 seconds. Process working set increased from about 46 MiB before
load to 407 MiB after load and 916 MiB after synthesis.

CPU model load took about 1.03 seconds. Its repeated short request took about
657 ms to generate the first chunk, and the longer passage took 10.86 to
11.26 seconds to generate roughly 58 seconds of speech. Process working set
was about 460 MiB after synthesis. These are individual local measurements,
not latency guarantees, first-sound timings, or evidence of long-running
memory stability. Earlier concurrent runs were excluded from these figures.

Five fake-backed tests passed for reuse, effective-provider fallback, failure
before the first chunk, avoiding replay after partial delivery, and a missing
model. A real missing-DLL-directory check also synthesized on CPU. Two public
normal-speed samples completed playback through the default Shokz headphones
using a one-off in-memory check; Pat confirmed both sounded clear without
clipping or distortion. Buffered playback, pitch-preserving acceleration, and controls are
not yet implemented or verified.
