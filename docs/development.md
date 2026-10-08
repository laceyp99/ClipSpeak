# Development setup

ClipSpeak currently has development diagnostics, a resident-synthesis benchmark,
and buffered playback of fixed public samples. It does not yet read the
clipboard or create a tray icon.

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
not verified GPU inference or audio playback. Run focused tests with
`uv run --locked pytest -q`. Controller tests will follow when that behavior exists.

## Buffered playback diagnostic

```powershell
uv run --locked clipspeak --play-sample short --speed 1.5 --volume 1 --cuda-dll-dir 'C:\Users\Patrick\AppData\Local\Programs\Python\Python312\Lib\site-packages\torch\lib'
uv run --locked clipspeak --play-sample long --speed 2 --cpu
```

The diagnostic uses the default output and only fixed public passages. Speed is
1 to 2, defaults to 1.5, and uses Pedalboard pitch-preserving time stretch after
normal-speed synthesis. Volume is 0 to 1 and defaults to 1. Text is split into
segments of at most 300 characters with no characters discarded. One text
segment and its Piper chunk are processed at a time. The audio callback reads a
preallocated two-second mono float32 ring. Backpressure stops the producer when
the ring is full; a processed chunk can temporarily exist outside the ring, so
the ring capacity is not a total process-memory bound. Audio processing runs on
the worker thread, never in the output callback.

The JSON reports source and stretched durations, actual audio frames submitted
to the output stream, stretch CPU time, ring peak, software starvation counts,
PortAudio output-underflow flags, and first-submission time. The estimated first
DAC time is based on PortAudio's callback clock and is not a microphone measurement
of audible sound. Natural completion drains the ring. An unexpected output stop
raises an error. The controls diagnostic below exercises Pause, Resume, and Stop.

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
clipping or distortion. Playback controls remain pending.

## Buffered Ryan validation

On 2026-10-08, the Windows default output was Headphones (OpenRun by Shokz).
It accepted mono float32 at Ryan's 22,050 Hz sample rate. The long passage
produced about 58.5 seconds of normal speech and played in 39.0 seconds at
1.5x, on both CUDA and CPU. Both runs recorded zero software starvations and
zero PortAudio output underflows; all processed frames reached the callback.
The ring peaked at its 44,100-frame capacity (two seconds, 172 KiB).
Pedalboard processing took about 0.87 seconds on the CUDA run and 0.93 seconds
on the CPU run across the passage. Pat confirmed the 1.5x sample sounded clear.

Short CPU samples at 1x with half volume and 2x with full volume also drained
without measured gaps or underflows. The long CPU run's first callback submission
was about 0.86 seconds after requesting playback, with estimated first DAC output
at 1.04 seconds. This excludes model loading and estimates device timing rather
than measuring audible sound. Single runs do not establish sustained memory
stability or device switching behavior.

Seventeen focused tests passed, including pitch/duration at 1x, 1.5x and 2x,
word preservation, ring order/backpressure, complete draining, volume scaling,
failed output cleanup, and safe processing errors. The fast-producer test exposed
a lost startup notification when a chunk exceeded the ring capacity. Startup now
checks buffer state directly, and that test passes. Pause, Resume, Stop, queue
snapshots, and longer stability/device-switch checks are still pending.

## Playback controls diagnostic

```powershell
uv run --locked clipspeak --controls-demo --cuda-dll-dir 'C:\Users\Patrick\AppData\Local\Programs\Python\Python312\Lib\site-packages\torch\lib'
uv run --locked clipspeak --controls-demo --cpu
```

This uses fixed public text. It plays at 1.5x by default, pauses after three
seconds, resumes two seconds later, then stops after another three seconds.
It also cancels a short request before its first audio and finally plays a fresh
short sample with the same resident voice. Console action markers precede JSON
metrics, including whether each reading was stopped. No clipboard text is read
or saved.

The `played_frames` and `played_duration_seconds` fields count frames delivered
to the output callback. On Stop, some of those frames may still be in device
buffers and are discarded, so they do not measure how much speech was heard.

Each `play_text` call takes a separate, one-use `PlaybackControl` handle. Pause
preserves the ring cursor and leaves the output stream sending silence; audio
already submitted to the device may finish before the pause becomes audible.
Resume reads from that retained cursor. Stop clears the ring, discards late
results, and aborts pending device output. It does not kill an inference thread.
The control call does not wait for inference, but the synchronous playback call
waits for its worker to finish before returning. The caller must finish that
call before reusing the resident voice for another reading. An unusually slow
inference can therefore delay the next reading even though stopped audio is
silent. Processing remains bounded by 300-character segments and the two-second
ring, with one processed chunk outside the ring.

This unit supplies playback controls only. FIFO submission, clearing pending
items, error recovery, tray wiring, and shutdown behavior belong to later units.
Stop uses [sounddevice abort](https://python-sounddevice.readthedocs.io/en/latest/api/streams.html#sounddevice.Stream.abort)
to discard pending output; natural completion uses buffer-draining completion.

### Control validation, 2026-10-08

CUDA and CPU demos passed on the default Shokz headphones at 1.5x. Pat confirmed
Pause, Resume at the same place, clean Stop, and fresh playback sounded correct.
Stop calls returned in about 2.3 ms (CUDA) and 2.9 ms (CPU). These measure the
control method and device-abort call, not microphone-measured silence latency.
Both pre-audio cancellations submitted zero frames. Fresh playback afterward
completed, with estimated first DAC output about 0.64 seconds on CUDA and
0.76 seconds on CPU, excluding model load. Both demos recorded zero software
starvations and device underflows, and the ring stayed within two seconds.

Twenty-three tests passed across synthesis and playback. Control tests establish
cursor/order preservation and bounded buffering during pause, silence before
Resume, Stop both active and paused, harmless repeated Stop/Resume after Stop,
late inference rejection, sequential resident-voice reuse, and rejection of a
reused control handle. Required checks for this Phase 2 unit passed. Queue/tray
integration, default-device changes, and sustained memory checks are still pending.
