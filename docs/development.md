# Development setup

ClipSpeak currently has development diagnostics, clipboard snapshots, a
resident-synthesis benchmark, and buffered playback of fixed public samples.
FIFO submission is available through the controller. Hotkeys and the tray are
not implemented yet.

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

## Clipboard snapshot diagnostic

```powershell
uv run --locked clipspeak --clipboard-check
```

This reads Windows `CF_UNICODETEXT` once and copies it into an app-owned Python
string before releasing the clipboard. It prints status and character count,
without displaying, saving, or speaking the text. It does not change focus or
modify the clipboard. A later clipboard change cannot change the returned
string. The diagnostic does not enqueue work; queue submission is the next unit.

Opening a busy clipboard is retried five times, 50 ms apart, for a maximum
200 ms of retry waits. Windows delayed rendering and native API calls can add
time beyond those waits. Empty/whitespace-only content, non-text content, native
read failures, and submissions above 100,000 characters have separate results.
Oversized text is rejected without truncation. A bounded UTF-16 read preserves
Unicode, including supplementary characters, without copying an arbitrary-sized
clipboard allocation. Pending-item and total-queue limits will be implemented
with the queue controller.

The reader follows [Microsoft's clipboard ownership rules](https://learn.microsoft.com/en-us/windows/win32/api/winuser/nf-winuser-getclipboarddata),
copying while the clipboard is open and releasing the lock before closing.

### Clipboard validation, 2026-10-08

Thirty tests passed across the project, including seven clipboard tests for
Unicode and immutable snapshots, transient/permanent contention, empty versus
non-text content, failed reads and cleanup, malformed data, allocation padding,
and the 100,000-character boundary (including supplementary Unicode characters).
The CLI's read-only diagnostic returned status/count without displaying text.

A live 22-character Unicode sample containing accents, Japanese, punctuation,
and an emoji matched exactly. Clipboard sequence number and contents remained
unchanged. A separate process holding the clipboard through a hidden window
caused the reader to return `unavailable` after about 202 ms, then read normally
after release. The initial windowless test holder did not block the reader;
using a window-associated holder verified the real contention path. The helper
window and process were closed, with no clipboard writes performed.

Representative browser, VS Code, and Obsidian copying, actual hotkey submission,
and queue snapshots remain part of later integration verification. Native delayed
rendering can exceed the retry wait duration; it has not been measured here.

## FIFO queue diagnostic

```powershell
uv run --locked clipspeak --queue-demo --cuda-dll-dir 'C:\Users\Patrick\AppData\Local\Programs\Python\Python312\Lib\site-packages\torch\lib'
uv run --locked clipspeak --queue-demo --cpu
```

The demo submits three fixed public samples together: item one at 1x and full
volume, item two at 1.5x and half volume, and item three at 2x and full volume.
These fixed settings are part of the demo rather than the `--speed`/`--volume`
arguments. It reports completed samples and playback metrics, then closes and
joins the queue worker. No copied text is displayed or persisted.

`QueueController` owns one worker, a resident voice, active item, pending deque,
and a one-use playback control for each reading. `submit` snapshots immutable
text, speed and volume. `submit_clipboard` reads and enqueues while holding the
submission lock, preserving read order across concurrent calls. Later clipboard
changes do not change queued text; identical text remains separate submissions.
The next item starts only after `play_text` returns following output drain and
synthesis-worker completion. The voice is loaded on the worker at startup and
reused. Submissions can wait in memory while that load runs; a load failure blocks
reading and preserves pending items.

Limits are 100,000 characters per item, 20 pending items, and 500,000 characters
including the active item. Rejected items leave the queue unchanged. Status
snapshots report counts/state and exception type, without copied text or raw
error messages. A playback/model error preserves work and halts advancement.
Public controls and recovery are described below. Close/join cleanup lets tests
and development harnesses release their workers safely.

Clipboard reads hold a separate submission lock, preserving read/enqueue order
without blocking playback controls. Native delayed rendering can still delay
another submission. Stop and Quit invalidate submissions already reading or
waiting for that lock. Busy-open retries are bounded as documented above;
delayed rendering duration remains an unmeasured integration limitation.

### Queue validation, 2026-10-08

Thirty-eight project tests passed. Queue tests cover clipboard text and speed/
volume snapshots, duplicate items, FIFO order across concurrent submissions,
waiting for playback completion, resident voice reuse, invalid inputs,
independent pending-count and total-character limits, preservation after errors,
model loading at startup, and cleanup during a delayed model load. Closed workers
release active/pending text; error snapshots retain only the exception type.

CUDA and CPU demos completed items one, two, and three in order at the specified
speeds and volumes, with zero software starvations and output underflows. Pending
item count and total characters returned to zero, and each worker joined on exit.
Pat confirmed correct order, a quieter second item, and no overlapping speech.
Controller control transitions and actionable error recovery remain pending, as
do tray/hotkey integration, device switching, and sustained memory verification.

## Queue controls and recovery diagnostic

```powershell
uv run --locked clipspeak --controller-demo --cuda-dll-dir 'C:\Users\Patrick\AppData\Local\Programs\Python\Python312\Lib\site-packages\torch\lib'
uv run --locked clipspeak --controller-demo --cpu
```

The fixed-public-text demo pauses an active reading, submits more while paused,
clears pending work, and resumes the retained playback cursor. It then stops a
second reading while paused and submits fresh work. One injected playback error
before audio tests retention and Resume retry, followed by the preserved pending
item. Finally it quits while paused with pending work and joins the worker.
Console action markers precede JSON metrics. The injected error is a deliberate
diagnostic fault, not evidence of a physical device failure.

Pause freezes active playback and queue advancement. Resume continues that cursor
when paused normally. After a synthesis/output error, it retries the active item
from the beginning with its original text, speed, and volume; the safe error
message explains this. After model-load failure, it retries model loading. Check
model files/runtime or output selection as the error message directs. Raw exception
messages and copied text are omitted. CUDA preference/fallback remains in the
resident synthesis layer.

Stop invalidates active and pending work and clears logical queue counts
immediately, including clipboard submissions still being read or waiting to be
read. Canceled inference may finish internally, but its results and errors cannot
revive the queue. Fresh work waits for the single worker to finish that inference.
Clear Queue removes pending items and preserves active playback or its failed
item awaiting recovery. Quit is terminal: clear, silence, reject new submissions,
and join outside UI handlers. Repeated commands are harmless. Default-device
refresh and real device-switch recovery still require later Windows verification.

### Queue control validation, 2026-10-08

Fifty-two project tests passed. Controllable fake playback/reads establish Pause
before and during reading, submissions while paused, preserved cursor/queue order,
Clear Queue preserving active work, Stop rejecting delayed successes and errors,
fresh work waiting for canceled inference, terminal Quit active or paused,
retry from the beginning with original settings, model-load retry, safe error
messages, and cancellation of clipboard reads or submissions waiting behind them.

Real CUDA and CPU controller demos completed the active reading after Clear Queue,
stopped a second active reading while paused, played fresh work, then recovered
from the injected fault and played the retained item before its pending follower.
Neither run spoke discarded pending items. Both recorded zero software starvations
and output underflows. Stop calls took about 10.4 ms on CUDA and 1.5 ms on CPU;
these measure the control/abort calls, not microphone-measured silence. Quit
returned logical queued characters to zero and each worker joined. After a replay,
Pat confirmed clean Pause/Resume, active completion after Clear Queue, no old
speech after Stop, and fresh/retry/follower playback in the expected order.

The Phase 3 fake-backed acceptance checks pass. Tray, hotkey, Settings, single
instance, live output-device switching, real device-failure recovery, and sustained
memory verification remain later work. A slow native clipboard renderer may still
delay another submission, but the controller controls no longer wait for that read.
