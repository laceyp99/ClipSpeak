# Development setup

ClipSpeak currently has a development diagnostic entry point. It does not yet
read the clipboard, synthesize speech, or create a tray icon.

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
directory. Actual automatic CPU fallback belongs to the next synthesis unit.

The diagnostic reports package imports and default output selection, not verified
GPU inference or audio playback. Focused controller tests will be added when
controller behavior exists; pytest is available in the development environment.

On 2026-10-07, the isolated environment passed a tiny ONNX Add inference with
CPU fallback disabled. Profiling confirmed the node ran on CUDAExecutionProvider
and returned the expected result. A cuDNN handle creation check also passed after
the process PATH adjustment. This establishes compatibility for a basic GPU
operation, not yet Ryan synthesis or sustained speech performance. Driver 560.94
and an RTX 4050 Laptop GPU were used; no driver update was required.
