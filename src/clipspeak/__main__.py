"""Development entry point for checking the selected runtime stack."""

import argparse
import ctypes
import json
import os
from importlib.metadata import version
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cuda-dll-dir", type=Path,
        help="Existing directory containing CUDA 12 and cuDNN 9 DLLs",
    )
    args = parser.parse_args()
    # Retain handles until the process exits. No global PATH modification or
    # import of another Python environment's packages is needed.
    dll_handles = []
    if args.cuda_dll_dir:
        directory = args.cuda_dll_dir.resolve(strict=True)
        # cuDNN loads companion libraries lazily using the process PATH.
        os.environ["PATH"] = str(directory) + os.pathsep + os.environ.get("PATH", "")
        dll_handles.append(os.add_dll_directory(str(directory)))
        for name in ("cudart64_12.dll", "cublasLt64_12.dll", "cublas64_12.dll", "cudnn64_9.dll"):
            dll_handles.append(ctypes.WinDLL(str(directory / name)))

    import onnxruntime
    import pedalboard
    import piper
    import pystray
    import sounddevice
    import tkinter

    print(json.dumps({
        "status": "Development tooling ready; speech/tray implementation pending",
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
