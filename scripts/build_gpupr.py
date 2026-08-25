# SPDX-License-Identifier: GPL-3.0-or-later
"""Build WTiVo's CUDA graph solver for the GPU installed in this PC."""
from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

if os.name != "nt":
    raise SystemExit("WTiVo GPUPr build currently supports Windows only.")

ROOT = Path(__file__).resolve().parents[1]
BUILD_DIR = ROOT / ".build" / "gpupr"
OUT_DIR = ROOT / "build"
BUILD_DIR.mkdir(parents=True, exist_ok=True)
OUT_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MAX_JOBS", str(min(6, os.cpu_count() or 1)))

import torch
from torch.utils.cpp_extension import load

forced_arch = os.environ.get("WTIVO_CUDA_ARCH", "").strip()
if torch.cuda.is_available():
    cc = torch.cuda.get_device_capability(0)
    detected_arch = f"{cc[0]}.{cc[1]}"
    gpu_name = torch.cuda.get_device_name(0)
else:
    cc = None
    detected_arch = ""
    gpu_name = "<no runtime GPU visible>"

if not forced_arch and not detected_arch:
    raise RuntimeError("CUDA GPU is unavailable in PyTorch and WTIVO_CUDA_ARCH was not supplied.")
arch = forced_arch or detected_arch
os.environ["TORCH_CUDA_ARCH_LIST"] = arch

cuda_home = os.environ.get("WTIVO_CUDA_HOME") or os.environ.get("CUDA_HOME") or os.environ.get("CUDA_PATH")
if not cuda_home:
    default = Path(r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8")
    if default.exists():
        cuda_home = str(default)
if not cuda_home or not (Path(cuda_home) / "bin" / "nvcc.exe").exists():
    raise RuntimeError("CUDA Toolkit with nvcc was not found. Install CUDA Toolkit 12.8 or set WTIVO_CUDA_HOME.")
os.environ["CUDA_HOME"] = cuda_home
os.environ["CUDA_PATH"] = cuda_home

extra_cuda = ["-O3", "--std=c++17"]
if os.environ.get("WTIVO_ALLOW_UNSUPPORTED_MSVC", "0") == "1":
    extra_cuda.append("-allow-unsupported-compiler")

print("[WTiVo GPU build]")
print(" Python :", sys.executable)
print(" Torch  :", torch.__version__)
print(" CUDA   :", torch.version.cuda)
print(" Toolkit:", cuda_home)
print(" GPU    :", gpu_name)
print(" CC     :", cc)
print(" Arch   :", arch)

mod = load(
    name="wtivo_gpupr",
    sources=[
        str(ROOT / "native" / "gpupr" / "wtivo_gpupr_bindings.cpp"),
        str(ROOT / "native" / "gpupr" / "gpu_push_relabel_fast.cu"),
    ],
    build_directory=str(BUILD_DIR),
    extra_cflags=["/O2", "/std:c++17", "/fp:precise"],
    extra_cuda_cflags=extra_cuda,
    with_cuda=True,
    verbose=True,
)

built = Path(mod.__file__).resolve()
dst = OUT_DIR / "wtivo_gpupr.pyd"
shutil.copy2(built, dst)
print("[SUCCESS] built:", dst)

sys.path.insert(0, str(OUT_DIR))
import wtivo_gpupr  # noqa: E402
for symbol in ("graph_cut_fast", "surface_extraction_topology_cuda"):
    if not hasattr(wtivo_gpupr, symbol):
        raise RuntimeError(f"Missing export after build: {symbol}")
print("[SUCCESS] GPUPr exports verified")
