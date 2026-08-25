# Windows build and compatibility

WTiVo v1.0 is a **Windows x64 + NVIDIA CUDA** release.

## Supported/tested baseline

The benchmark machine used Windows 10 x64, Python 3.12, PyTorch 2.8.0+cu128, CUDA Toolkit 12.8, an NVIDIA RTX 5050 Laptop GPU, approximately 8 GB dedicated VRAM, and 16 CPU threads.

NVIDIA's CUDA 12.8 Windows guide lists Windows 10 22H2/Windows 11 and Visual Studio 2022/MSVC 193x as supported. WTiVo therefore deliberately selects Visual Studio 2022 and the known-good MSVC 14.38 (cl 19.38) toolset instead of blindly using a newer Visual Studio 2026 compiler.

## Prerequisites

`Setup-Windows.cmd` can install Python 3.12 and Visual Studio Build Tools with winget when available. The NVIDIA driver and CUDA Toolkit are intentionally treated as NVIDIA prerequisites; WTiVo does not bundle or silently replace them.

Install CUDA Toolkit 12.8 from NVIDIA if `nvcc --version` is unavailable.

## Per-machine CUDA build

The CUDA extension is not hard-coded to the author's RTX 5050. During setup, PyTorch reports the installed GPU compute capability and `scripts/build_gpupr.py` sets `TORCH_CUDA_ARCH_LIST` to that capability before compilation.

You may override this for advanced build/testing use:

```bat
set WTIVO_CUDA_ARCH=8.6
Setup-Windows.cmd
```

Do not override it unless you know why.

## Local dependency layout

WTiVo writes generated dependencies only below the checkout:

```text
.venv/                 Python environment
.deps/vcpkg/           pinned vcpkg checkout + installed ports
.build/native/         CMake/Ninja objects
.build/gpupr/          PyTorch extension build objects
build/                 final Python extension modules
.wtivo-env.cmd         machine-local CUDA/runtime paths
```

All of these are ignored by Git.

## Why source-first

A `.pyd` built on one machine is tied to Python ABI, compiler/runtime libraries, PyTorch/CUDA ABI, and potentially GPU architecture. GitHub releases therefore should contain source, not the author's prebuilt `.pyd` files, unless a future CI-produced binary matrix is added and tested.

## Memory guidance

The benchmark 1536/12M configuration is aimed at roughly 8 GB VRAM + 16 GB system RAM. Lower-memory systems should begin with 1024/1024. Higher final resolutions can create extremely large OpenVDB fields and final meshes even when the graph proxy remains fixed at 12M points.
