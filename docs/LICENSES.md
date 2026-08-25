# License map

See the root `LICENSE` for GPL-3.0-or-later and `LICENSING.md` for the commercial-use explanation.

The key point is simple: **WTiVo is free for commercial use, but the open-source CGAL build is copyleft.** Commercial use and proprietary redistribution are not the same thing.

| Source/component | License treatment |
|---|---|
| WTiVo combined open-source distribution | GPL-3.0-or-later |
| CelloCut-derived portions | Apache-2.0 notices retained |
| FaithC-derived/inspired contour portions | Apache-2.0 notices retained |
| CGAL 3D Triangulations | GPL v3+ open-source option; commercial alternative |
| OpenVDB 12.x | Apache-2.0 |
| oneTBB | Apache-2.0 |
| Eigen | MPL-2.0 path enforced with `EIGEN_MPL2_ONLY` |
| pybind11 | BSD 3-Clause |
| PyTorch | main project BSD 3-Clause + wheel third-party notices |
| NumPy | main project BSD 3-Clause + wheel third-party notices |
| Trimesh | MIT |
| vcpkg | MIT; installed ports keep their own licenses |
| CUDA Toolkit / NVIDIA driver | external NVIDIA proprietary terms; not redistributed here |

For redistributed dependency binaries, preserve the exact license/copyright files that accompany the exact binary versions you distribute. WTiVo's source installer avoids bundling those third-party binaries in the Git repository.
