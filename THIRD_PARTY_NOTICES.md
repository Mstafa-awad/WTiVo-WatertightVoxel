# Third-party notices

WTiVo does not vendor complete copies of the libraries below. `Setup-Windows.cmd` downloads/builds dependencies from their upstream sources or official package channels. Their license files remain authoritative. vcpkg also installs per-port copyright/license information under `.deps/vcpkg/installed/x64-windows/share/<port>/copyright`.

| Component | WTiVo use | License / status | Upstream |
|---|---|---|---|
| CelloCut | tetra-cell-cut method, graph semantics, modified code lineage | Apache-2.0 | https://github.com/rangeryx-66/CelloCut |
| FaithC | contour/FCT/QEF concepts and modified contour lineage | Apache-2.0 | https://github.com/Luo-Yihao/FaithC |
| CGAL | 3D Delaunay tetrahedralization | **GPL v3+ for the open-source 3D Triangulations package**, commercial alternative available | https://www.cgal.org/license.html |
| OpenVDB | sparse unsigned/signed narrow-band grids, mesh-to-volume | Apache-2.0 for OpenVDB 12.x | https://github.com/AcademySoftwareFoundation/openvdb |
| oneTBB | CPU parallelism and CGAL parallel support | Apache-2.0 | https://github.com/uxlfoundation/oneTBB |
| Eigen | matrix types used by native topology code | primarily MPL-2.0; WTiVo defines `EIGEN_MPL2_ONLY` | https://gitlab.com/libeigen/eigen |
| pybind11 | Python/C++ bindings | BSD 3-Clause | https://github.com/pybind/pybind11 |
| PyTorch | CUDA tensor runtime and extension build | main project BSD 3-Clause; wheels contain their own third-party notices | https://github.com/pytorch/pytorch |
| NumPy | host arrays | main project BSD 3-Clause; wheels may contain additional notices | https://github.com/numpy/numpy |
| Trimesh | mesh I/O | MIT | https://github.com/mikedh/trimesh |
| vcpkg | source dependency manager | MIT; ports keep upstream licenses | https://github.com/microsoft/vcpkg |
| NVIDIA CUDA | compiler/runtime/driver | proprietary NVIDIA terms; external prerequisite, not bundled | https://developer.nvidia.com/cuda-toolkit |

## License copies in this repository

`LICENSES/` contains the full GPL/Apache/MPL and standard permissive license texts used by WTiVo's source/dependency map. For dependencies installed later (especially PyTorch and NumPy wheels), the license/notice files shipped by those exact installed packages are controlling and should be kept with any redistribution of those packages.

## Important CGAL consequence

The open-source WTiVo build uses CGAL 3D Triangulations. CGAL labels this package GPL and states that distributing software based on GPL CGAL data structures requires distributing the software's source under the GPL. This is the reason the combined WTiVo open-source distribution uses GPL-3.0-or-later instead of MIT-only.

Commercial use is still allowed under GPL. Closed-source redistribution is a different question and may require CGAL's commercial license plus a separate review of all other dependencies.

## Exact installed dependency notices

After a successful Windows build, WTiVo runs `scripts/collect_dependency_licenses.py`. It copies the actual vcpkg port copyright/license records and license/NOTICE files shipped by installed Python distributions into `build/installed-licenses/`.

If you redistribute a bundle containing those dependency binaries, preserve the corresponding generated notices as well as WTiVo's `LICENSE`, `NOTICE`, `LICENSING.md`, and `THIRD_PARTY_NOTICES.md`.
