# Source provenance and attribution

WTiVo is an independent project maintained at `Mstafa-awad/WTiVo-WatertightVoxel-Optimizer`. It is not an official CelloCut, FaithC, CGAL, OpenVDB, or NVIDIA project and is not endorsed by those projects.

This document records the lineage of the production v1.0 source so downstream users can understand what is original WTiVo work, what is modified upstream-lineage work, and why licenses are retained.

## CelloCut lineage

Upstream: **CelloCut: Constructive Watertight Remeshing via Tetrahedral Cell Cuts**, Xuan Yang, Yuhang Zeng, Dinglong Fang, Guochuan Tang, Jiaju Jiang, Ben Li, Wei Zhou, Xiao-Xiao Long, and Cheng Lin.

Repository: <https://github.com/rangeryx-66/CelloCut>  
License: Apache-2.0.

WTiVo retains the tetrahedral cell-cut formulation, fill-aware graph objective semantics, topology/face ordering conventions, and modified execution code derived from the Apache-2.0 CelloCut implementation. Modified files carry an Apache-2.0 SPDX header and a prominent `Modified for WTiVo in 2026` notice.

## FaithC lineage

Upstream: **Faithful Contouring: Near-Lossless 3D Voxel Representation Free from Iso-surface**, Yihao Luo et al., CVPR 2026.

Repository: <https://github.com/Luo-Yihao/FaithC>  
License: Apache-2.0.

WTiVo uses FaithC/FCT/QEF contouring concepts in its sparse-field contour bridge and manifold contouring work. The WTiVo implementation is integrated with the OpenVDB + tetra-cell-cut pipeline rather than distributed as an unmodified FaithC checkout. FaithC attribution is retained in `NOTICE` and `THIRD_PARTY_NOTICES.md`.

## WTiVo production modifications

The v1.0 production line adds or consolidates the following work developed during WTiVo's optimization process:

- sparse global OpenVDB unsigned/signed narrow-band fields;
- direct FaithC/QEF candidate extraction into a fixed 12,000,000-point proxy;
- removal of the production thick-proxy triangle mesh and QEM proxy decimator;
- parallel CGAL Delaunay tetrahedralization with direct four-neighbor export;
- Windows memory-lifecycle changes that avoid forced working-set eviction in the hot path;
- a CUDA reduced-graph multi-discharge Push-Relabel schedule while preserving the graph capacity objective;
- final exact residual reachability classification;
- component-aware manifold contour splitting and highly local residual edge-fan/tiny-loop finalization;
- mandatory exact undirected edge-degree watertight validation.

## File map

| File | Lineage / license |
|---|---|
| `wtivo.py` | modified Apache-2.0 CelloCut-lineage production orchestration |
| `native/core/wtivo_core.cpp` | modified Apache-2.0 topology code; uses GPL CGAL 3D Triangulations at build/runtime |
| `native/vdb/wtivo_vdb.cpp` | Apache-2.0 WTiVo sparse OpenVDB/FaithC-compatible bridge and finalizer |
| `native/gpupr/*` | Apache-2.0 WTiVo CUDA graph implementation derived from the CelloCut graph objective |
| setup, CI, packaging, documentation | WTiVo-authored GPL-3.0-or-later files unless marked otherwise |

## Combined-license consequence

The public build links/uses CGAL's GPL-licensed 3D Triangulations package. Therefore the combined open-source WTiVo program is distributed under **GPL-3.0-or-later**, while Apache-2.0-derived files retain their file-level Apache terms and notices. See `LICENSING.md` for the practical distribution consequences.

No trademark or endorsement rights are claimed.
