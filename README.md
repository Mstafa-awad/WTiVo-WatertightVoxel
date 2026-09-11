# WTiVo: WatertightVoxel Optimizer

**WTiVo** is a Windows/NVIDIA watertight remeshing pipeline for turning defective triangle meshes into dense, closed, manifold meshes using a sparse voxel field, tetrahedral cell cut, CUDA graph optimization, and a manifold contour finalizer.

Repository target: `Mstafa-awad/WTiVo-WatertightVoxel-Optimizer`

> **Release status:** v1.0.0 source release. The production algorithm was benchmarked on one Windows RTX 5050 Laptop system. The repository is source-first: each Windows PC builds its own native extensions instead of using the author's machine-specific `.pyd` files.

## What WTiVo changes

WTiVo grew from experiments around the Apache-2.0 CelloCut cell-cut formulation and Apache-2.0 FaithC contouring work. It is **not** presented as a replacement authorship claim over those projects; attribution and file-level notices are kept in `NOTICE`, `LICENSING.md`, and `THIRD_PARTY_NOTICES.md`.

The production path is:

```text
input triangle mesh
        |
        v
sparse OpenVDB unsigned distance field
        |
        v
FaithC/QEF candidate anchors
        |
        v
fixed 12,000,000 point proxy
(no proxy triangles, no QEM decimator)
        |
        v
CGAL parallel 3D Delaunay tetrahedralization
+ direct tetra-neighbor export
        |
        v
CelloCut-compatible inside/outside labels + lambda_fill capacities
        |
        v
WTiVo CUDA reduced graph
multi-discharge Push-Relabel
        |
        v
cut surface
        |
        v
signed sparse OpenVDB field
        |
        v
manifold FaithC-style contour
+ local edge-manifold finalizer
        |
        v
exact edge-degree watertight validation
        |
        v
GLB / OBJ / supported Trimesh output
```

The major production changes are:

1. **No full thick proxy mesh and no QEM proxy decimation.** The sparse thick field goes directly to FaithC/QEF anchors and an exact fixed 12M point budget.
2. **Sparse OpenVDB instead of a giant dense UDF allocation.** Only the narrow-band field is stored.
3. **Direct tetra neighbors.** CGAL's Delaunay cells export the four CelloCut-compatible neighbor slots directly, avoiding a separate huge face-adjacency rebuild.
4. **Exact graph objective, faster GPU schedule.** WTiVo keeps the cell-cut capacity formula/objective but the CUDA solver performs several legal push/relabel operations per active vertex before requeueing. A final residual BFS still determines the exact source/sink partition.
5. **Memory lifecycle tuned for Windows.** The thick sparse field is freed immediately after label sampling; large topology arrays are uploaded in chunks; Windows working-set eviction is not forced during the hot path.
6. **Manifold final contour.** Ambiguous dual-contour cells are split by local surface component, followed by a very local edge-fan/tiny-loop finalizer. No global smoothing is added.
7. **Final validation is mandatory.** WTiVo reports `watertight=True/False` and the exact number of non-degree-2 undirected edge groups.

## Tested benchmark

Reference production run:

| Item | Test value |
|---|---|
| OS | Windows 10 x64, build 26200.9168 |
| GPU | NVIDIA GeForce RTX 5050 Laptop GPU, ~8 GB dedicated VRAM |
| PyTorch | 2.8.0+cu128 |
| CUDA Toolkit/runtime | 12.8 |
| CPU threads used | 16 |
| System RAM | ~16 GB |
| `--input-res` | 1536 |
| `--final-res` | 1536 |
| proxy budget | fixed 12,000,000 points |
| `--lambda_fill` | 10 |
| graph solve | ~8.76 s |
| total runtime | ~1.49 min |
| final mesh | 12,404,563 vertices / 24,809,250 faces |
| final validation | `watertight=True`, `bad_edge_groups=0` |

This is a benchmark, **not a promise that every asset or GPU will take 1.49 minutes**. The production run did use several GB of VRAM; it is incorrect to describe VRAM use as zero. The important result is that the tested 1536/12M path completed on an ~8 GB GPU without a VRAM OOM.

## Supported platform for v1.0

- Windows 10 22H2 or Windows 11 x64
- NVIDIA CUDA-capable GPU
- NVIDIA driver compatible with PyTorch CUDA 12.8
- CUDA Toolkit 12.8 with `nvcc`
- Python 3.12 x64
- Visual Studio 2022 Build Tools with **MSVC v14.38 / cl 19.38** (the known-good CUDA-compatible toolset)
- 16 GB system RAM recommended for the 1536/12M benchmark configuration
- ~8 GB VRAM recommended for the benchmark configuration

RTX 20/30/40/50-family GPUs are intended targets, but **the only benchmarked machine so far is the RTX 5050 Laptop system above**. AMD-only, Intel-only, CPU-only, Linux, and macOS systems are not supported by the v1.0 fast CUDA path.

## Quick start

Clone/download the repository to a short normal Windows path, for example:

```bat
git clone https://github.com/Mstafa-awad/WTiVo-WatertightVoxel.git
cd WTiVo-WatertightVoxel-Optimizer
Setup-Windows.cmd
```

The setup script creates a private `.venv` and `.deps` inside WTiVo, installs/builds dependencies, detects the installed NVIDIA GPU compute capability, and builds the native extensions for that PC. It also collects the exact license/NOTICE files from the dependencies actually installed on that PC into `build/installed-licenses/`.

Then run:

```bat
Run-WTiVo.cmd --input "model.glb" --output "model_watertight.glb"
```

Reference benchmark settings explicitly:

```bat
Run-WTiVo.cmd ^
  --input "model.glb" ^
  --output "model_watertight.glb" ^
  --input-res 1536 ^
  --final-res 1536 ^
  --lambda_fill 10
```

A lighter first test on lower-memory hardware:

```bat
Run-WTiVo.cmd --input "model.glb" --output "model_watertight.glb" --input-res 1024 --final-res 1024 --lambda_fill 10
```

## Main arguments

| Argument | Default | Meaning |
|---|---:|---|
| `--input` | required | input mesh supported by Trimesh |
| `--output` | required | output path; `.glb` recommended |
| `--input-res` | 1536 | sparse thick field + label geometry resolution |
| `--final-res` | 1536 | final signed field + contour density |
| `--lambda_fill` | 10 | CelloCut-compatible filling regularization |
| `--threads` | logical CPU count | CPU/TBB work threads |
| `--proxy_feature_weight` | 1.5 | retention priority for strong local geometric features |
| `--gpupr_local_steps` | 8 | legal local GPU push/relabel steps before requeue |
| `--thick_band_voxels` | 3 | unsigned narrow-band width |
| `--thin_band_voxels` | 3 | signed narrow-band width |
| `--faithc_component_mode` | `auto` | `auto`, `keep_all`, or `largest` |
| `--faithc_tri_mode` | `auto` | final quad triangulation rule |

`--decimate_ratio` and arbitrary proxy budgets are deliberately absent from the production v1.0 path. The tested proxy is fixed at 12M points.

## Windows installer behavior

`Setup-Windows.cmd` is designed not to depend on the author's ComfyUI or Conda installation. It:

- finds/installs Python 3.12;
- creates `.venv`;
- installs PyTorch 2.8.0 CUDA 12.8 from the official PyTorch index;
- checks that PyTorch sees an NVIDIA GPU;
- requires CUDA Toolkit 12.8 for `nvcc`;
- selects Visual Studio **2022**, not Visual Studio 2026, for CUDA 12.8 compatibility;
- prefers/installs the known-good MSVC 14.38 toolset;
- downloads a pinned vcpkg registry release (`2026.07.29`) into `.deps`;
- builds CGAL/Eigen/OpenVDB/oneTBB locally;
- compiles `wtivo_core`, `wtivo_vdb`, and `wtivo_gpupr` for the user's machine;
- writes a local `.wtivo-env.cmd` with resolved runtime paths;
- runs native import and small topology/field sanity tests;
- collects exact installed dependency license/NOTICE files for redistribution records.

See `docs/WINDOWS.md`, `docs/VALIDATION.md`, and `docs/TROUBLESHOOTING.md` before reporting setup failures.

## License and commercial use

**Commercial use is allowed**, but WTiVo is not MIT-only. The open-source build uses CGAL 3D Triangulations, which CGAL documents as GPL. Therefore the combined open-source WTiVo distribution is **GPL-3.0-or-later**. Apache-2.0 CelloCut/FaithC-derived files keep their Apache notices and are GPLv3-compatible when distributed as part of the combined program.

You may use WTiVo inside a business, render/process commercial assets, provide paid services with it, and sell GPL-compliant distributions. If you distribute WTiVo or a modified binary, GPL source obligations apply. If you need to embed/distribute WTiVo as closed-source proprietary software, review the licenses and obtain a commercial CGAL license where appropriate.

See [`LICENSING.md`](LICENSING.md), [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md), and [`docs/PROVENANCE.md`](docs/PROVENANCE.md). This repository does not redistribute the NVIDIA CUDA Toolkit or driver.

## Attribution

WTiVo is deeply indebted to:

- **CelloCut: Constructive Watertight Remeshing via Tetrahedral Cell Cuts** — Xuan Yang et al., 2026 — https://github.com/rangeryx-66/CelloCut
- **FaithC / Faithful Contouring** — Yihao Luo et al., CVPR 2026 — https://github.com/Luo-Yihao/FaithC
- OpenVDB, CGAL, oneTBB, Eigen, PyTorch, NumPy, Trimesh, and pybind11.

Please cite the upstream papers when their methods are material to your work; see `CITATION.cff` and `NOTICE`.

## Validation and CI

Before packaging a release:

```bat
.venv\Scripts\python.exe scripts\source_audit.py
.venv\Scripts\python.exe scripts\verify_install.py
```

GitHub Actions includes lightweight source auditing on Windows and Linux plus a manual Windows native CPU/OpenVDB build workflow. Standard GitHub-hosted runners do not provide an NVIDIA GPU, so the production CUDA solver cannot be runtime-benchmarked there. A real NVIDIA Windows machine is still required for end-to-end GPU validation. The release does **not** claim that a Linux/Colab test validates Windows CUDA/MSVC behavior.

## Project policy

- No generated `.pyd`, CUDA binaries, `.venv`, `.deps`, or vcpkg build trees are committed.
- Native binaries are built on the user's PC.
- Geometry behavior is kept separate from packaging/installer changes.
- Performance claims in the README are tied to the documented benchmark machine.


## 🚀 SUPPORT MOSTAADTECH

### ❤️ Enjoying this project / workflow?

I’m **MostAadTech**, I create FREE ComfyUI workflows, local AI tools, 3D pipelines, and open-source projects.

If this project or workflow helped you, **please consider following me or supporting my work**. It helps me keep building, testing, and releasing more free tools and workflows.

---

## 💜 Support Me on Patreon

👉 **[Support MostAadTech on Patreon](https://www.patreon.com/cw/MostafaAwad/membership)**

Your support helps me spend more time developing **FREE AI tools, ComfyUI workflows, and 3D pipelines**.

---

## 🌐 Follow MostAadTech

* ▶️ **[YouTube](https://www.youtube.com/@MostAadTech)** — Tutorials, workflows & AI projects
* 📸 **[Instagram](https://www.instagram.com/mostaadtech/)** — Projects, updates & behind the scenes
* 𝕏 **[X / Twitter](https://x.com/MostAadTech)** — Updates, releases & experiments
* 💻 **[GitHub](https://github.com/Mstafa-awad)** — Open-source projects & code

---

### ⭐ One Follow Helps

**Follow • Star • Share • Support**

Every follow, GitHub star, share, and Patreon supporter helps me continue making **FREE tools for the AI community.**

**Thank you for supporting MostAadTech! ❤️**

