# Validation status

WTiVo v1.0 separates **algorithm validation** from **public packaging validation** so the repository does not claim tests that were not actually performed.

## Proven production run

The production algorithm represented by this source tree was run end-to-end on the author's Windows reference machine before the public-repository cleanup:

- Windows 10 x64 build 26200.9168
- NVIDIA GeForce RTX 5050 Laptop GPU (~8 GB dedicated VRAM)
- PyTorch 2.8.0+cu128 / CUDA 12.8
- ~16 GB system RAM
- 16 CPU threads
- input/graph resolution 1536
- final resolution 1536
- fixed 12,000,000 proxy points
- `lambda_fill=10`
- graph solve about 8.76 s
- total runtime about 1.49 min
- final mesh 12,404,563 vertices / 24,809,250 faces
- exact final validation: `watertight=True`, `bad_edge_groups=0`

The development reference graph retained adjusted/raw flow `18842398/18838836` while using the faster multi-discharge schedule.

## Source-preservation checks performed for the release

During source packaging:

- the CUDA solver body was checked against the proven v6.30 implementation; after the WTiVo license/header and include-name change, the CUDA body from `#include <cuda_runtime.h>` onward is unchanged;
- the OpenVDB/FaithC finalizer source was checked against the proven v6.21 implementation; after the WTiVo header/module rename, the only normalized tail change is the Python module documentation string;
- the CGAL tetra path preserves the proven `Exact_predicates_inexact_constructions_kernel`, `Parallel_tag`, lock grid 50, input ordering, finite-cell ordering, and CelloCut-compatible neighbor-slot map `{3,2,1,0}`;
- `wtivo.py` retains the fixed 12M proxy, zero tet-label padding fix, v6.30 GPU graph path, v6.21 finalizer statistics, and mandatory final watertight result.

## Automated source-package checks

`scripts/source_audit.py` verifies:

- Python syntax;
- production-path invariants;
- required legal/provenance files;
- no generated `.pyd`, `.dll`, `.obj`, `.lib`, or `.pdb` files in the source package;
- no committed `.venv`, `.deps`, `.build`, or `build` tree;
- no author-machine development paths;
- no hidden control characters in publishable text;
- expected GPL/Apache SPDX/modified-file notices.

GitHub Actions runs this audit on both Linux and Windows. A separate manual Windows workflow builds/imports the CPU/CGAL/OpenVDB native extensions.

## What has not been falsely claimed

The source-release packaging environment used to assemble this repository is not a Windows NVIDIA CUDA host, so it cannot truthfully certify a fresh MSVC/CUDA build of this cleaned source tree by itself. Google Colab is Linux and would not validate the Windows MSVC/DLL path either. Standard GitHub-hosted Windows runners do not provide an NVIDIA CUDA GPU.

Therefore the final portability gate for a tagged release is intentionally:

1. clone the repository into a fresh Windows folder;
2. run `Setup-Windows.cmd`;
3. pass `scripts/verify_install.py`;
4. run at least one real mesh end-to-end on an NVIDIA Windows machine.

This limitation is documented rather than hidden. See `docs/RELEASE_CHECKLIST.md`.
