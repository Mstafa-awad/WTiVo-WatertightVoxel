# Benchmarks

## Reference production result

The release benchmark is the final v6.33.1-equivalent WTiVo production path.

| Metric | Result |
|---|---:|
| Input/graph resolution | 1536 |
| Final resolution | 1536 |
| Point budget | 12,000,000 |
| lambda_fill | 10 |
| GPU graph solve | ~8.76 s |
| Total runtime | ~1.49 min |
| Final vertices | 12,404,563 |
| Final faces | 24,809,250 |
| Watertight | True |
| Bad edge groups | 0 |

Test system: Windows 10 build 26200.9168, NVIDIA RTX 5050 Laptop GPU (~8 GB dedicated VRAM), PyTorch 2.8.0+cu128, CUDA 12.8, ~16 GB system RAM, 16 CPU threads.

## Interpreting the numbers

These numbers are not universal. Runtime changes with surface area, triangle count, OpenVDB active voxels, Delaunay complexity, graph size, CPU speed, storage/memory pressure, and GPU. Do not quote the 1.49-minute number without the hardware/settings above.

The graph optimization was validated during development by retaining the reference adjusted/raw flow `18842398/18838836` on the benchmark asset while reducing solver time to the ~9-second range.

The final manifold repair on that run reduced residual non-manifold edge groups to zero and the independent edge-degree checker reported `watertight=True`.
