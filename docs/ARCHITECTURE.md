# WTiVo architecture

## 1. Input

WTiVo loads a triangle mesh with Trimesh using `process=False`, preserving the provided geometry as much as the loader allows. Scenes are concatenated before reconstruction.

## 2. Sparse thick field

The input is normalized into the WTiVo unit-domain mapping. `wtivo_vdb.SparseUDF` creates a global sparse unsigned OpenVDB narrow-band field. WTiVo does not construct a dense `R^3` NumPy/Torch field.

The geometry inset remains `EPS = 1 / input_res`. A crucial production fix is that tet-centroid OpenVDB label queries use `query_padding = 0`; applying EPS again at label-query time produced a measurable mismatch and worse surfaces during development.

## 3. Direct point-budget proxy

Instead of decoding a complete thick proxy surface and QEM-decimating it, WTiVo asks the contour bridge for candidate QEF/FCT anchors and directly selects up to the fixed production target of 12,000,000 points. Strong local features receive extra retention priority (`proxy_feature_weight`, default 1.5).

No proxy triangles enter CGAL. No QEM pass runs in the production path.

## 4. CGAL tetrahedralization

`wtivo_core.tetrahedralize_neighbors()` uses CGAL's 3D Delaunay triangulation with `Parallel_tag` and oneTBB. It preserves the historical CelloCut face-slot order `012, 013, 023, 123`; CGAL neighbor slots are mapped with `{3,2,1,0}`. The lock grid is 50, matching the proven production topology path.

The function exports tetrahedra and their four neighbors directly, removing the need for a separate massive face-hash adjacency build.

## 5. Cell labels

The thick sparse field samples each tetrahedron centroid. The inside/outside seed mapping and `lambda_fill` graph capacities retain the CelloCut-compatible cell-cut objective.

## 6. GPU reduced graph

Tetrahedra and neighbors are uploaded to CUDA in small host-staging chunks. `wtivo_gpupr` builds the exact reduced graph and solves it using WTiVo's multi-discharge CUDA Push-Relabel schedule.

The optimization versus the earlier solver is scheduling, not a different energy: an active CUDA thread can perform several legal push/relabel operations before it is requeued. Periodic global relabeling remains, and an exact residual BFS determines the final partition.

The tested default is `gpupr_local_steps=8`.

## 7. Cut-surface extraction

The final tet labels define boundary faces. The GPU-resident topology is streamed in chunks to materialize the oriented graph-cut surface without first restoring the full topology as another duplicate host array.

## 8. Signed sparse field and final contour

The cut surface is converted to a signed OpenVDB narrow-band level set. The FaithC-style decoder creates one or more topology vertices where local sign ambiguity requires them.

The manifold finalizer is intentionally local:

- ambiguous cell crossing components can receive duplicated topology tokens at the same QEF coordinate;
- residual degree>2 edge fans can be separated locally;
- only tiny closed degree-1 loops within strict size bounds are capped;
- no global Laplacian/Taubin smoothing is performed.

## 9. Validation

WTiVo sorts packed undirected triangle edges and requires every edge group to have degree exactly 2. The result is printed as:

```text
[FINAL] watertight=True | bad_edge_groups=0
```

This audit is topology-based. It does not claim the output is artist-retopologized, low-poly, UV-preserving, or animation-ready.
