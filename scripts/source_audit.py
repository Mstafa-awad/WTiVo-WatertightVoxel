# SPDX-License-Identifier: GPL-3.0-or-later
from __future__ import annotations
import ast, pathlib, re, sys
ROOT=pathlib.Path(__file__).resolve().parents[1]
errors=[]
def ok(cond,msg):
    if not cond: errors.append(msg)
def text(rel): return (ROOT/rel).read_text(encoding='utf-8-sig' if rel.endswith('.ps1') else 'utf-8')

required=['README.md','LICENSE','LICENSING.md','NOTICE','THIRD_PARTY_NOTICES.md','wtivo.py','CMakeLists.txt','vcpkg.json','Setup-Windows.cmd','Run-WTiVo.cmd','native/core/wtivo_core.cpp','native/vdb/wtivo_vdb.cpp','native/gpupr/wtivo_gpupr_bindings.cpp','native/gpupr/gpu_push_relabel_fast.cu','native/gpupr/gpu_push_relabel_fast.h','scripts/setup_windows.ps1','scripts/build_gpupr.py','scripts/verify_install.py']
for r in required: ok((ROOT/r).is_file(),f'missing required file: {r}')

for r in ['wtivo.py','scripts/build_gpupr.py','scripts/verify_install.py','scripts/source_audit.py','scripts/package_release.py']:
    p=ROOT/r
    if p.exists():
        try: ast.parse(p.read_text(encoding='utf-8'))
        except SyntaxError as e: errors.append(f'{r}: Python syntax: {e}')

w=text('wtivo.py')
ok('PROXY_POINTS = 12_000_000' in w,'fixed 12M proxy missing')
ok('LABEL_QUERY_PADDING = 0.0' in w,'label-query padding fix missing')
ok('--decimate_ratio' in w and 'intentionally unavailable' in w,'production no-ratio marker missing')
ok('gpupr.graph_cut_fast(' in w,'fast GPU graph path missing')
ok('surface_extraction_topology_cuda' in w,'GPU topology surface extraction missing')
ok('FaithC-FINALIZE-v6.21' in w,'watertight finalizer stats missing')
ok('[FINAL] watertight=' in w,'mandatory final watertight print missing')

core=text('native/core/wtivo_core.cpp')
for needle in ['CGAL::Parallel_tag','neighbor_lut[4] = {3, 2, 1, 0}','50);','tetrahedralize_neighbors','is_watertight']:
    ok(needle in core,f'core invariant missing: {needle}')
ok(core.count('const std::size_t NV = static_cast<std::size_t>(vertices.shape(0));')==1,'duplicate NV declaration detected')

vdb=text('native/vdb/wtivo_vdb.cpp')
for needle in ['faithc_point_budget','sample_tet_labels','faithc_mesh','repair_bad_final','PYBIND11_MODULE(wtivo_vdb']:
    ok(needle in vdb,f'VDB/finalizer invariant missing: {needle}')

gpu=text('native/gpupr/gpu_push_relabel_fast.cu')
for needle in ['local_steps','global_relabel','residual','cellocut_gpu_push_relabel_fast']:
    ok(needle.lower() in gpu.lower(),f'GPU solver invariant missing: {needle}')

# No machine-specific absolute paths or hidden control characters in publishable text.
text_suffixes={'.md','.txt','.py','.ps1','.cmd','.yml','.yaml','.json','.cff','.cpp','.cu','.h','.cmake'}
for p in ROOT.rglob('*'):
    if not p.is_file():
        continue
    if p.suffix.lower() in text_suffixes or p.name in {'CMakeLists.txt','LICENSE','NOTICE','.gitignore','.gitattributes','VERSION'}:
        data=p.read_bytes()
        bad=[b for b in data if b < 32 and b not in (9,10,13)]
        ok(not bad,f'control character found in {p.relative_to(ROOT)}')
        try:
            decoded=data.decode('utf-8-sig')
        except UnicodeDecodeError:
            continue
        if p.resolve() != (ROOT/'scripts/source_audit.py').resolve():
            markers=('C:'+chr(92)+'Users'+chr(92)+'super', 'OneDrive'+chr(92)+'Desktop'+chr(92)+'CelloCut - Copy', 'python'+'_embeded')
            for marker in markers:
                ok(marker not in decoded,f'machine-specific development path in {p.relative_to(ROOT)}')

# Public release metadata / licensing files must be present.
for r in ['docs/PROVENANCE.md','docs/RELEASE_CHECKLIST.md','docs/VALIDATION.md','docs/WINDOWS.md','docs/BENCHMARKS.md','docs/TROUBLESHOOTING.md','scripts/collect_dependency_licenses.py','CITATION.cff','VERSION']:
    ok((ROOT/r).is_file(),f'missing release/legal file: {r}')

# No generated binaries/build dirs in a source release.
for p in ROOT.rglob('*'):
    if p.is_file() and p.suffix.lower() in {'.pyd','.dll','.obj','.lib','.pdb'}: errors.append(f'generated binary committed: {p.relative_to(ROOT)}')
for d in ['.venv','.deps','.build','build','__pycache__']:
    ok(not (ROOT/d).exists(),f'generated directory present in source package: {d}')

# License checks.
ok('GNU GENERAL PUBLIC LICENSE' in text('LICENSE'),'root LICENSE is not GPLv3 text')
for r in ['wtivo.py','native/core/wtivo_core.cpp','native/vdb/wtivo_vdb.cpp','native/gpupr/wtivo_gpupr_bindings.cpp','native/gpupr/gpu_push_relabel_fast.cu','native/gpupr/gpu_push_relabel_fast.h']:
    s=text(r); ok('SPDX-License-Identifier: Apache-2.0' in s,f'{r}: Apache SPDX missing'); ok('Modified for WTiVo in 2026' in s,f'{r}: prominent modification notice missing')

if errors:
    print('WTiVo SOURCE AUDIT: FAIL')
    for e in errors: print(' -',e)
    raise SystemExit(1)
print('WTiVo SOURCE AUDIT: PASS')
print(' - Python syntax OK')
print(' - production invariants present')
print(' - no generated native binaries committed')
print(' - GPL/Apache source markers present')
