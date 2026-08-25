# SPDX-License-Identifier: GPL-3.0-or-later
from __future__ import annotations
import pathlib, shutil, zipfile
ROOT=pathlib.Path(__file__).resolve().parents[1]
VERSION=(ROOT/'VERSION').read_text().strip()
OUT=ROOT.parent/f'WTiVo-WatertightVoxel-Optimizer-v{VERSION}-source.zip'
EXCLUDE_DIRS={'.git','.venv','.deps','.build','build','__pycache__','.vs','.vscode'}
EXCLUDE_SUFFIX={'.pyd','.dll','.lib','.obj','.pdb','.pyc'}
with zipfile.ZipFile(OUT,'w',zipfile.ZIP_DEFLATED,compresslevel=9) as z:
    for p in sorted(ROOT.rglob('*')):
        if not p.is_file(): continue
        rel=p.relative_to(ROOT)
        if any(part in EXCLUDE_DIRS for part in rel.parts): continue
        if p.suffix.lower() in EXCLUDE_SUFFIX: continue
        z.write(p,pathlib.Path(f'WTiVo-WatertightVoxel-Optimizer-v{VERSION}')/rel)
print(OUT)
