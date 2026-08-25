# GitHub release checklist

Before publishing a WTiVo source release:

- [ ] Run `python scripts/source_audit.py`.
- [ ] On a fresh Windows 10/11 x64 + NVIDIA machine or clean folder, run `Setup-Windows.cmd`.
- [ ] Confirm `scripts/verify_install.py` passes.
- [ ] Run one real mesh and confirm the final line reports the expected watertight status.
- [ ] Do not commit `.venv/`, `.deps/`, `.build/`, `build/`, `.pyd`, `.dll`, `.obj`, `.lib`, or `.pdb` files.
- [ ] Keep `LICENSE`, `NOTICE`, `LICENSING.md`, `THIRD_PARTY_NOTICES.md`, and `LICENSES/`.
- [ ] Tag the commit (for example `v1.0.0`) only after the Windows build/run passes.
- [ ] If publishing prebuilt dependency binaries in the future, include the exact generated `build/installed-licenses/` notices from that build and review binary redistribution terms.

The source package intentionally does not bundle CUDA Toolkit/driver binaries.
