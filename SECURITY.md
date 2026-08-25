# Security policy

Please do not attach confidential/proprietary meshes to public security reports.

For dependency vulnerabilities, identify the affected dependency/version and whether it is downloaded by WTiVo setup or part of WTiVo source. For WTiVo code issues, include the minimal reproduction and environment information without secrets.

WTiVo processes untrusted mesh files through third-party parsers and native geometry code. Run untrusted assets with normal OS protections and keep Python, Trimesh, PyTorch, vcpkg ports, NVIDIA drivers, and build tools patched.
