"""sitecustomize.py loaded via PYTHONPATH prepend during cache bootstrap.

Loaded automatically by Python's `site` module before user code runs. Two jobs:

1. Install the Triton autotune-disk-cache `pre_hook` patch. See
   `triton_autotune_disk_cache.py` in the parent directory for what it does.
2. Re-execute the container's existing `/usr/lib/python3.12/sitecustomize.py`
   so we don't drop the Ubuntu `apport_python_hook` handler. The container's
   sitecustomize is small (6 lines) — apport handles uncaught exceptions, not
   critical, but worth preserving so we don't silently change crash behavior.

Disable with `TRITON_AUTOTUNE_PREHOOK_PATCH_DISABLE=1` (the patch's own kill
switch). This file remains a no-op for the Triton install when set.
"""

import os
import sys

# (1) Arm + install our Triton autotune patch. `arm()` registers a
# MetaPathFinder so the patch lands on the first `import triton.runtime.autotuner`
# regardless of import order — this is the fix for the failure mode where
# `install()` alone ran before Triton was importable and silently no-op'd.
# `install()` covers the rare case where Triton was already pulled in.
try:
    _shim_dir = os.path.dirname(os.path.abspath(__file__))
    _pc_dir = os.path.dirname(_shim_dir)
    if _pc_dir not in sys.path:
        sys.path.insert(0, _pc_dir)
    import triton_autotune_disk_cache  # noqa: E402
    triton_autotune_disk_cache.arm()
    triton_autotune_disk_cache.install()
except Exception:
    pass

# (2) Chain to the container's apport sitecustomize. Use runpy to execute the
# file by path so it works whether or not the container changes Python version.
try:
    import runpy
    _apport_path = "/usr/lib/python3.12/sitecustomize.py"
    if os.path.exists(_apport_path):
        runpy.run_path(_apport_path, run_name="__sitecustomize_chain__")
except Exception:
    pass
