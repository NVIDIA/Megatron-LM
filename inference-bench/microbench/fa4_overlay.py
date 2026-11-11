"""Import a newer FlashAttention-4 from a private pip --target directory instead of the container's.

    FA4_OVERLAY=<dir> python3 microbench/fa4_decode_splits_bench.py ...

Must be imported before anything imports flash_attn. Two things stop a plain PYTHONPATH from
working: the container's FA2 `flash_attn` is a regular package and shadows the FA4 wheel's
`flash_attn/cute` (a namespace subpackage), and nvidia-cutlass-dsl adds its modules through a .pth
file, which only site directories process.
"""

import os
import sys
import types

overlay = os.environ.get("FA4_OVERLAY")
if overlay:
    # The overlay's cutlass DSL packages (what its .pth file adds: dsl_packages in 4.8,
    # python_packages before) go ahead of the container's copies, already on sys.path.
    extra = [overlay] + [
        os.path.join(overlay, "nvidia_cutlass_dsl", d)
        for d in ("dsl_packages", "python_packages")
        if os.path.isdir(os.path.join(overlay, "nvidia_cutlass_dsl", d))
    ]
    # The overlay's cutlass is a namespace package, which any regular package of that name later
    # on sys.path wins over, so the container's DSL packages leave sys.path.
    sys.path[:] = extra + [
        p for p in sys.path if os.path.basename(p) not in ("python_packages", "dsl_packages")
    ]
    for module in [m for m in sys.modules if m == "cutlass" or m.startswith("cutlass.")]:
        del sys.modules[module]
    package = types.ModuleType("flash_attn")
    package.__path__ = [os.path.join(overlay, "flash_attn")]
    sys.modules["flash_attn"] = package
