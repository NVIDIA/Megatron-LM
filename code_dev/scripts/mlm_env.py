"""Import THIS FIRST in any container-side MLM script (TPSP worktree copy).

Identical to the HYBRID mlm_env but points MLM at the gemma4-e4b-tp-sp WORKTREE
so TP/SP code changes on this branch are the ones exercised. Also populates
nvidia_resiliency_ext.__version__ before megatron import (container shim).
"""
import sys

MLM = "/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_genai/users/ataghibakhsh/Gemma4_mlm/Megatron-LM-tpsp"
if MLM not in sys.path:
    sys.path.insert(0, MLM)

try:
    import nvidia_resiliency_ext as _nvrx  # noqa: E402
    if not hasattr(_nvrx, "__version__"):
        v = None
        try:
            from importlib.metadata import version
            v = version("nvidia_resiliency_ext")
        except Exception:
            v = None
        try:
            from packaging.version import Version as _V
            if v is None or _V(v) < _V("0.6.0"):
                v = "0.6.0"
        except Exception:
            v = v or "0.6.0"
        _nvrx.__version__ = v
except Exception:
    pass
