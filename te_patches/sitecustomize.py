"""Auto-executed by Python at startup. Redirects TE's context_parallel to patched version."""
import sys, importlib.util, os

_PATCHED = os.path.join(os.path.dirname(os.path.abspath(__file__)), "context_parallel.py")
_MODULE = "transformer_engine.pytorch.attention.dot_product_attention.context_parallel"

class _TeCPPatcher:
    def find_spec(self, fullname, path, target=None):
        if fullname == _MODULE:
            return importlib.util.spec_from_file_location(fullname, _PATCHED)
        return None

sys.meta_path.insert(0, _TeCPPatcher())
