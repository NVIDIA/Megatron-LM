# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Bit-exact correctness tests for ``--deterministic-mode``.

Two-run comparison + parametrize over preset × parallelism. The package
import sets the determinism env vars eagerly so cuBLAS / TE / NCCL capture
them at their respective first-use sites. ``apply_determinism_env`` uses
``setdefault`` — if pytest's collection has already touched CUDA via
another module before this package imports, the writes silently no-op and
the launcher's shell-side exports are what actually take effect (the CI
recipe relies on this defense-in-depth).
"""

import os

from megatron.training.determinism import apply_determinism_env

# Same vLLM guard as ``kernels/__init__.py``: importing vLLM sets TRITON_CACHE_AUTOTUNING=1,
# which apply_determinism_env rejects without a shared TRITON_CACHE_DIR.
if os.environ.get("TRITON_CACHE_AUTOTUNING", "1") == "1" and not os.environ.get("TRITON_CACHE_DIR"):
    os.environ["TRITON_CACHE_AUTOTUNING"] = "0"
apply_determinism_env(os.environ)
