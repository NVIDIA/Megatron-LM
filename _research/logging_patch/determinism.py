"""Deterministic run configuration.

Opt-in via ``APERTUS_DETERMINISTIC=1``. Call ``set_deterministic()`` from
``install()`` before Megatron builds the model. Seeds are taken from
``--seed`` inside Megatron; we only flip the global flags that disable
nondeterministic kernels.

Note: ``torch.use_deterministic_algorithms(True)`` requires
``CUBLAS_WORKSPACE_CONFIG`` to be set before any CUDA kernel launch, so this
must run early (before ``import torch.cuda`` side effects).
"""

from __future__ import annotations

import os


def set_deterministic() -> bool:
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    os.environ.setdefault("CUDNN_DETERMINISTIC", "1")
    try:
        import torch
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        return False
    return True
