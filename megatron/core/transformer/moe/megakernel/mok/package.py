# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""The MOK implementations behind MCore's megakernel backends.

``moe_megakernel_backend="cutemok"`` calls cuDNN frontend's CuTe MoK (``cudnn.moe_ep.cutemok``,
added in https://github.com/NVIDIA/cudnn-frontend/pull/2767) through its public API. CuTe MoK is
an optimized CuTe DSL version of MOK and is expected to outperform the standalone ``mok`` package.

``moe_megakernel_backend="mok"`` is deprecated. It runs as ``"cutemok"`` when CuTe MoK is
available, and otherwise calls the standalone ``mok`` package, our MCore-integration version of
MOK (https://github.com/QiZhangNV/mixture-of-kittens/tree/qizhang/mcore-integration), through its
``functional`` layer interface; either way with a FutureWarning.

Both share this adapter.
"""

import functools
import sys
import warnings
from types import ModuleType

import torch

_CUTEMOK_GPUS = ((10, 0), (10, 3))  # CuTe MoK runs on SM100 and SM103 GPUs only.


def cutemok_api() -> ModuleType:
    """Return cuDNN frontend's public CuTe MoK module (``moe_megakernel_backend="cutemok"``)."""
    from cudnn.moe_ep import cutemok

    return cutemok


def mok_modules() -> tuple[ModuleType, ModuleType]:
    """Return the ``mok`` package's ``(functional, ops)`` (``moe_megakernel_backend="mok"``)."""
    from mok import functional, ops

    return functional, ops


def clear_mok_caches() -> None:
    """Drop the cached CuTe MoK plans and, if the mok package is loaded, its workspaces."""
    from megatron.core.transformer.moe.megakernel.mok.runtime import clear_cutemok_plans

    clear_cutemok_plans()
    if "mok.functional" in sys.modules:
        sys.modules["mok.functional"].clear_workspace_cache()


@functools.lru_cache(maxsize=None)
def cutemok_available() -> bool:
    """Whether cuDNN frontend's CuTe MoK can run here: on its GPUs and with its CuTe DSL."""
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() not in _CUTEMOK_GPUS:
        return False
    try:
        cutemok_api()
    except ImportError:
        return False
    # CuTe MoK checks its CuTe DSL only when it compiles, so check it now with cuDNN's own gate.
    # The gate is private: if it cannot be imported, treat CuTe MoK as unavailable.
    try:
        from cudnn.moe_ep._cutemok.cutedsl._version_gate import require_cutedsl
    except ImportError:
        return False
    try:
        require_cutedsl()
    except RuntimeError:
        return False
    return True


def resolve_mok_backend(backend: str) -> str:
    """Return the implementation a backend name runs; deprecated "mok" runs "cutemok" if it can."""
    if backend != "mok":
        return backend
    if cutemok_available():
        warnings.warn(
            'moe_megakernel_backend="mok" is deprecated and runs as "cutemok" (cuDNN frontend\'s '
            'CuTe MoK, an optimized version of MOK); use "cutemok" instead.',
            FutureWarning,
            stacklevel=2,
        )
        return "cutemok"
    warnings.warn(
        'moe_megakernel_backend="mok" is deprecated while cuDNN frontend\'s CuTe MoK '
        "(cudnn.moe_ep.cutemok, which needs an SM100/SM103 GPU and its CuTe DSL) is not "
        "available here, so it falls back to the standalone mok package; "
        'Suggest installing cuDNN frontend with CuTe MoK and use "cutemok" instead.',
        FutureWarning,
        stacklevel=2,
    )
    return "mok"
