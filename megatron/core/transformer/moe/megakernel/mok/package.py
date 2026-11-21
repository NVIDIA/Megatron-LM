# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""The MOK implementations behind MCore's two megakernel backends.

``moe_megakernel_backend="mok"`` calls the standalone ``mok`` package, our MCore-integration
version of MOK (https://github.com/QiZhangNV/mixture-of-kittens/tree/qizhang/mcore-integration),
through its ``functional`` layer interface.

``moe_megakernel_backend="cutemok"`` calls cuDNN frontend's CuTe MoK (``cudnn.moe_ep.cutemok``,
added in https://github.com/NVIDIA/cudnn-frontend/pull/2767) through its public API. CuTe MoK is
an optimized CuTe DSL version of MOK and is expected to outperform ``mok``.

Both backends share this adapter.
"""

from types import ModuleType


def cutemok_api() -> ModuleType:
    """Return cuDNN frontend's public CuTe MoK module (``moe_megakernel_backend="cutemok"``)."""
    from cudnn.moe_ep import cutemok

    return cutemok


def mok_modules() -> tuple[ModuleType, ModuleType]:
    """Return the ``mok`` package's ``(functional, ops)`` (``moe_megakernel_backend="mok"``)."""
    from mok import functional, ops

    return functional, ops


def clear_mok_caches(backend: str) -> None:
    """Drop the cached CuTe MoK plans (``cutemok``) or ``mok`` workspaces (``mok``)."""
    if backend == "cutemok":
        from megatron.core.transformer.moe.megakernel.mok.runtime import clear_cutemok_plans

        clear_cutemok_plans()
    else:
        mok_modules()[0].clear_workspace_cache()
