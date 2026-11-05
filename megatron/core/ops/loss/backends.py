# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
# License for these changes: BSD-3-Clause.

"""Selection of the vocab-parallel cross entropy a model binds."""

from __future__ import annotations

from functools import partial
from typing import Callable, Optional

import torch

from megatron.core.extensions.transformer_engine import te_cross_entropy
from megatron.core.ops.loss.cross_entropy import unfused_cross_entropy
from megatron.core.ops.loss.fused_cross_entropy import (
    fused_vocab_parallel_cross_entropy as _fused_ce,
)
from megatron.core.utils import is_te_min_version

CrossEntropyTarget = Callable[
    [torch.Tensor, torch.Tensor, Optional[torch.distributed.ProcessGroup]], torch.Tensor
]
"""``loss = target(logits, labels, tp_group)``, with ``logits`` [s, b, vocab/tp] and the
target owning every reduction over ``tp_group``. Existing native/explicit-TE targets may
consume ``logits`` in place. The opt-in ``tiled_te`` target preserves forward logits.
"""


def _te_has_memory_efficient_cross_entropy() -> bool:
    """Check TE 2.19's tiled/statistics CE and input-ownership API once at binding.

    TE PR #3273 introduced both in 2.19. Match training's 2.19 stable safety
    floor and probe the API too, since version metadata alone is insufficient.
    """
    from megatron.core.extensions import transformer_engine as te_extensions

    return (
        is_te_min_version("2.19.0")
        and te_extensions.te_parallel_cross_entropy is not None
        and getattr(te_extensions, "_TE_FUSED_PARALLEL_CE_OVERWRITE_INPUT", False)
    )


def select_cross_entropy(
    cross_entropy_loss_fusion: bool = False,
    cross_entropy_fusion_impl: str = "native",
    cuda_graph_impl: Optional[str] = None,
    *,
    inplace_backward: bool = False,
) -> CrossEntropyTarget:
    """Which cross entropy a config asks for, resolved once rather than on every forward.

    Shared by every provider: the choice depends on the config, not on which backend supplies
    the rest of the model, which is how it behaved before this was a provider decision.
    """
    if inplace_backward and not (
        cross_entropy_loss_fusion and cross_entropy_fusion_impl == "tiled_te"
    ):
        raise ValueError("Private logits backward requires fused tiled_te cross entropy")
    if cross_entropy_fusion_impl not in ("native", "te", "tiled_te"):
        raise ValueError(
            f"unknown cross_entropy_fusion_impl={cross_entropy_fusion_impl!r}. "
            "Valid choices: native, te, tiled_te"
        )
    if not cross_entropy_loss_fusion:
        return unfused_cross_entropy
    if cross_entropy_fusion_impl == "native":
        return _fused_ce

    if cross_entropy_fusion_impl == "tiled_te":
        if _te_has_memory_efficient_cross_entropy():
            return partial(
                te_cross_entropy,
                cuda_graph_capturable=cuda_graph_impl == "full_iteration",
                overwrite_input=inplace_backward,
            )
        from megatron.core.ops.loss.tiled_cross_entropy import triton_cross_entropy

        if inplace_backward:
            return partial(triton_cross_entropy, inplace_backward=True)
        return triton_cross_entropy

    # Full-iteration CUDA graphs capture the loss too, which TE only supports from 2.7.0.
    capturable = cuda_graph_impl == "full_iteration"
    if capturable and not is_te_min_version("2.7.0"):
        from megatron.core.utils import get_te_version

        raise AssertionError(
            "CUDA graph compatible cross entropy requires TransformerEngine >= 2.7.0, but "
            f"found version {get_te_version()}. Please upgrade TransformerEngine or set "
            "cuda_graph_impl to a value other than 'full_iteration'."
        )
    return partial(te_cross_entropy, cuda_graph_capturable=capturable)
