# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

"""Selection of the vocab-parallel cross entropy a model binds."""

from __future__ import annotations

from functools import partial
from typing import Callable, Optional

import torch

from megatron.core.extensions.transformer_engine import te_cross_entropy
from megatron.core.ops._backends import require
from megatron.core.ops.loss.cross_entropy import unfused_cross_entropy
from megatron.core.ops.loss.fused_cross_entropy import (
    fused_vocab_parallel_cross_entropy as _fused_ce,
)
from megatron.core.utils import is_te_min_version

CrossEntropyTarget = Callable[
    [torch.Tensor, torch.Tensor, Optional[torch.distributed.ProcessGroup]], torch.Tensor
]
"""``loss = target(logits, labels, tp_group)``, with ``logits`` [s, b, vocab/tp] and the
target owning every reduction over ``tp_group``. Targets may *consume* ``logits``, replacing
them with softmax intermediates or gradients, so a caller that needs them afterwards passes
a clone. Targets supporting ``return_argmax=True`` return ``(loss, original_global_argmax)``,
with ``None`` predictions when unavailable.
"""


def select_cross_entropy(
    cross_entropy_loss_fusion: bool = False,
    cross_entropy_fusion_impl: str = "native",
    cuda_graph_impl: Optional[str] = None,
) -> CrossEntropyTarget:
    """Which cross entropy a config asks for, resolved once rather than on every forward.

    Shared by every provider: the choice depends on the config, not on which backend supplies
    the rest of the model, which is how it behaved before this was a provider decision.
    """
    if cross_entropy_fusion_impl not in ("native", "te", "liger"):
        raise ValueError(
            f"unknown cross_entropy_fusion_impl={cross_entropy_fusion_impl!r}. "
            "Valid choices: native, te, liger"
        )
    if not cross_entropy_loss_fusion:
        return unfused_cross_entropy
    if cross_entropy_fusion_impl == "native":
        return _fused_ce
    if cross_entropy_fusion_impl == "liger":
        require(
            "liger_kernel.megatron",
            "LigerMegatronCrossEntropy",
            needed_by="Liger vocab-parallel cross entropy",
        )
        from megatron.core.ops.loss.liger_cross_entropy import liger_cross_entropy

        return liger_cross_entropy

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
