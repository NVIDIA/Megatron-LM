# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Transformer Engine vocab-parallel cross entropy."""

from __future__ import annotations

from functools import partial

import torch

from megatron.core.ops import _availability

__all__ = ["TECrossEntropyBackend"]

_BACKEND_NAME = "te_fused_cross_entropy"


def _te_cross_entropy(
    logits: torch.Tensor,
    labels: torch.Tensor,
    tp_group: torch.distributed.ProcessGroup | None = None,
    *,
    target,
    cuda_graph_capturable: bool,
) -> torch.Tensor:
    """Adapt TE's kernel to the family contract, including its label-stride requirement."""
    if tp_group is None:
        from megatron.core.parallel_state import get_tensor_model_parallel_group

        tp_group = get_tensor_model_parallel_group()
    labels = torch.as_strided(labels, labels.size(), (labels.size()[1], 1))
    return target(logits, labels, tp_group, cuda_graph_capturable)


class TECrossEntropyBackend:
    """Owns ``vocab_parallel_cross_entropy`` using Transformer Engine's fused kernel.

    Both the TE version check and the CUDA-graph capture mode are resolved here, once, so the
    forward pass has no version branch left.
    """

    def __init__(self, *, cuda_graph_capturable: bool = False) -> None:
        _availability.require("transformer_engine", backend=_BACKEND_NAME)
        from megatron.core.extensions.transformer_engine import te_parallel_cross_entropy

        if te_parallel_cross_entropy is None:
            raise ImportError(
                "Transformer Engine is installed but does not expose a parallel cross entropy. "
                "Use --cross-entropy-fusion-impl native instead."
            )
        if cuda_graph_capturable:
            from megatron.core.utils import get_te_version, is_te_min_version

            if not is_te_min_version("2.7.0"):
                raise ValueError(
                    "CUDA graph compatible cross entropy requires TransformerEngine >= 2.7.0, "
                    f"but found version {get_te_version()}. Please upgrade TransformerEngine "
                    "or set cuda_graph_impl to a value other than 'full_iteration'."
                )
        self._target = te_parallel_cross_entropy
        self._cuda_graph_capturable = cuda_graph_capturable

    def vocab_parallel_cross_entropy(self):
        """Return the TE target with its construction-time choices already bound."""
        return partial(
            _te_cross_entropy,
            target=self._target,
            cuda_graph_capturable=self._cuda_graph_capturable,
        )
