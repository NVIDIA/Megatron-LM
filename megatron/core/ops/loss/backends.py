# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Every vocab-parallel cross entropy backend. The contract they meet is in this package's ``__init__``."""

from __future__ import annotations

from functools import partial

import torch

from megatron.core.fusions.fused_cross_entropy import (
    fused_vocab_parallel_cross_entropy as _megatron_fused_kernel,
)
from megatron.core.tensor_parallel.cross_entropy import (
    vocab_parallel_cross_entropy as _megatron_kernel,
)

__all__ = [
    "LossMegatron",
    "LossMegatronFused",
    "LossTEFused",
    "fused_vocab_parallel_cross_entropy",
    "vocab_parallel_cross_entropy",
]


def _default_tp_group(tp_group):
    """Fill in the default tensor-parallel group for kernels that cannot take ``None``."""
    if tp_group is not None:
        return tp_group
    from megatron.core.parallel_state import get_tensor_model_parallel_group

    return get_tensor_model_parallel_group()


def vocab_parallel_cross_entropy(
    logits: torch.Tensor,
    labels: torch.Tensor,
    tp_group: torch.distributed.ProcessGroup | None = None,
) -> torch.Tensor:
    """Megatron's custom-autograd cross entropy, in family contract order."""
    return _megatron_kernel(logits, labels, tp_group=tp_group)


def fused_vocab_parallel_cross_entropy(
    logits: torch.Tensor,
    labels: torch.Tensor,
    tp_group: torch.distributed.ProcessGroup | None = None,
) -> torch.Tensor:
    """Megatron's compiled cross entropy, in family contract order.

    The kernel reads rank and size straight off the group, so unlike the reference target it
    cannot take ``None``. Filling that in is this adapter's job, not the caller's.
    """
    return _megatron_fused_kernel(logits, labels, _default_tp_group(tp_group))


def _te_cross_entropy(logits, labels, tp_group=None, *, target, cuda_graph_capturable):
    """Adapt TE's kernel to the family contract, including its label-stride requirement."""
    labels = torch.as_strided(labels, labels.size(), (labels.size()[1], 1))
    return target(logits, labels, _default_tp_group(tp_group), cuda_graph_capturable)


class LossMegatron:
    """Megatron's unfused implementation."""

    #: The target that already runs under --deterministic-mode today, since the rule in
    #: megatron/training/determinism.py requires cross entropy fusion to be off.
    DETERMINISM = "deterministic"

    def vocab_parallel_cross_entropy(self):
        """Return the reference target."""
        return vocab_parallel_cross_entropy


class LossMegatronFused:
    """Megatron's compiled reductions and custom backward."""

    #: megatron/training/determinism.py has always required cross_entropy_loss_fusion=False.
    DETERMINISM = "nondeterministic"

    def vocab_parallel_cross_entropy(self):
        """Return the fused target."""
        return fused_vocab_parallel_cross_entropy


class LossTEFused:
    """Transformer Engine's fused kernel.

    Both the TE version check and the CUDA-graph capture mode are resolved here, once, so the
    forward pass has no version branch left.
    """

    #: Covered by the same cross_entropy_loss_fusion=False rule; not separately audited.
    DETERMINISM = "nondeterministic"

    REQUIRES = "transformer_engine"

    @classmethod
    def from_options(cls, options) -> "LossTEFused":
        """Read the one setting this backend must bind up front."""
        return cls(cuda_graph_capturable=options.cuda_graph_impl == "full_iteration")

    def __init__(self, *, cuda_graph_capturable: bool = False) -> None:
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
