# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

from typing import Literal, Optional, Tuple, Union

import torch

from megatron.core import tensor_parallel
from megatron.core.fusions.fused_cross_entropy import fused_vocab_parallel_cross_entropy
from megatron.core.fusions.fused_linear_cross_entropy import linear_cross_entropy
from megatron.core.transformer.enums import CudaGraphScope
from megatron.core.utils import is_te_min_version

try:
    from megatron.core.extensions.transformer_engine import te_parallel_cross_entropy
except:
    te_parallel_cross_entropy = None


class LinearCrossEntropyModule(tensor_parallel.ColumnParallelLinear):
    """
    A module that combines a ColumnParallelLinear layer with fused
    linear + cross-entropy loss computation over a tensor-parallel vocabulary.
    """

    def forward(
        self,
        input_: torch.Tensor,
        weight: Optional[torch.Tensor] = None,
        runtime_gather_output: Optional[bool] = None,
        output_cross_entropy_loss: bool = False,
        labels: Optional[torch.Tensor] = None,
        reduction: Literal["none", "sum", "mean"] = "none",
        ignore_index: int = -100,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Optional[torch.Tensor]]]:
        """Run either the plain ColumnParallelLinear or fused linear+cross-entropy."""
        if output_cross_entropy_loss:
            assert labels is not None, "labels cannot be None when outputting cross-entropy loss."
            return self._compute_linear_and_cross_entropy_loss(
                hidden=input_,
                weight=weight if weight is not None else self.weight,
                labels=labels,
                reduction=reduction,
                ignore_index=ignore_index,
            )

        # Fall back to standard ColumnParallelLinear forward.
        # ColumnParallelLinear.forward returns (output, bias) or just output
        # depending on configuration, so keep the return type as Tensor.
        return super().forward(input_, weight, runtime_gather_output)

    def _compute_linear_and_cross_entropy_loss(
        self,
        hidden: torch.Tensor,
        weight: torch.Tensor,
        runtime_gather_output: Optional[bool] = None,
        labels: Optional[torch.Tensor] = None,
        reduction: Literal["none", "sum", "mean"] = "none",
        ignore_index: int = -100,
    ) -> torch.Tensor:
        """Compute fused linear + cross-entropy over tensor-parallel vocab."""
        if (
            self.config.cross_entropy_loss_fusion
            and self.config.cross_entropy_fusion_impl == 'linear'
        ):
            assert (
                weight is not None
            ), "weight cannot be None when using fused linear cross entropy."
            assert (
                labels is not None
            ), "labels cannot be None when using fused linear cross entropy."

            # [b s] => [s b]
            labels = labels.transpose(0, 1).contiguous()
            loss = linear_cross_entropy(
                hidden,
                self.weight,
                labels,
                sequence_parallel=self.sequence_parallel,
                reduction=reduction,
                ignore_index=ignore_index,
                tp_group=self.tp_group,
            )
            # If reduction != "none" this will be a scalar; for "none" it should
            # match [s, b] and can be reshaped back to [b, s].
            if reduction == "none":
                loss = loss.view_as(labels).transpose(0, 1).contiguous()
        else:
            logits, _ = super().forward(hidden, weight, runtime_gather_output)
            loss = self._compute_cross_entropy_loss(labels, logits)

        return loss

    def _compute_cross_entropy_loss(
        self, labels: torch.Tensor, logits: torch.Tensor
    ) -> Optional[torch.Tensor]:
        """Compute (possibly fused) vocab-parallel cross-entropy loss."""
        loss = None

        # [b s] => [s b]
        labels = labels.transpose(0, 1).contiguous()
        if self.config.cross_entropy_loss_fusion:
            if self.config.cross_entropy_fusion_impl == 'te':
                if te_parallel_cross_entropy is not None:
                    labels = torch.as_strided(labels, labels.size(), (labels.size()[1], 1))
                    # Use is_cg_capturable=True for full iteration CUDA graphs
                    # to avoid torch.equal checks
                    is_cg_capturable = (
                        hasattr(self.config, 'cuda_graph_scope')
                        and CudaGraphScope.full_iteration in self.config.cuda_graph_scope
                    )
                    if is_cg_capturable and not is_te_min_version("2.7.0"):
                        from megatron.core.utils import get_te_version

                        current_version = get_te_version()
                        raise AssertionError(
                            f"CUDA graph compatible cross entropy requires "
                            f"TransformerEngine >= 2.7.0, but found version {current_version}. "
                            "Please upgrade TransformerEngine "
                            f"or set cuda_graph_scope to a value other than 'full_iteration'."
                        )

                    loss = te_parallel_cross_entropy(
                        logits, labels, self.tp_group, is_cg_capturable
                    )
                else:
                    raise RuntimeError("Trying to use a TE block when it's not present.")
            elif self.config.cross_entropy_fusion_impl == 'native':
                loss = fused_vocab_parallel_cross_entropy(logits, labels, self.tp_group)
        else:
            loss = tensor_parallel.vocab_parallel_cross_entropy(logits, labels)

        # [s b] => [b, s]
        loss = loss.transpose(0, 1).contiguous()
        return loss
