# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

from typing import Literal, Optional, Tuple, Union

import torch

from megatron.core import tensor_parallel
from megatron.core.fusions.fused_linear_cross_entropy import linear_cross_entropy
from megatron.core.transformer.padding_aware_gemm import (
    current_live_rows_for,
    lm_head_wgrad_uses_grouped,
    padding_aware_linear,
    static_capacity_rows,
)


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

        live_rows = (
            current_live_rows_for(input_.shape[:-1].numel())
            if self.config.padding_aware_lm_head
            else None
        )
        if (
            live_rows is not None
            and not runtime_gather_output
            and not self.gather_output
            and not self.sequence_parallel
            and self.bias is None
            and self.embedding_activation_buffer is None
            and not self.explicit_expert_comm
        ):
            # Padding-aware LM head: the GEMM covers only the live rows of the THD buffer.
            weight = weight if weight is not None else self.weight
            output = padding_aware_linear(
                input_,
                weight,
                live_rows,
                gradient_accumulation_fusion=self.gradient_accumulation_fusion,
                allreduce_dgrad=self.allreduce_dgrad,
                tp_group=self.tp_group,
                wgrad_grouped=lm_head_wgrad_uses_grouped(
                    static_capacity_rows(self.config) or input_.shape[:-1].numel()
                ),
            )
            return output, None
        # Fall back to standard ColumnParallelLinear forward.
        # ColumnParallelLinear.forward returns (output, bias) or just output
        # depending on configuration, so keep the return type as Tensor.
        return super().forward(input_, weight, runtime_gather_output)

    def _compute_linear_and_cross_entropy_loss(
        self,
        hidden: torch.Tensor,
        weight: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        reduction: Literal["none", "sum", "mean"] = "none",
        ignore_index: int = -100,
    ) -> torch.Tensor:
        """Compute fused linear + cross-entropy over tensor-parallel vocab."""
        assert self.config.cross_entropy_loss_fusion, "Cross-entropy loss fusion must be enabled."
        assert self.config.cross_entropy_fusion_impl == "linear", (
            "Cross-entropy loss fusion implementation must be 'linear' to use "
            "_compute_linear_and_cross_entropy_loss."
        )
        assert weight is not None, "weight cannot be None when using fused linear cross entropy."
        assert labels is not None, "labels cannot be None when using fused linear cross entropy."

        # [b s] => [s b]
        labels = labels.transpose(0, 1).contiguous()
        loss = linear_cross_entropy(
            hidden,
            weight,
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

        return loss
