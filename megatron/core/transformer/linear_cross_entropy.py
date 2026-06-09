# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

from typing import Literal, Optional, Tuple, Union

import torch

from megatron.core import tensor_parallel
from megatron.core.fusions.fused_linear_cross_entropy import linear_cross_entropy
from megatron.core.tensor_parallel.mappings import gather_from_tensor_model_parallel_region
from megatron.core.transformer.mxfp8_output_proj import (
    is_mxfp8_output_proj_active,
    mxfp8_column_parallel_linear,
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

        if is_mxfp8_output_proj_active(self.config):
            return self._forward_mxfp8(input_, weight, runtime_gather_output)

        # Fall back to standard ColumnParallelLinear forward.
        # ColumnParallelLinear.forward returns (output, bias) or just output
        # depending on configuration, so keep the return type as Tensor.
        return super().forward(input_, weight, runtime_gather_output)

    def _forward_mxfp8(
        self,
        input_: torch.Tensor,
        weight: Optional[torch.Tensor],
        runtime_gather_output: Optional[bool],
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """MXFP8 replacement for ColumnParallelLinear.forward.

        Mirrors the pre/post steps of ColumnParallelLinear.forward (shape
        check, optional bias, gather_output) and delegates the GEMM to the
        custom MXFP8 autograd function. Uses only the subset of behaviors
        relevant to the LM head (no explicit_expert_comm, no defer_embedding_wgrad,
        no CPU offloading of activations on this layer).
        """
        if weight is None:
            if self.weight is None:
                raise RuntimeError(
                    "weight was not supplied to LinearCrossEntropyModule forward "
                    "and skip_weight_param_allocation is True."
                )
            weight = self.weight
        else:
            expected_shape = (self.output_size_per_partition, self.input_size)
            if weight.shape != expected_shape:
                raise RuntimeError(
                    f"supplied weight's shape is {tuple(weight.shape)}, "
                    f"not {expected_shape} as expected"
                )

        bias = self.bias if not self.skip_bias_add else None

        allreduce_dgrad = False if self.explicit_expert_comm else self.allreduce_dgrad
        sequence_parallel = False if self.explicit_expert_comm else self.sequence_parallel

        fp8_dgrad = bool(getattr(self.config, "fp8_output_proj_dgrad", True))
        fp8_wgrad = bool(getattr(self.config, "fp8_output_proj_wgrad", True))

        output_parallel = mxfp8_column_parallel_linear(
            module=self,
            input_=input_,
            weight=weight,
            bias=bias,
            gradient_accumulation_fusion=self.gradient_accumulation_fusion,
            allreduce_dgrad=allreduce_dgrad,
            sequence_parallel=sequence_parallel,
            tp_group=self.tp_group,
            fp8_dgrad=fp8_dgrad,
            fp8_wgrad=fp8_wgrad,
        )

        gather_output = self.gather_output
        if runtime_gather_output is not None:
            gather_output = runtime_gather_output
        if gather_output:
            output = gather_from_tensor_model_parallel_region(
                output_parallel, group=self.tp_group
            )
        else:
            output = output_parallel

        output_bias = self.bias if self.skip_bias_add else None
        return output, output_bias

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
