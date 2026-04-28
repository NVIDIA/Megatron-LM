# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

import copy
from typing import Literal, Optional, Tuple, Union

import torch

from megatron.core import tensor_parallel
from megatron.core.extensions.transformer_engine import HAVE_TE, TEColumnParallelLinear
from megatron.core.fp8_utils import get_fp8_context
from megatron.core.fusions.fused_linear_cross_entropy import linear_cross_entropy
from megatron.core.tensor_parallel.mappings import gather_from_tensor_model_parallel_region
from megatron.core.tensor_parallel.utils import divide


def is_te_mxfp8_output_proj_active(config) -> bool:
    """Return whether the LM-head output projection should use TE MXFP8."""
    if not HAVE_TE:
        return False
    if not getattr(config, "fp8_output_proj", False):
        return False
    if not getattr(config, "fp8", False):
        return False

    fp8_recipe = getattr(config, "fp8_recipe", None)
    recipe_value = getattr(fp8_recipe, "value", fp8_recipe)
    return str(recipe_value).lower() == "mxfp8" or str(fp8_recipe).lower().endswith(".mxfp8")


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


class TELinearCrossEntropyModule(TEColumnParallelLinear):
    """TE ColumnParallelLinear variant for the LM head plus optional CE fusion.

    This module is used only when ``fp8_output_proj`` is enabled with the MXFP8
    recipe. It deliberately does not support the Megatron output-layer-specific
    deferred embedding wgrad buffers.
    """

    def __init__(
        self,
        input_size,
        output_size,
        *,
        config,
        init_method,
        bias=True,
        gather_output=False,
        stride=1,
        keep_master_weight_for_test=False,
        skip_bias_add=False,
        skip_weight_param_allocation: bool = False,
        embedding_activation_buffer=None,
        grad_output_buffer=None,
        is_expert: bool = False,
        tp_comm_buffer_name: Optional[str] = None,
        disable_grad_reduce: bool = False,
        tp_group: Optional[torch.distributed.ProcessGroup] = None,
    ):
        if not is_te_mxfp8_output_proj_active(config):
            raise RuntimeError(
                "TELinearCrossEntropyModule is only valid when fp8_output_proj=True, "
                "fp8=True, and fp8_recipe='mxfp8'."
            )
        if (
            getattr(config, "cross_entropy_loss_fusion", False)
            and getattr(config, "cross_entropy_fusion_impl", None) == "linear"
        ):
            raise ValueError(
                "fp8_output_proj is incompatible with cross_entropy_loss_fusion + "
                "cross_entropy_fusion_impl='linear': the fused linear+CE kernel runs "
                "the LM-head GEMM in bf16, silently bypassing MXFP8."
            )
        if keep_master_weight_for_test:
            raise ValueError("TE output projection does not support keep_master_weight_for_test.")
        if skip_weight_param_allocation:
            raise ValueError("TE output projection does not support skip_weight_param_allocation.")
        if embedding_activation_buffer is not None or grad_output_buffer is not None:
            raise ValueError(
                "TE MXFP8 output projection does not support defer_embedding_wgrad_compute."
            )
        if disable_grad_reduce:
            raise ValueError("TE output projection does not support disable_grad_reduce.")

        te_config = copy.copy(config)
        # The output layer is outside the transformer-layer schedule that calls
        # TELinear.backward_dw(), so it must not use TE delayed wgrad.
        te_config.delay_wgrad_compute = False

        super().__init__(
            input_size=input_size,
            output_size=output_size,
            config=te_config,
            init_method=init_method,
            gather_output=False,
            bias=bias,
            skip_bias_add=skip_bias_add,
            is_expert=is_expert,
            skip_weight_param_allocation=skip_weight_param_allocation,
            tp_comm_buffer_name=tp_comm_buffer_name,
            tp_group=tp_group,
            stride=stride,
        )

        self.input_size = input_size
        self.output_size = output_size
        self.output_size_per_partition = divide(output_size, self.tp_size)
        self.gather_output = gather_output
        self.skip_bias_add = skip_bias_add
        self.embedding_activation_buffer = None
        self.grad_output_buffer = None
        self.disable_grad_reduce = False
        self.tp_group = self._tp_group

        self._register_load_state_dict_pre_hook(
            lambda state_dict, prefix, *args, **kwargs: state_dict.setdefault(
                f"{prefix}_extra_state"
            )
        )

    def get_extra_state(self):
        # MXFP8 has no persistent recipe state; keep LM head _extra_state empty
        # so GPTModel.sharded_state_dict's no-extra-state invariant still holds.
        return None

    def set_extra_state(self, state):
        return

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
        """Run TE MXFP8 output projection or fused linear+cross-entropy."""
        if output_cross_entropy_loss:
            assert labels is not None, "labels cannot be None when outputting cross-entropy loss."
            return LinearCrossEntropyModule._compute_linear_and_cross_entropy_loss(
                self,
                hidden=input_,
                weight=weight if weight is not None else self.weight,
                labels=labels,
                reduction=reduction,
                ignore_index=ignore_index,
            )

        if weight is not None and weight is not self.weight:
            raise RuntimeError("TE MXFP8 output projection does not support runtime weight.")

        with get_fp8_context(self.config):
            torch.cuda.nvtx.range_push("mxfp8_output_proj_telinear")
            try:
                output_parallel, output_bias = super().forward(input_)
            finally:
                torch.cuda.nvtx.range_pop()

        gather_output = self.gather_output
        if runtime_gather_output is not None:
            gather_output = runtime_gather_output
        if gather_output:
            output = gather_from_tensor_model_parallel_region(output_parallel, group=self.tp_group)
        else:
            output = output_parallel
        return output, output_bias
