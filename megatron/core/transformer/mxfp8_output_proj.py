# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""MXFP8 LM-head output projection backed by Transformer Engine."""

import copy
from typing import Optional

import torch

from megatron.core.extensions.transformer_engine import HAVE_TE, TEColumnParallelLinear
from megatron.core.fp8_utils import get_fp8_context
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


class TELinearCrossEntropyModule(TEColumnParallelLinear):
    """TE ColumnParallelLinear variant for the LM head under MXFP8.

    Active only when ``fp8_output_proj=True`` with ``fp8_recipe='mxfp8'``. Acts
    as a drop-in replacement for the plain ColumnParallelLinear used as the GPT
    output layer.
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
    ):
        """Run TE MXFP8 output projection. Returns ``(output, bias)``."""
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
