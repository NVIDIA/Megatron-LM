# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

import warnings
from copy import deepcopy
from typing import Optional

import torch
import torch.nn.functional as F

from megatron.core.dist_checkpointing.mapping import ShardedStateDict
from megatron.core.fusions.fused_bias_geglu import bias_geglu_impl
from megatron.core.fusions.fused_bias_gelu import bias_gelu_impl
from megatron.core.fusions.fused_bias_swiglu import bias_swiglu_impl
from megatron.core.tensor_parallel.mappings import (
    copy_to_tensor_model_parallel_region,
    gather_from_sequence_parallel_region,
    reduce_from_tensor_model_parallel_region,
    reduce_scatter_to_sequence_parallel_region,
)
from megatron.core.tensor_parallel.random import (
    get_cuda_rng_tracker,
    get_data_parallel_rng_tracker_name,
)
from megatron.core.transformer.mlp import MLP
from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.utils import make_sharded_tensor_for_checkpoint


class SharedExpertMLP(MLP):
    """
    MLP layer for Shared Experts.
    """

    # This stream is used when '--moe-shared-expert-overlap' is set.
    # The shared experts are scheduled into this stream to be overlapped with the dispatcher.
    stream = None

    def __init__(self, config: TransformerConfig, spec: ModuleSpec):
        config = deepcopy(config)
        assert config.add_bias_linear == False, "bias is not supported in the shared experts, "
        "please set '--disable-bias-linear' instead."

        config.ffn_hidden_size = config.moe_shared_expert_intermediate_size
        super().__init__(config=config, submodules=spec.submodules)

        self.use_shared_expert_gate = spec.params.get("gate", False)
        if self.use_shared_expert_gate:
            self.gate_weight = torch.nn.Parameter(torch.empty((1, self.config.hidden_size)))
            if config.perform_initialization:
                if get_cuda_rng_tracker().is_initialized():
                    with get_cuda_rng_tracker().fork(get_data_parallel_rng_tracker_name()):
                        config.init_method(self.gate_weight)
            else:
                config.init_method(self.gate_weight)
            self.gate_weight.data = self.gate_weight.data.to(dtype=config.params_dtype)
            setattr(self.gate_weight, 'sequence_parallel', self.config.sequence_parallel)
        else:
            self.gate_weight = None

        if self.config.moe_shared_expert_overlap:
            # disable TP related AG/RS communications in the linear module
            for linear in [self.linear_fc1, self.linear_fc2]:
                if hasattr(linear, 'parallel_mode'):
                    # TELinear
                    linear.parallel_mode = None
                else:
                    # MCore legacy Linear
                    linear.explicit_expert_comm = True

            # The overlapped version is splitted into some separated functions and is put inside
            # the token dispatcher. These functions should be called in this order and no one can
            # be skipped:
            #     pre_forward_comm(input)
            #     linear_fc1_forward_and_act()
            #     linear_fc2_forward()
            #     post_forward_comm()
            #     output = get_output()
            #
            # We use cached intermediate results to avoid messy arg passing in the dispatcher.
            self.cached_fc1_input = None
            self.cached_fc2_input = None
            self.cached_fc2_output = None
            self.cached_output = None
            self.gate_score = None

            if self.stream is None:
                self.stream = torch.cuda.Stream()

    def forward(self, hidden_states):
        """Forward function"""
        output, _ = super().forward(hidden_states)
        if self.use_shared_expert_gate:
            logits = torch.nn.functional.linear(hidden_states, self.gate_weight)
            gate_score = torch.nn.functional.sigmoid(logits)
            output = output * gate_score
        return output

    def sharded_state_dict(
        self, prefix: str = '', sharded_offsets: tuple = (), metadata: Optional[dict] = None
    ) -> ShardedStateDict:
        """Gets sharded state dict."""
        sharded_state_dict = super().sharded_state_dict(prefix, sharded_offsets, metadata)
        if self.use_shared_expert_gate:
            name = 'gate_weight'
            state_dict = self.state_dict(prefix='', keep_vars=True)
            sub_sd = {
                f'{prefix}{name}': make_sharded_tensor_for_checkpoint(
                    state_dict[name], f'{prefix}{name}', prepend_offsets=sharded_offsets
                )
            }
            sharded_state_dict.update(sub_sd)
        return sharded_state_dict

    def pre_forward_comm(self, input):
        """
        All Gather for SP before forward.
        This function is used to overlap shared experts with the dispatcher.
        It is only useful when --moe-shared-expert-overlap is set and may be changed.
        """
        assert self.config.moe_shared_expert_overlap
        assert self.cached_output is None
        self.stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(self.stream):
            if self.use_shared_expert_gate:
                logits = torch.nn.functional.linear(input, self.gate_weight)
                self.gate_score = torch.nn.functional.sigmoid(logits)
            if self.config.sequence_parallel:
                self.cached_fc1_input = gather_from_sequence_parallel_region(
                    input, tensor_parallel_output_grad=True
                )
            else:
                self.cached_fc1_input = copy_to_tensor_model_parallel_region(input)
            set_tensor_grad_fn_sequence_sr(self.cached_fc1_input, torch.iinfo(torch.int).max)

    def linear_fc1_forward_and_act(self, overlapped_comm_output=None):
        """
        Do Linear FC1 and activation function forward.
        This function is used to overlap shared experts with the dispatcher.
        It is only useful when --moe-shared-expert-overlap is set and may be changed.
        """
        assert self.config.moe_shared_expert_overlap
        assert self.cached_fc1_input is not None
        if overlapped_comm_output is not None:
            set_tensor_grad_fn_sequence_sr(overlapped_comm_output, torch.iinfo(torch.int).max)
        with torch.cuda.stream(self.stream):
            # [s, b, 4 * h/p]
            intermediate_parallel, bias_parallel = self.linear_fc1(self.cached_fc1_input)
            self.cached_fc1_input = None

            if self.config.bias_activation_fusion:
                if self.activation_func == F.gelu:
                    if self.config.gated_linear_unit:
                        intermediate_parallel = bias_geglu_impl(
                            intermediate_parallel, bias_parallel
                        )
                    else:
                        assert self.config.add_bias_linear is True
                        intermediate_parallel = bias_gelu_impl(intermediate_parallel, bias_parallel)
                elif self.activation_func == F.silu and self.config.gated_linear_unit:
                    intermediate_parallel = bias_swiglu_impl(
                        intermediate_parallel,
                        bias_parallel,
                        self.config.activation_func_fp8_input_store,
                    )
                else:
                    raise ValueError("Only support fusion of gelu and swiglu")
            else:
                if bias_parallel is not None:
                    intermediate_parallel = intermediate_parallel + bias_parallel
                if self.config.gated_linear_unit:

                    def glu(x):
                        x = torch.chunk(x, 2, dim=-1)
                        return self.config.activation_func(x[0]) * x[1]

                    intermediate_parallel = glu(intermediate_parallel)
                else:
                    intermediate_parallel = self.activation_func(intermediate_parallel)

            self.cached_fc2_input = intermediate_parallel

    def linear_fc2_forward(self, overlapped_comm_output=None):
        """
        Do Linear FC2 forward.
        This function is used to overlap shared experts with the dispatcher.
        It is only useful when --moe-shared-expert-overlap is set and may be changed.
        """
        assert self.config.moe_shared_expert_overlap
        assert self.cached_fc2_input is not None
        if overlapped_comm_output is not None:
            set_tensor_grad_fn_sequence_sr(overlapped_comm_output, torch.iinfo(torch.int).max)
        with torch.cuda.stream(self.stream):
            # [s, b, h]
            self.cached_fc2_output, _ = self.linear_fc2(self.cached_fc2_input)
            self.cached_fc2_input = None

    def post_forward_comm(self):
        """
        Reduce scatter for SP after forward.
        This function is used to overlap shared experts with the dispatcher.
        It is only useful when --moe-shared-expert-overlap is set and may be changed.
        """
        assert self.config.moe_shared_expert_overlap
        assert self.cached_fc2_output is not None
        with torch.cuda.stream(self.stream):
            if self.config.sequence_parallel:
                self.cached_output = reduce_scatter_to_sequence_parallel_region(
                    self.cached_fc2_output
                )
            else:
                self.cached_output = reduce_from_tensor_model_parallel_region(
                    self.cached_fc2_output
                )
            self.cached_fc2_output = None
            set_tensor_grad_fn_sequence_sr(self.cached_output, torch.iinfo(torch.int).max)

    def get_output(self):
        """
        Gets the module forward output.
        This function is used to overlap shared experts with the dispatcher.
        It is only useful when --moe-shared-expert-overlap is set and may be changed.
        """
        assert self.config.moe_shared_expert_overlap
        assert self.cached_output is not None
        with torch.cuda.stream(self.stream):
            if self.use_shared_expert_gate:
                assert self.gate_score is not None
                output = self.cached_output * self.gate_score
                self.gate_score = None
            else:
                output = self.cached_output
            self.cached_output = None
        torch.cuda.current_stream().wait_stream(self.stream)
        return output


TORCH_MAJOR = int(torch.__version__.split(".")[0])
TORCH_MINOR = int(torch.__version__.split(".")[1])
TORCH_LAST = torch.__version__.split(".")[2]


def set_tensor_grad_fn_sequence_sr(tensor, value):
    """
    Set sequence_sr for the grad_fn of a tensor to control the backward order.
    For older PyTorch version, do nothing (backward order is not changed).
    The bigger the value is, the earlier the grad_fn is scheduled.
    """
    if (
        (TORCH_MAJOR > 2)
        or (TORCH_MAJOR == 2 and TORCH_MINOR > 2)
        or (TORCH_MAJOR == 2 and TORCH_MINOR == 2 and '+' not in TORCH_LAST)
    ):
        # In NVIDIA PyTorch container 24.01, the PyTorch version is 2.2.0a0+81ea7a4,
        # which does not contian the set_sequence_nr commit.
        if tensor is not None and tensor.grad_fn is not None:
            tensor.grad_fn._set_sequence_nr(value)
    else:
        warnings.warn(
            "WARNING : PyTorch is too old to set sequence_sr and the performance may not "
            "optimal. Please use PyTorch >= 2.2.0 for better performance."
        )
