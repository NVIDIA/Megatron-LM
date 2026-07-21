# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import warnings
from copy import deepcopy
from enum import Enum
from functools import wraps
from typing import Optional

import torch
import torch.nn.functional as F

from megatron.core.dist_checkpointing.mapping import ShardedStateDict
from megatron.core.extensions.transformer_engine import HAVE_TE
from megatron.core.fusions.fused_bias_geglu import bias_geglu_impl
from megatron.core.fusions.fused_bias_gelu import bias_gelu_impl
from megatron.core.fusions.fused_bias_swiglu import bias_swiglu_impl
from megatron.core.tensor_parallel.mappings import (
    copy_to_tensor_model_parallel_region,
    gather_from_sequence_parallel_region,
    reduce_from_tensor_model_parallel_region,
    reduce_scatter_to_sequence_parallel_region,
)
from megatron.core.transformer.mlp import MLP, MLPSubmodules
from megatron.core.transformer.moe.moe_utils import ProcessGroupCollection
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.typed_torch import apply_module
from megatron.core.utils import (
    get_pg_size,
    is_te_min_version,
    is_torch_min_version,
    make_sharded_tensor_for_checkpoint,
)

if HAVE_TE:
    import transformer_engine as te

    from megatron.core.extensions.transformer_engine import TELinear, set_save_original_input
    from megatron.core.tensor_parallel.random import get_cuda_rng_tracker
else:
    te = None
    TELinear, set_save_original_input = None, None
    get_cuda_rng_tracker = None


class SharedExpertState(Enum):
    """State machine states for SharedExpertMLP overlapped forward pass."""

    IDLE = 0
    PRE_FORWARD_COMM_DONE = 1
    FC1_FORWARD_DONE = 2
    FC2_FORWARD_DONE = 3
    POST_FORWARD_COMM_DONE = 4


def overlap_state_check(required_state: "SharedExpertState", next_state: "SharedExpertState"):
    """
    Decorator to validate overlap state and cached variables before method execution,
    and update state after method execution.

    Args:
        required_state: The expected SharedExpertState before this method runs.
        next_state: The SharedExpertState to transition to after method execution.
    """

    def decorator(method):
        @wraps(method)
        def wrapper(self, *args, **kwargs):
            # Check overlap is enabled
            assert (
                self.config.moe_shared_expert_overlap
            ), f"{method.__name__} requires --moe-shared-expert-overlap to be set"
            # Check state machine
            assert self._overlap_state == required_state, (
                f"{method.__name__} must be called from {required_state.name} state, "
                f"but current state is {self._overlap_state.name}"
            )
            # Execute method
            result = method(self, *args, **kwargs)
            # Update state after method execution
            self._overlap_state = next_state
            return result

        return wrapper

    return decorator


class _BackwardStreamWait(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, stream):
        """forward"""
        ctx.stream = stream
        return input

    @staticmethod
    def backward(ctx, grad_output):
        """backward with stream wait"""
        ctx.stream.wait_stream(torch.cuda.current_stream())
        return grad_output, None


class SharedExpertMLP(MLP):
    """
    MLP layer for Shared Experts.
    """

    # This stream is used when '--moe-shared-expert-overlap' is set.
    # The shared experts are scheduled into this stream to be overlapped with the dispatcher.
    stream = None

    def __init__(
        self,
        config: TransformerConfig,
        submodules: MLPSubmodules,
        gate: bool,
        pg_collection: Optional[ProcessGroupCollection] = None,
        name: str | None = None,
    ):
        """
        Args:
            name (str | None): module instance name passed top-down from its paranet module
        """
        config = deepcopy(config)
        assert config.add_bias_linear == False, "bias is not supported in the shared experts, "
        "please set '--disable-bias-linear' instead."

        config.ffn_hidden_size = config.moe_shared_expert_intermediate_size
        # TODO(Hepteract): pass pg_collection to MLP after refactoring MLP
        super().__init__(config=config, submodules=submodules, tp_group=pg_collection.tp, name=name)

        self.use_shared_expert_gate = gate
        if self.use_shared_expert_gate:
            # TODO: Add support for GPU initialization, which requires updating the golden values.
            self.gate_weight = torch.nn.Parameter(torch.empty((1, self.config.hidden_size)))
            if config.perform_initialization:
                config.init_method(self.gate_weight)
            self.gate_weight.data = self.gate_weight.data.to(dtype=config.params_dtype)
            setattr(self.gate_weight, 'sequence_parallel', self.config.sequence_parallel)
        else:
            self.gate_weight = None

        if (
            self.config.fp8
            and self.config.fp8_recipe != 'delayed'
            and is_te_min_version("2.6.0dev0")
        ) or (self.config.fp4 and is_te_min_version("2.7.0.dev0")):
            # For fp8/fp4 training, the output of pre_mlp_layernorm is saved by router, and
            # the shared expert linear_fc1 also saves the quantized tensor of this output.
            # Here we set the linear_fc1 to save the original input tensors to avoid the extra
            # memory usage of the quantized tensor.
            shared_experts_recompute = (
                config.recompute_granularity == 'selective'
                and "shared_experts" in config.recompute_modules
            )
            if not shared_experts_recompute and HAVE_TE and isinstance(self.linear_fc1, TELinear):
                set_save_original_input(self.linear_fc1)

        if self.config.moe_shared_expert_overlap:
            # disable TP related AG/RS communications in the linear module
            for linear in [self.linear_fc1, self.linear_fc2]:
                if hasattr(linear, 'parallel_mode'):
                    # TELinear
                    linear.parallel_mode = None
                    linear.ub_overlap_rs_fprop = False
                    linear.ub_overlap_ag_dgrad = False
                    linear.ub_overlap_ag_fprop = False
                    linear.ub_overlap_rs_dgrad = False
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

            # State machine to ensure correct calling order of overlapped forward methods
            self._overlap_state = SharedExpertState.IDLE

            if SharedExpertMLP.stream is None:
                SharedExpertMLP.stream = torch.cuda.Stream()
            self.stream = SharedExpertMLP.stream

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
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
                    state_dict[name],
                    f'{prefix}{name}',
                    prepend_offsets=sharded_offsets,
                    tp_group=self.tp_group,
                    dp_cp_group=metadata['dp_cp_group'],
                )
            }
            sharded_state_dict.update(sub_sd)
        return sharded_state_dict

    def wait_current_stream(self):
        """Wait for the current stream to complete."""
        self.stream.wait_stream(torch.cuda.current_stream())

    @overlap_state_check(SharedExpertState.IDLE, SharedExpertState.PRE_FORWARD_COMM_DONE)
    def pre_forward_comm(self, input, wait_current_stream=True):
        """
        All Gather for SP before forward.
        This function is used to overlap shared experts with the dispatcher.
        It is only useful when --moe-shared-expert-overlap is set and may be changed.
        """
        if wait_current_stream:
            self.wait_current_stream()
        with torch.cuda.stream(self.stream):
            if self.use_shared_expert_gate:
                logits = torch.nn.functional.linear(input, self.gate_weight)
                self.gate_score = torch.nn.functional.sigmoid(logits)
            if self.config.sequence_parallel:
                self.cached_fc1_input = gather_from_sequence_parallel_region(
                    input, tensor_parallel_output_grad=True, group=self.tp_group
                )
            else:
                self.cached_fc1_input = copy_to_tensor_model_parallel_region(
                    input, group=self.tp_group
                )
            set_tensor_grad_fn_sequence_sr(self.cached_fc1_input, torch.iinfo(torch.int).max)

    @overlap_state_check(
        SharedExpertState.PRE_FORWARD_COMM_DONE, SharedExpertState.FC1_FORWARD_DONE
    )
    def linear_fc1_forward_and_act(self, overlapped_comm_output=None):
        """
        Do Linear FC1 and activation function forward.
        This function is used to overlap shared experts with the dispatcher.
        It is only useful when --moe-shared-expert-overlap is set and may be changed.
        """
        with torch.cuda.stream(self.stream):
            # [s, b, 4 * h/p]
            intermediate_parallel, bias_parallel = apply_module(self.linear_fc1)(
                self.cached_fc1_input
            )
            self.cached_fc1_input = None

            if self.config.use_te_activation_func:
                if bias_parallel is not None:
                    intermediate_parallel = intermediate_parallel + bias_parallel
                intermediate_parallel = self.activation_func(intermediate_parallel)
            elif self.config.bias_activation_fusion:
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
                        clamp_value=self.config.activation_func_clamp_value,
                    )
                else:
                    raise ValueError("Only support fusion of gelu and swiglu")
            else:
                if bias_parallel is not None:
                    intermediate_parallel = intermediate_parallel + bias_parallel
                if self.config.gated_linear_unit:

                    def glu(x):
                        x_glu, x_linear = torch.chunk(x, 2, dim=-1)
                        if (clamp_value := self.config.activation_func_clamp_value) is not None:
                            x_glu = x_glu.clamp(min=None, max=clamp_value)
                            x_linear = x_linear.clamp(min=-clamp_value, max=clamp_value)
                        return self.config.activation_func(x_glu) * (
                            x_linear + self.config.glu_linear_offset
                        )

                    intermediate_parallel = glu(intermediate_parallel)
                else:
                    intermediate_parallel = self.activation_func(intermediate_parallel)

            self.cached_fc2_input = intermediate_parallel
        # Tensor sequence number is used to control the backward order.
        # Decrease the sequence number of the expert output to make the comm launched first
        # in the backward order.
        if overlapped_comm_output is not None and overlapped_comm_output.grad_fn is not None:
            target_sequence_nr = overlapped_comm_output.grad_fn._sequence_nr() - 1
            set_tensor_grad_fn_sequence_sr(intermediate_parallel, target_sequence_nr)
            # Make sure the shared expert fc1 backward is launched after the routed fc1 backward
            self.cached_fc2_input = _BackwardStreamWait.apply(intermediate_parallel, self.stream)

    @overlap_state_check(SharedExpertState.FC1_FORWARD_DONE, SharedExpertState.FC2_FORWARD_DONE)
    def linear_fc2_forward(self, overlapped_comm_output=None):
        """
        Do Linear FC2 forward.
        This function is used to overlap shared experts with the dispatcher.
        It is only useful when --moe-shared-expert-overlap is set and may be changed.
        """
        if overlapped_comm_output is not None:
            set_tensor_grad_fn_sequence_sr(overlapped_comm_output, torch.iinfo(torch.int).max)
        assert self.cached_fc2_input is not None
        with torch.cuda.stream(self.stream):
            # [s, b, h]
            self.cached_fc2_output, _ = apply_module(self.linear_fc2)(self.cached_fc2_input)
            self.cached_fc2_input = None

    @overlap_state_check(
        SharedExpertState.FC2_FORWARD_DONE, SharedExpertState.POST_FORWARD_COMM_DONE
    )
    def post_forward_comm(self):
        """
        Reduce scatter for SP after forward.
        This function is used to overlap shared experts with the dispatcher.
        It is only useful when --moe-shared-expert-overlap is set and may be changed.
        """
        with torch.cuda.stream(self.stream):
            if self.config.sequence_parallel:
                self.cached_output = reduce_scatter_to_sequence_parallel_region(
                    self.cached_fc2_output, group=self.tp_group
                )
            else:
                self.cached_output = reduce_from_tensor_model_parallel_region(
                    self.cached_fc2_output, group=self.tp_group
                )
            self.cached_fc2_output = None
            set_tensor_grad_fn_sequence_sr(self.cached_output, torch.iinfo(torch.int).max)

    @overlap_state_check(SharedExpertState.POST_FORWARD_COMM_DONE, SharedExpertState.IDLE)
    def get_output(self):
        """
        Gets the module forward output.
        This function is used to overlap shared experts with the dispatcher.
        It is only useful when --moe-shared-expert-overlap is set and may be changed.
        """
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

    def backward_dw(self):
        """Compute delayed weight gradients for shared experts."""
        super().backward_dw()


class FusedSharedExpertMLP(SharedExpertMLP):
    """Shared expert MLP implemented with TE GroupedLinear(num_groups=1) fused ops."""

    def __init__(
        self,
        config: TransformerConfig,
        submodules: MLPSubmodules,
        gate: bool,
        pg_collection: Optional[ProcessGroupCollection] = None,
        name: str | None = None,
    ):
        super().__init__(
            config=config, submodules=submodules, gate=gate, pg_collection=pg_collection, name=name
        )
        self._fused_grouped_swiglu_ops = None
        self._fused_grouped_swiglu_recipe = None
        self._validate_fused_grouped_swiglu()

    def _validate_fused_grouped_swiglu(self) -> None:
        """Validate the requested GroupedLinear(num_groups=1) SwiGLU path."""
        if not HAVE_TE:
            raise RuntimeError(
                f"{self.__class__.__name__} requires Transformer Engine when "
                "use_grouped_gemm_for_shared_expert=True."
            )
        if not is_te_min_version("2.14.0"):
            raise RuntimeError(
                f"{self.__class__.__name__} requires Transformer Engine >= 2.14.0 "
                "(needs pytorch.ops.GroupedLinear and pytorch.ops.ScaledSwiGLU)."
            )
        if self.config.add_bias_linear:
            raise ValueError(
                f"{self.__class__.__name__} does not support add_bias_linear=True; "
                "the CuTeGEMM fused kernel requires bias-free linear layers."
            )
        if not self.config.gated_linear_unit or self.config.activation_func != F.silu:
            raise ValueError(
                f"{self.__class__.__name__} requires SwiGLU activation "
                "(activation_func=F.silu, gated_linear_unit=True) for the CuTeGEMM "
                f"fused kernel, but got activation_func={self.config.activation_func}, "
                f"gated_linear_unit={self.config.gated_linear_unit}."
            )
        if self.config.activation_func_clamp_value is not None and (
            not is_te_min_version("2.17.0.dev0")
            or not hasattr(te.pytorch.ops, "ScaledClampedQGeGLU")
        ):
            raise RuntimeError(
                f"{self.__class__.__name__} requires Transformer Engine >= 2.17.0.dev0 "
                "with pytorch.ops.ScaledClampedQGeGLU when "
                "activation_func_clamp_value is set."
            )
        if self.config.moe_shared_expert_glu_interleave_size is None:
            raise ValueError(
                f"{self.__class__.__name__} requires "
                "moe_shared_expert_glu_interleave_size to be set when "
                "use_grouped_gemm_for_shared_expert=True."
            )
        if not isinstance(self.linear_fc1, te.pytorch.Linear):
            raise ValueError(
                f"{self.__class__.__name__} expects FC1 to be Transformer Engine Linear, "
                f"but found {self.linear_fc1.__class__.__name__}."
            )
        if not isinstance(self.linear_fc2, te.pytorch.Linear):
            raise ValueError(
                f"{self.__class__.__name__} expects FC2 to be Transformer Engine Linear, "
                f"but found {self.linear_fc2.__class__.__name__}."
            )

    def _get_fused_grouped_swiglu_recipe(self):
        """Create the TE recipe used to select the fused grouped MLP kernel."""
        if self._fused_grouped_swiglu_recipe is None:
            fp4_recipe = getattr(self.config.fp4_recipe, "value", self.config.fp4_recipe)
            fp8_recipe = getattr(self.config.fp8_recipe, "value", self.config.fp8_recipe)
            if self.config.fp4 and fp4_recipe == "nvfp4":
                self._fused_grouped_swiglu_recipe = te.common.recipe.NVFP4BlockScaling()
            elif self.config.fp8 and fp8_recipe == "mxfp8":
                self._fused_grouped_swiglu_recipe = te.common.recipe.MXFP8BlockScaling()
            else:
                raise ValueError(
                    f"{self.__class__.__name__} requires fp4_recipe='nvfp4' or "
                    f"fp8_recipe='mxfp8', but got fp4={self.config.fp4}, "
                    f"fp4_recipe={self.config.fp4_recipe}, fp8={self.config.fp8}, "
                    f"fp8_recipe={self.config.fp8_recipe}."
                )
        return self._fused_grouped_swiglu_recipe

    def _make_fused_grouped_swiglu_ops(self) -> torch.nn.Module:
        """Construct the grouped-linear shared-expert MLP operations."""
        ops = te.pytorch.ops.Sequential()
        tp_world_size = get_pg_size(self.tp_group)
        rng_state_tracker_function = None
        if get_cuda_rng_tracker().is_initialized():
            rng_state_tracker_function = get_cuda_rng_tracker

        glu_interleave_size = self.config.moe_shared_expert_glu_interleave_size
        fc1_weight = self.linear_fc1.weight
        op = te.pytorch.ops.GroupedLinear(
            num_groups=1,
            in_features=fc1_weight.size(1),
            out_features=fc1_weight.size(0) * tp_world_size,
            device="meta",
            dtype=fc1_weight.dtype,
            bias=False,
            rng_state_tracker_function=rng_state_tracker_function,
            accumulate_into_main_grad=self.linear_fc1.fuse_wgrad_accumulation,
        )
        op.weight0 = fc1_weight
        op._glu_interleave_size = glu_interleave_size
        ops.append(op)

        clamp_value = self.config.activation_func_clamp_value
        if clamp_value is None:
            activation_op = te.pytorch.ops.ScaledSwiGLU(glu_interleave_size=glu_interleave_size)
        else:
            activation_op = te.pytorch.ops.ScaledClampedQGeGLU(
                glu_interleave_size=glu_interleave_size,
                alpha=1.0,
                limit=clamp_value,
                glu_linear_offset=0.0,
            )
        ops.append(activation_op)

        fc2_weight = self.linear_fc2.weight
        op = te.pytorch.ops.GroupedLinear(
            num_groups=1,
            in_features=fc2_weight.size(1),
            out_features=fc2_weight.size(0),
            device="meta",
            dtype=fc2_weight.dtype,
            bias=False,
            rng_state_tracker_function=rng_state_tracker_function,
            accumulate_into_main_grad=self.linear_fc2.fuse_wgrad_accumulation,
        )
        op.weight0 = fc2_weight
        ops.append(op)

        def forward_pre_hook(_module, *_) -> None:
            for source in (self.linear_fc1, self.linear_fc2):
                for hook_id, hook in list(source._forward_pre_hooks.items()):
                    if hook_id in source._forward_pre_hooks_with_kwargs:
                        ret = hook(source, (), {})
                    else:
                        ret = hook(source, ())
                    if ret is not None:
                        raise RuntimeError(
                            f"{self.__class__.__name__} cannot replay a pre-forward hook "
                            f"on {source.__class__.__name__} that modifies inputs."
                        )

        ops.register_forward_pre_hook(forward_pre_hook)
        return ops

    def _fused_grouped_swiglu_no_comm(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Run the fused shared expert MLP on tensor-parallel prepared input."""
        orig_shape = hidden_states.shape
        hidden_size = hidden_states.size(-1)
        hidden_states_2d = hidden_states.view(-1, hidden_size)
        total_tokens = hidden_states_2d.size(0)
        tokens_per_expert = torch.full(
            (1,), total_tokens, dtype=torch.long, device=hidden_states.device
        )
        scales = torch.ones(total_tokens, device=hidden_states.device, dtype=hidden_states.dtype)

        recipe = self._get_fused_grouped_swiglu_recipe()
        if self._fused_grouped_swiglu_ops is None:
            with te.pytorch.fp8_autocast(enabled=True, fp8_recipe=recipe):
                self._fused_grouped_swiglu_ops = (self._make_fused_grouped_swiglu_ops(),)

        with te.pytorch.fp8_autocast(enabled=True, fp8_recipe=recipe):
            output = self._fused_grouped_swiglu_ops[0](
                hidden_states_2d, tokens_per_expert, scales, tokens_per_expert
            )
        return output.view(*orig_shape[:-1], output.size(-1))

    def _fused_grouped_swiglu_forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Run the fused shared expert MLP with the same TP comms as the dense MLP path."""
        if self.config.sequence_parallel:
            fc1_input = gather_from_sequence_parallel_region(
                hidden_states, tensor_parallel_output_grad=True
            )
        else:
            fc1_input = copy_to_tensor_model_parallel_region(hidden_states)

        fc2_output = self._fused_grouped_swiglu_no_comm(fc1_input)

        if self.config.sequence_parallel:
            output = reduce_scatter_to_sequence_parallel_region(fc2_output)
        else:
            output = reduce_from_tensor_model_parallel_region(fc2_output)
        return output

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Forward function."""
        output = self._fused_grouped_swiglu_forward(hidden_states)
        if self.use_shared_expert_gate:
            logits = torch.nn.functional.linear(hidden_states, self.gate_weight)
            gate_score = torch.nn.functional.sigmoid(logits)
            output = output * gate_score
        return output

    @overlap_state_check(
        SharedExpertState.PRE_FORWARD_COMM_DONE, SharedExpertState.FC1_FORWARD_DONE
    )
    def linear_fc1_forward_and_act(self, overlapped_comm_output=None):
        """Run fused FC1, activation, and FC2 for overlapped shared experts."""
        del overlapped_comm_output
        with torch.cuda.stream(self.stream):
            self.cached_fc2_output = self._fused_grouped_swiglu_no_comm(self.cached_fc1_input)
            self.cached_fc1_input = None

    @overlap_state_check(SharedExpertState.FC1_FORWARD_DONE, SharedExpertState.FC2_FORWARD_DONE)
    def linear_fc2_forward(self, overlapped_comm_output=None):
        """Skip FC2 because the fused path computes FC2 during linear_fc1_forward_and_act."""
        if overlapped_comm_output is not None:
            set_tensor_grad_fn_sequence_sr(overlapped_comm_output, torch.iinfo(torch.int).max)
        assert self.cached_fc2_output is not None

    def backward_dw(self):
        """Compute delayed weight gradients for fused shared experts."""
        if self.config.delay_wgrad_compute:
            if self._fused_grouped_swiglu_ops is not None:
                (seq,) = self._fused_grouped_swiglu_ops
                fused_children = list(seq.children())
                assert len(fused_children) >= 3, "expected FC1, activation, FC2 in fused TE ops"
                fused_children[2].backward_dw()
                fused_children[0].backward_dw()
                if hasattr(self.linear_fc2, "_trigger_wgrad_accumulation_and_reduce_hooks"):
                    self.linear_fc2._trigger_wgrad_accumulation_and_reduce_hooks()
                if hasattr(self.linear_fc1, "_trigger_wgrad_accumulation_and_reduce_hooks"):
                    self.linear_fc1._trigger_wgrad_accumulation_and_reduce_hooks()
            return
        super().backward_dw()


def set_tensor_grad_fn_sequence_sr(tensor, value):
    """
    Set sequence_sr for the grad_fn of a tensor to control the backward order.
    For older PyTorch version, do nothing (backward order is not changed).
    The bigger the value is, the earlier the grad_fn is scheduled.
    """
    if is_torch_min_version("2.2.0"):
        if tensor is not None and tensor.grad_fn is not None:
            tensor.grad_fn._set_sequence_nr(value)
    else:
        warnings.warn(
            "WARNING : PyTorch is too old to set sequence_sr and the performance may not "
            "be optimal. Please use PyTorch >= 2.2.0 for better performance."
        )
