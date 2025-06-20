# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

import dataclasses
import io
import os
import pickle
import warnings
from typing import Any, Callable, List, Optional, Tuple, Type

import torch
import torch.nn.functional as F
import transformer_engine as te
from packaging.version import Version as PkgVersion
from torch import Tensor
from torch.nn.parameter import Parameter

from megatron.core.dist_checkpointing.utils import replace_prefix_for_sharding
from megatron.core.model_parallel_config import ModelParallelConfig
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.parallel_state import (
    get_context_parallel_group,
    get_expert_data_parallel_rank,
    get_expert_model_parallel_rank,
    get_expert_model_parallel_world_size,
    get_hierarchical_context_parallel_groups,
    get_tensor_model_parallel_group,
)
from megatron.core.process_groups_config import ModelCommProcessGroups
from megatron.core.tensor_parallel.layers import (
    _initialize_affine_weight_cpu,
    set_tensor_model_parallel_attributes,
)
from megatron.core.tensor_parallel.random import (
    get_cuda_rng_tracker,
    get_data_parallel_rng_tracker_name,
    get_expert_parallel_rng_tracker_name,
)
from megatron.core.tensor_parallel.utils import divide
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.transformer.utils import make_sharded_tensors_for_checkpoint
from megatron.core.utils import (
    get_te_version,
    get_tensor_model_parallel_group_if_none,
    is_te_min_version,
    is_torch_min_version,
)


def _get_extra_te_kwargs(config: TransformerConfig):
    extra_transformer_engine_kwargs = {"params_dtype": config.params_dtype}

    if is_te_min_version("0.12.0"):
        if config.use_cpu_initialization:
            extra_transformer_engine_kwargs["device"] = 'cpu'
        elif config.init_model_with_meta_device:
            extra_transformer_engine_kwargs["device"] = "meta"
        else:
            extra_transformer_engine_kwargs["device"] = torch.cuda.current_device()
    return extra_transformer_engine_kwargs


def condition_init_method(config, init_method):
    """Condition TE init_method on config.perform_initialization."""
    return init_method if config.perform_initialization else (lambda w: None)


class TENorm:
    """
    A conditional wrapper to initialize an instance of Transformer-Engine's
    `LayerNorm` or `RMSNorm` based on input
    """

    # TODO should we ditch normalization config and just use spec to choose LayerNorm vs RMSNorm?
    def __new__(cls, config: TransformerConfig, hidden_size: int, eps: float = 1e-5):
        if config.normalization == "LayerNorm":
            instance = te.pytorch.LayerNorm(
                hidden_size=hidden_size,
                eps=eps,
                sequence_parallel=config.sequence_parallel,
                zero_centered_gamma=config.layernorm_zero_centered_gamma,
                **_get_extra_te_kwargs(config),
            )
        elif config.normalization == "RMSNorm":
            assert hasattr(
                te.pytorch, "RMSNorm"
            ), "Transformer-Engine >= v0.11 required to use this feature"
            instance = te.pytorch.RMSNorm(
                hidden_size=hidden_size,
                eps=eps,
                sequence_parallel=config.sequence_parallel,
                zero_centered_gamma=config.layernorm_zero_centered_gamma,
                **_get_extra_te_kwargs(config),
            )
        else:
            raise Exception('Only LayerNorm and RMSNorm are curently supported')

        return instance


class TELinear(te.pytorch.Linear):
    """
    Wrapper for the Transformer-Engine's `Linear` layer.

    Note that if Megatron's parallel_state has not been initialized
    yet, the tp_group passed to TE will be None and must be set later
    via set_tensor_parallel_group().

    parallel_mode currently supports 3 different values:
        - "column": Split the weight matrix along output dimension (used in TEColumnParallelLinear)
        - "row": Split the weight matrix along input dimension (used in TERowParallelLinear)
        - "duplicated": No tensor parallelism and weight is duplicated across TP ranks
        - Note: For expert linear layers, we will disable communication logic here
                as TP communication is handled in token_dispatcher.
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        *,
        parallel_mode: Optional[str],
        config: ModelParallelConfig,
        init_method: Callable,
        bias: bool,
        skip_bias_add: bool,
        skip_weight_param_allocation: bool,
        tp_comm_buffer_name: Optional[str] = None,
        is_expert: bool = False,
        symmetric_ar_type: Optional[str] = None,
        tp_group: Optional[torch.distributed.ProcessGroup] = None,
    ):
        self.config = config

        # TE returns a zero length Tensor when bias=False and
        # return_bias=True, but we prefer None.  So in that case we
        # tell TE to not return the bias, and return None
        # ourselves. This way our forward always returns two values
        # and we don't have to deal with the zero length Tensor.
        self.te_return_bias = skip_bias_add and bias
        self.is_first_microbatch = True
        self.disable_parameter_transpose_cache = self.config.disable_parameter_transpose_cache
        self.symmetric_ar_type = symmetric_ar_type
        if skip_weight_param_allocation:
            raise ValueError(
                'Transformer Engine linear layers do not support skip_weight_param_allocation'
            )

        extra_kwargs = _get_extra_te_kwargs(config)

        if self.config.delay_wgrad_compute:
            if is_te_min_version("2.3.0"):
                extra_kwargs["delay_wgrad_compute"] = self.config.delay_wgrad_compute
            else:
                raise RuntimeError("Only TE with version >=2.3.0 supports delay_wgrad_compute now.")
        if (
            self.config.tp_comm_overlap
            and tp_comm_buffer_name
            and tp_comm_buffer_name not in ['qkv', 'proj', 'fc1', 'fc2']
        ):
            self.config.tp_comm_overlap = False
            warnings.warn(
                f"The user buffer name {tp_comm_buffer_name} is not supported in"
                "Transformer Engine. Disabling TP communication overlap "
                "for this layer."
            )

        if is_te_min_version("0.8.0"):
            if self.config.tp_comm_overlap:
                if is_te_min_version("1.5.0"):
                    # Use old overlap flags if they were supplied instead
                    extra_kwargs["ub_overlap_ag"] = (
                        self.config.tp_comm_overlap_ag
                        if hasattr(self.config, "tp_comm_overlap_ag")
                        else self.config.tp_comm_split_ag or self.config.tp_comm_atomic_ag
                    )
                    extra_kwargs["ub_overlap_rs"] = (
                        self.config.tp_comm_overlap_rs
                        if hasattr(self.config, "tp_comm_overlap_rs")
                        else self.config.tp_comm_split_rs or self.config.tp_comm_atomic_rs
                    )
                    # Disable ub overlap for experts.
                    if is_expert:
                        extra_kwargs["ub_overlap_ag"] = False
                        extra_kwargs["ub_overlap_rs"] = False
                else:
                    extra_kwargs["ub_split_ag"] = self.config.tp_comm_split_ag
                    extra_kwargs["ub_atomic_gemm_ag"] = self.config.tp_comm_atomic_ag
                    extra_kwargs["ub_split_rs"] = self.config.tp_comm_split_rs
                    extra_kwargs["ub_atomic_gemm_rs"] = self.config.tp_comm_atomic_rs
                    # Disable ub overlap for experts.
                    if is_expert:
                        extra_kwargs["ub_split_ag"] = False
                        extra_kwargs["ub_atomic_gemm_ag"] = False
                        extra_kwargs["ub_split_rs"] = False
                        extra_kwargs["ub_atomic_gemm_rs"] = False
                if is_te_min_version("1.0.0", check_equality=False):
                    assert (
                        tp_comm_buffer_name is not None
                    ), "Buffer name should be set to configure communication overlap settings"
                    extra_kwargs["ub_name"] = tp_comm_buffer_name

        if symmetric_ar_type is not None:
            assert is_torch_min_version("2.7.0a0"), "Must have at least torch version 2.7 or higher"
            assert is_te_min_version("2.3.0") or get_te_version() == PkgVersion(
                "2.3.0.dev0+39c0e70"
            ), "Must have at least TE version 2.3 or higher to use symmetric memory all reduce"
            extra_kwargs["symmetric_ar_type"] = symmetric_ar_type
        if parallel_mode == "duplicated":
            assert tp_group is None, "duplicated linear should not have tp_group set"
            tp_size = 1
        else:
            assert tp_group is not None, "Parallel linear should always have tp_group set"
            tp_size = tp_group.size()

        self.expert_parallel = self.config.expert_model_parallel_size > 1
        if is_expert:
            rng_tracker_name = get_expert_parallel_rng_tracker_name()
        else:
            if parallel_mode == "duplicated":
                rng_tracker_name = get_data_parallel_rng_tracker_name()
            else:
                rng_tracker_name = None
        if is_te_min_version("1.7.0"):
            extra_kwargs["rng_tracker_name"] = rng_tracker_name

        te_parallel_mode = parallel_mode
        if parallel_mode == "duplicated":
            # Handle non-parallel case
            tp_group = None
            tp_size = 1
            explicit_expert_comm = False
            te_parallel_mode = None
        else:
            # Disable communications in TE when using TP or EP by
            explicit_expert_comm = is_expert and (tp_size > 1 or self.expert_parallel)

            if explicit_expert_comm:
                if parallel_mode == "column":
                    output_size = divide(output_size, tp_size)
                elif parallel_mode == "row":
                    input_size = divide(input_size, tp_size)
                te_parallel_mode = None
                tp_size = 1
                tp_group = None

        super().__init__(
            in_features=input_size,
            out_features=output_size,
            sequence_parallel=self.config.sequence_parallel,
            fuse_wgrad_accumulation=self.config.gradient_accumulation_fusion,
            # Pass None if not initialized for backward compatibility with the ckpt converter.
            tp_group=tp_group if torch.distributed.is_initialized() else None,
            tp_size=tp_size,
            get_rng_state_tracker=(
                get_cuda_rng_tracker if get_cuda_rng_tracker().is_initialized() else None
            ),
            init_method=condition_init_method(config, init_method),
            bias=bias,
            return_bias=self.te_return_bias,
            parallel_mode=te_parallel_mode,
            **extra_kwargs,
        )

        for param in self.parameters():
            if is_expert:
                # Reduce the gradient on the expert_data_parallel group for expert linear layers
                setattr(param, 'allreduce', not self.expert_parallel)
            else:
                # Reduce the gradient on DP group
                setattr(param, 'allreduce', True)
                if parallel_mode == "duplicated":
                    # Reduce the gradient further on the TP group since the weight is
                    # duplicated across TP ranks
                    setattr(param, 'sequence_parallel', self.config.sequence_parallel)

    def forward(self, x):
        """Forward."""
        _is_first_microbatch = (
            None if self.disable_parameter_transpose_cache else self.is_first_microbatch
        )
        out = super().forward(x, is_first_microbatch=_is_first_microbatch)
        self.is_first_microbatch = False

        # TE only returns a tuple when return_bias is True, otherwise
        # it returns a single Tensor, we always want to return two
        # values regardless of the arguments.
        if self.te_return_bias:
            return out
        return out, None

    def sharded_state_dict(self, prefix='', sharded_offsets=(), metadata=None):
        """Replicate cross TP/DP."""

        # Provide the dist-ckpt support when TELinear is directly used
        # It can only happen with duplicated parallel mode
        assert (
            self.parallel_mode is None
        ), "TELinear sharded_state_dict can only be used with duplicated parallel mode"
        state_dict = self.state_dict(prefix='', keep_vars=True)
        return make_sharded_tensors_for_checkpoint(state_dict, prefix, None, sharded_offsets)

    def backward_dw(self):
        """Compute weight gradients during the backward pass if delay_wgrad_compute is enabled."""
        if self.config.delay_wgrad_compute:
            super().backward_dw()


class TELayerNormColumnParallelLinear(te.pytorch.LayerNormLinear):
    """
    Wrapper for the Transformer-Engine's `LayerNormLinear` layer that combines
    layernorm and linear layers
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        *,
        config: TransformerConfig,
        init_method: Callable,
        gather_output: bool,
        bias: bool,
        skip_bias_add: bool,
        is_expert: bool,
        skip_weight_param_allocation: bool = False,
        tp_comm_buffer_name: Optional[str] = None,
        tp_group: Optional[torch.distributed.ProcessGroup] = None,
    ):
        self.config = config

        if gather_output:
            raise ValueError('Transformer Engine linear layers do not support gather_output = True')

        if is_expert:
            raise ValueError('Transformer Engine linear layers do not yet support MoE')

        if skip_weight_param_allocation:
            raise ValueError(
                'Transformer Engine linear layers do not support skip_weight_param_allocation'
            )

        # TODO: For backward compatibility, remove in v0.15.
        tp_group = get_tensor_model_parallel_group_if_none(tp_group, is_expert=is_expert)

        # TE returns a zero length Tensor when bias=False and
        # return_bias=True, but we prefer None.  So in that case we
        # tell TE to not return the bias, and return None
        # ourselves. This way our forward always returns two values
        # and we don't have to deal with the zero length Tensor.
        self.te_return_bias = skip_bias_add and bias
        self.is_first_microbatch = True
        self.disable_parameter_transpose_cache = self.config.disable_parameter_transpose_cache
        extra_kwargs = _get_extra_te_kwargs(config)
        self.tp_size = tp_group.size()
        self.tp_rank = tp_group.rank()

        if self.config.delay_wgrad_compute:
            if is_te_min_version("2.3.0"):
                extra_kwargs["delay_wgrad_compute"] = self.config.delay_wgrad_compute
            else:
                raise RuntimeError("Only TE with version >=2.3.0 supports delay_wgrad_compute now.")

        # Only Transformer-Engine version >= 0.11.0 supports `RMSNorm`
        if is_te_min_version("0.11.0"):
            extra_kwargs["normalization"] = self.config.normalization
        elif self.config.normalization != "LayerNorm":
            te_version = get_te_version()
            raise ValueError(
                f"Transformer Engine v{te_version} does not support {self.config.normalization}."
            )

        if is_te_min_version("0.8.0"):
            if self.config.tp_comm_overlap:
                extra_kwargs["ub_bulk_wgrad"] = self.config.tp_comm_bulk_wgrad
                extra_kwargs["ub_bulk_dgrad"] = self.config.tp_comm_bulk_dgrad
                if is_te_min_version("1.5.0", check_equality=False):
                    # Use old overlap flags if they were supplied instead
                    extra_kwargs["ub_overlap_ag"] = (
                        self.config.tp_comm_overlap_ag
                        if hasattr(self.config, "tp_comm_overlap_ag")
                        else self.config.tp_comm_split_ag or self.config.tp_comm_atomic_ag
                    )
                    if is_te_min_version("1.6.0.dev0", check_equality=False):
                        extra_kwargs["ub_overlap_rs_dgrad"] = (
                            self.config.tp_comm_overlap_rs_dgrad
                            if hasattr(self.config, "tp_comm_overlap_rs_dgrad")
                            else False
                        )
                    if tp_comm_buffer_name == 'qkv' and self.config.tp_comm_overlap_disable_qkv:
                        extra_kwargs["ub_overlap_ag"] = False
                        extra_kwargs["ub_overlap_rs_dgrad"] = False

                    if tp_comm_buffer_name == 'fc1' and self.config.tp_comm_overlap_disable_fc1:
                        extra_kwargs["ub_overlap_ag"] = False
                        extra_kwargs["ub_overlap_rs_dgrad"] = False
                else:
                    extra_kwargs["ub_atomic_gemm_ag"] = self.config.tp_comm_atomic_ag
                    extra_kwargs["ub_split_ag"] = self.config.tp_comm_split_ag
                if is_te_min_version("1.0.0", check_equality=False):
                    assert (
                        tp_comm_buffer_name is not None
                    ), "Buffer name should be set to configure communication overlap settings"
                    extra_kwargs["ub_name"] = tp_comm_buffer_name

        if self.config.symmetric_ar_type is not None:
            assert is_torch_min_version("2.7.0a0"), "Must have at least torch version 2.7 or higher"
            assert is_te_min_version("2.3.0") or get_te_version() == PkgVersion(
                "2.3.0.dev0+39c0e70"
            ), "Must have at least TE version 2.3 or higher to use symmetric memory all reduce"
            extra_kwargs["symmetric_ar_type"] = self.config.symmetric_ar_type

        super().__init__(
            in_features=input_size,
            out_features=output_size,
            eps=self.config.layernorm_epsilon,
            sequence_parallel=self.config.sequence_parallel,
            fuse_wgrad_accumulation=self.config.gradient_accumulation_fusion,
            tp_group=tp_group if torch.distributed.is_initialized() else None,
            tp_size=self.config.tensor_model_parallel_size,
            get_rng_state_tracker=(
                get_cuda_rng_tracker if get_cuda_rng_tracker().is_initialized() else None
            ),
            init_method=(
                condition_init_method(config, init_method)
                if not config.use_cpu_initialization
                else lambda w: None
            ),
            bias=bias,
            return_bias=self.te_return_bias,
            parallel_mode="column",
            return_layernorm_output=False,
            zero_centered_gamma=self.config.layernorm_zero_centered_gamma,
            **extra_kwargs,
        )

        if config.use_cpu_initialization:
            output_size_per_partition = divide(output_size, self.tp_size)
            _ = _initialize_affine_weight_cpu(
                self.weight,
                output_size,
                input_size,
                output_size_per_partition,
                0,
                init_method=condition_init_method(config, init_method),
                stride=1,
                return_master_weight=False,
                rank=self.tp_rank,
                world_size=self.tp_size,
                skip_set_tensor_parallel_attributes=True,
            )
            if bias:
                self.bias = Parameter(
                    torch.empty(output_size_per_partition, dtype=config.params_dtype)
                )
                set_tensor_model_parallel_attributes(self.bias, True, 0, 1)
                with torch.no_grad():
                    self.bias.zero_()
                setattr(self.bias, 'allreduce', True)

    def forward(self, x):
        """Forward."""
        _is_first_microbatch = (
            None if self.disable_parameter_transpose_cache else self.is_first_microbatch
        )
        out = super().forward(x, is_first_microbatch=_is_first_microbatch)
        self.is_first_microbatch = False

        # TE only returns a tuple when return_bias is True, otherwise
        # it returns a single Tensor, we always want to return two
        # values regardless of the arguments.
        if self.te_return_bias:
            return out
        return out, None

    def sharded_state_dict(self, prefix='', sharded_offsets=(), metadata=None):
        """Sharding along axis 0, bias sharded"""
        state_dict = self.state_dict(prefix='', keep_vars=True)
        return make_sharded_tensors_for_checkpoint(
            state_dict, prefix, {'weight': 0, 'bias': 0}, sharded_offsets
        )

    def __repr__(self):
        return (
            f"{type(self).__name__}(in_features={self.in_features}, "
            f"out_features={self.out_features}, bias={self.use_bias}, TP={self.tp_size})"
        )

    def backward_dw(self):
        """Compute weight gradients during the backward pass if delay_wgrad_compute is enabled."""
        if self.config.delay_wgrad_compute:
            super().backward_dw()


class TEColumnParallelLinear(TELinear):
    """
    Wrapper for the Transformer-Engine's `Linear` layer but specialized similar
    to megatron's `ColumnParallelLinear` layer.
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        *,
        config: ModelParallelConfig,
        init_method: Callable,
        gather_output: bool,
        bias: bool,
        skip_bias_add: bool,
        is_expert: bool,
        skip_weight_param_allocation: bool = False,
        tp_comm_buffer_name: Optional[str] = None,
        tp_group: Optional[torch.distributed.ProcessGroup] = None,
    ):
        if gather_output:
            raise ValueError('Transformer Engine linear layers do not support gather_output = True')
        tp_group = get_tensor_model_parallel_group_if_none(tp_group, is_expert=is_expert)
        world_size = tp_group.size()
        rank = tp_group.rank()

        super().__init__(
            input_size=input_size,
            output_size=output_size,
            parallel_mode="column",
            config=config,
            init_method=(
                condition_init_method(config, init_method)
                if not config.use_cpu_initialization
                else lambda w: None
            ),
            bias=bias,
            skip_bias_add=skip_bias_add,
            is_expert=is_expert,
            skip_weight_param_allocation=skip_weight_param_allocation,
            tp_comm_buffer_name=tp_comm_buffer_name,
            symmetric_ar_type=config.symmetric_ar_type,
            tp_group=tp_group,
        )

        if config.use_cpu_initialization:
            output_size_per_partition = divide(output_size, world_size)
            _ = _initialize_affine_weight_cpu(
                self.weight,
                output_size,
                input_size,
                output_size_per_partition,
                0,
                init_method=condition_init_method(config, init_method),
                stride=1,
                return_master_weight=False,
                rank=rank,
                world_size=world_size,
                skip_set_tensor_parallel_attributes=True,
            )
            if bias:
                self.bias = Parameter(
                    torch.empty(output_size_per_partition, dtype=config.params_dtype)
                )
                set_tensor_model_parallel_attributes(self.bias, True, 0, 1)
                with torch.no_grad():
                    self.bias.zero_()
                setattr(self.bias, 'allreduce', True)

    def sharded_state_dict(self, prefix='', sharded_offsets=(), metadata=None):
        """Sharding along axis 0, bias sharded"""
        state_dict = self.state_dict(prefix='', keep_vars=True)
        return make_sharded_tensors_for_checkpoint(
            state_dict, prefix, {'weight': 0, 'bias': 0}, sharded_offsets
        )

    def __repr__(self):
        return (
            f"{type(self).__name__}(in_features={self.in_features}, "
            f"out_features={self.out_features}, bias={self.use_bias}, TP={self.tp_size})"
        )

    def backward_dw(self):
        """Compute weight gradients during the backward pass if delay_wgrad_compute is enabled."""
        if self.config.delay_wgrad_compute:
            super().backward_dw()


class TERowParallelLinear(TELinear):
    """
    Wrapper for the Transformer-Engine's `Linear` layer but specialized similar
    to megatron's `RowParallelLinear` layer.
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        *,
        config: ModelParallelConfig,
        init_method: Callable,
        bias: bool,
        input_is_parallel: bool,
        skip_bias_add: bool,
        is_expert: bool,
        tp_comm_buffer_name: Optional[str] = None,
        tp_group: Optional[torch.distributed.ProcessGroup] = None,
    ):
        if not input_is_parallel:
            raise ValueError(
                "Transformer Engine linear layers do not support input_is_parallel = False"
            )
        tp_group = get_tensor_model_parallel_group_if_none(tp_group, is_expert=is_expert)

        super().__init__(
            input_size=input_size,
            output_size=output_size,
            parallel_mode="row",
            config=config,
            init_method=(
                condition_init_method(config, init_method)
                if not config.use_cpu_initialization
                else lambda w: None
            ),
            bias=bias,
            skip_bias_add=skip_bias_add,
            skip_weight_param_allocation=False,
            # We don't currently use this for row parallel layers # pylint: disable=line-too-long
            is_expert=is_expert,
            tp_comm_buffer_name=tp_comm_buffer_name,
            symmetric_ar_type=config.symmetric_ar_type,
            tp_group=tp_group,
        )
        if config.use_cpu_initialization:
            world_size = tp_group.size()
            rank = tp_group.rank()
            input_size_per_partition = divide(input_size, world_size)
            self.master_weight = _initialize_affine_weight_cpu(
                self.weight,
                output_size,
                input_size,
                input_size_per_partition,
                1,
                init_method=condition_init_method(config, init_method),
                stride=1,
                return_master_weight=False,
                params_dtype=config.params_dtype,
                rank=rank,
                world_size=world_size,
                skip_set_tensor_parallel_attributes=True,
            )
            if bias:
                self.bias = Parameter(torch.empty(output_size, dtype=config.params_dtype))
                # Always initialize bias to zero.
                with torch.no_grad():
                    self.bias.zero_()
                setattr(self.bias, 'allreduce', True)
                setattr(self.bias, 'sequence_parallel', config.sequence_parallel)

    def sharded_state_dict(self, prefix='', sharded_offsets=(), metadata=None):
        """Sharding along axis 1, bias not sharded"""
        state_dict = self.state_dict(prefix='', keep_vars=True)
        return make_sharded_tensors_for_checkpoint(
            state_dict, prefix, {'weight': 1}, sharded_offsets
        )

    def __repr__(self):
        return (
            f"{type(self).__name__}(in_features={self.in_features}, "
            f"out_features={self.out_features}, bias={self.use_bias}, TP={self.tp_size})"
        )

    def backward_dw(self):
        """Compute weight gradients during the backward pass if delay_wgrad_compute is enabled."""
        if self.config.delay_wgrad_compute:
            super().backward_dw()


class TEDotProductAttention(te.pytorch.DotProductAttention):
    """
    Wrapper for the Transformer-Engine's `DotProductAttention` layer that also
    has "flash attention" enabled.

    Note that if Megatron's parallel_state has not been initialized yet, the
    tp_group and cp_group passed to TE will be None and must be set later
    via set_tensor_parallel_group() and set_context_parallel_group().
    """

    cp_stream: torch.cuda.Stream = None

    def __init__(
        self,
        config: TransformerConfig,
        layer_number: int,
        attn_mask_type: AttnMaskType,
        attention_type: str,
        attention_dropout: Optional[float] = None,
        softmax_scale: Optional[float] = None,
        k_channels: Optional[int] = None,
        v_channels: Optional[int] = None,
        cp_comm_type: str = "p2p",
        model_comm_pgs: ModelCommProcessGroups = None,
    ):
        self.config = config
        self.te_forward_mask_type = False
        self.qkv_format: str = 'sbhd'

        if self.config.apply_query_key_layer_scaling != bool(
            int(os.getenv('NVTE_APPLY_QK_LAYER_SCALING', '0'))
        ):
            raise ValueError(
                f"apply_query_key_layer_scaling is {self.config.apply_query_key_layer_scaling} "
                f"but environment variable NVTE_APPLY_QK_LAYER_SCALING is "
                f"{os.getenv('NVTE_APPLY_QK_LAYER_SCALING')}. Transformer Engine does not support "
                f"setting query key layer scaling via argument, so these two must match."
            )

        extra_kwargs: dict[str, Any] = {}
        if is_te_min_version("0.11.0"):
            extra_kwargs["num_gqa_groups"] = self.config.num_query_groups
        elif self.config.num_query_groups != self.config.num_attention_heads:
            raise ValueError(
                f"Transformer Engine v{get_te_version()} does not support Grouped Query Attention, "
                f"use a newer version of Transformer Engine. "
                f"(num_query_groups ({self.config.num_query_groups}) != "
                f"num_attention_heads ({self.config.num_attention_heads}))"
            )

        if model_comm_pgs is None:
            # For backward compatibility, remove in v0.14 and raise error
            # raise ValueError("TEDotProductAttention was called without ModelCommProcessGroups")
            model_comm_pgs = ModelCommProcessGroups(
                tp=get_tensor_model_parallel_group(check_initialized=False),
                cp=get_context_parallel_group(check_initialized=False),
                hcp=get_hierarchical_context_parallel_groups(check_initialized=False),
            )
        else:
            assert hasattr(
                model_comm_pgs, 'tp'
            ), "TEDotProductAttention model_comm_pgs must have tp pg"
            assert hasattr(
                model_comm_pgs, 'cp'
            ), "TEDotProductAttention model_comm_pgs must have cp pg"
            if cp_comm_type == "a2a+p2p":
                assert hasattr(
                    model_comm_pgs, 'hcp'
                ), "TEDotProductAttention model_comm_pgs must have hierarchical cp pg"

        if is_te_min_version("0.10.0"):
            extra_kwargs["attention_type"] = attention_type
            # older version don't need attention_type

        if is_te_min_version("0.12.0", check_equality=False):
            self.te_forward_mask_type = True

        # This check is important as CP config can be disabled while having a valid CP group
        # Example - Disabling CP for encoder while a valid CP group exists for decoder
        if self.config.context_parallel_size > 1:
            assert is_te_min_version(
                "1.0.0"
            ), "Only Transformer-Engine version >= 1.0.0 supports context parallelism!"
            if getattr(TEDotProductAttention, "cp_stream") is None:
                TEDotProductAttention.cp_stream = torch.cuda.Stream()
            extra_kwargs["cp_group"] = model_comm_pgs.cp
            extra_kwargs["cp_global_ranks"] = torch.distributed.get_process_group_ranks(
                model_comm_pgs.cp
            )
            extra_kwargs["cp_stream"] = TEDotProductAttention.cp_stream
            if is_te_min_version("1.10.0"):
                if cp_comm_type is None:
                    extra_kwargs["cp_comm_type"] = "p2p"
                elif cp_comm_type == "a2a+p2p":
                    assert is_te_min_version("1.12.0"), (
                        f"Transformer-Engine v{get_te_version()} must be >= 1.12.0 to support"
                        "hierarchical cp commucation."
                    )
                    extra_kwargs["cp_comm_type"] = "a2a+p2p"
                    extra_kwargs["cp_group"] = get_hierarchical_context_parallel_groups(
                        check_initialized=False
                    )
                else:
                    extra_kwargs["cp_comm_type"] = cp_comm_type

        if self.config.deterministic_mode:
            if int(os.getenv("NVTE_ALLOW_NONDETERMINISTIC_ALGO", "1")) != 0:
                raise RuntimeError(
                    "deterministic_mode is on and we are using DotProductAttention from "
                    "Transformer Engine, but NVTE_ALLOW_NONDETERMINISTIC_ALGO is not 0. "
                    f"Currently set to: {os.getenv('NVTE_ALLOW_NONDETERMINISTIC_ALGO', 'not set')}."
                )

        if config.window_size is not None:
            # Check version
            assert is_te_min_version("1.2.0"), (
                f"Transformer-Engine v{get_te_version()} must be >= 1.2.0 to support"
                "sliding window attention."
            )
            extra_kwargs['window_size'] = config.window_size

        if is_te_min_version("1.10.0"):
            # TE 1.10.0 introduces the ability to set the different k and v channels
            kv_channels = (
                (k_channels, v_channels)
                if k_channels is not None and v_channels is not None
                else self.config.kv_channels
            )
            extra_kwargs['softmax_scale'] = softmax_scale
        else:
            kv_channels = self.config.kv_channels

        self.kept_packed_seq_params = set(
            field.name for field in dataclasses.fields(PackedSeqParams)
        )
        if get_te_version() < PkgVersion("1.3.0"):
            # TE 1.3.0 introduces precomputing max_seqlen to remove unnecessary kernels and D2H
            # copies (#555)
            # These two arguments did not exist prior to 1.3.0
            self.kept_packed_seq_params.discard("max_seqlen_q")
            self.kept_packed_seq_params.discard("max_seqlen_kv")

        if get_te_version() < PkgVersion("1.10.0"):
            # TE 1.8.0 introduces cu_seqlens_padded which is the cu_seqlens with paddings counted
            # in each individual sequence in THD format dataset
            # These two arguments did not exist prior to 1.8.0. Full support added in 1.10.0 (#1012)
            self.kept_packed_seq_params.discard("cu_seqlens_q_padded")
            self.kept_packed_seq_params.discard("cu_seqlens_kv_padded")

        super().__init__(
            num_attention_heads=self.config.num_attention_heads,
            kv_channels=kv_channels,
            attention_dropout=(
                self.config.attention_dropout if attention_dropout is None else attention_dropout
            ),
            attn_mask_type=attn_mask_type.name,
            sequence_parallel=self.config.sequence_parallel,
            tp_size=self.config.tensor_model_parallel_size,
            get_rng_state_tracker=(
                get_cuda_rng_tracker if get_cuda_rng_tracker().is_initialized() else None
            ),
            tp_group=model_comm_pgs.tp,
            layer_number=layer_number,
            **extra_kwargs,
        )

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        attention_mask: Tensor,
        attn_mask_type: AttnMaskType,
        attention_bias: Tensor = None,
        packed_seq_params: PackedSeqParams = None,
    ):
        """Forward."""
        packed_seq_kwargs = (
            {key: getattr(packed_seq_params, key) for key in self.kept_packed_seq_params}
            if packed_seq_params is not None
            else {}
        )
        # overwrite self.qkv_format depending on self.config.apply_rope_fusion, which can be set
        # after init
        if self.config.apply_rope_fusion and is_te_min_version("0.13.0", check_equality=False):
            self.qkv_format = 'bshd'

        qkv_format = packed_seq_kwargs.get('qkv_format', self.qkv_format)

        # WAR for peak memory usage.
        # See https://gitlab-master.nvidia.com/ADLR/megatron-lm/-/merge_requests/2388
        if self.config.apply_rope_fusion and qkv_format == 'bshd':
            query, key, value = [x.transpose(0, 1).contiguous() for x in (query, key, value)]
            # In PyTorch, the following two tensors are in fact the same:
            #   Tensor with shape (1, S, H, D) and stride (S*H*D, H*D, D, 1)
            #   Tensor with shape (1, S, H, D) and stride (H*D, H*D, D, 1)
            # Stride for a dimension that is 1 has no meaning, so tensors created two different ways
            # can have same shape but different strides.
            # We unify them to the first one to pass the stride check in TE
            if value.shape == key.shape and value.shape[0] == 1 and value.stride() != key.stride():
                value = value.as_strided(value.shape, key.stride())

        attention_bias_kwargs = {}
        if attention_bias is not None:
            assert is_te_min_version("1.2.0"), (
                f"Transformer-Engine v{get_te_version()} must be >= 1.2.0 to support"
                "`attention_bias`."
            )
            attention_bias_kwargs = dict(
                core_attention_bias_type='post_scale_bias', core_attention_bias=attention_bias
            )

        if self.te_forward_mask_type:
            if qkv_format == 'thd' and is_te_min_version("1.7.0"):
                # thd format uses flash attention with cuDNN kernel which requires is_padding=True,
                # so the only acceptable mask types are `padding_causal` and `padding`. These do not
                # necessarily indicate there are padded tokens in the sequence.
                if attn_mask_type == AttnMaskType.causal:
                    attn_mask_type = AttnMaskType.padding_causal
                elif attn_mask_type == AttnMaskType.no_mask:
                    attn_mask_type = AttnMaskType.padding
            core_attn_out = super().forward(
                query,
                key,
                value,
                attention_mask,
                attn_mask_type=attn_mask_type.name,
                **attention_bias_kwargs,
                **packed_seq_kwargs,
            )
        else:
            core_attn_out = super().forward(
                query, key, value, attention_mask, **attention_bias_kwargs, **packed_seq_kwargs
            )

        if self.config.apply_rope_fusion and qkv_format == 'bshd':
            return core_attn_out.transpose(0, 1)
        else:
            return core_attn_out


if is_te_min_version("1.9.0.dev0"):

    class TEGroupedLinear(te.pytorch.GroupedLinear):
        """
        Wrapper for the Transformer-Engine's `GroupedLinear` layer.

        Note that if Megatron's parallel_state has not been initialized
        yet, the tp_group passed to TE will be None and must be set later
        via set_tensor_parallel_group().
        """

        def __init__(
            self,
            num_gemms: int,
            input_size: int,
            output_size: int,
            *,
            parallel_mode: Optional[str],
            config: ModelParallelConfig,
            init_method: Callable,
            bias: bool,
            skip_bias_add: bool,
            is_expert: bool = False,
            tp_comm_buffer_name: Optional[str] = None,
            tp_group: Optional[torch.distributed.ProcessGroup] = None,
        ):
            self.config = config

            # TE returns a zero length Tensor when bias=False and
            # return_bias=True, but we prefer None.  So in that case we
            # tell TE to not return the bias, and return None
            # ourselves. This way our forward always returns two values
            # and we don't have to deal with the zero length Tensor.
            self.te_return_bias = skip_bias_add and bias
            self.is_first_microbatch = True
            self.disable_parameter_transpose_cache = self.config.disable_parameter_transpose_cache

            extra_kwargs = _get_extra_te_kwargs(config)

            if self.config.delay_wgrad_compute:
                if is_te_min_version("2.3.0"):
                    extra_kwargs["delay_wgrad_compute"] = self.config.delay_wgrad_compute
                else:
                    raise RuntimeError(
                        "Only TE with version >=2.3.0 supports delay_wgrad_compute now."
                    )

            extra_kwargs["ub_name"] = tp_comm_buffer_name

            self.expert_parallel = self.config.expert_model_parallel_size > 1
            if is_expert:
                extra_kwargs["rng_tracker_name"] = get_expert_parallel_rng_tracker_name()

            # The comms between TP and EP group is explicitly handled by MoE token dispatcher.
            # So we disable comms by making TE agnostic of model parallel.
            tp_group = get_tensor_model_parallel_group_if_none(tp_group, is_expert=is_expert)
            tp_size = tp_group.size()

            self.explicit_expert_comm = is_expert and (tp_size > 1 or self.expert_parallel)

            if self.explicit_expert_comm:
                if parallel_mode == "column":
                    output_size = divide(output_size, tp_size)
                elif parallel_mode == "row":
                    input_size = divide(input_size, tp_size)
                parallel_mode = None
                tp_size = 1
                tp_group = None

            super().__init__(
                num_gemms=num_gemms,
                in_features=input_size,
                out_features=output_size,
                sequence_parallel=self.config.sequence_parallel,
                fuse_wgrad_accumulation=self.config.gradient_accumulation_fusion,
                tp_group=tp_group if torch.distributed.is_initialized() else None,
                tp_size=tp_size,
                get_rng_state_tracker=(
                    get_cuda_rng_tracker if get_cuda_rng_tracker().is_initialized() else None
                ),
                init_method=condition_init_method(config, init_method),
                bias=bias,
                return_bias=self.te_return_bias,
                parallel_mode=parallel_mode,
                **extra_kwargs,
            )

            for param in self.parameters():
                setattr(param, 'allreduce', not (is_expert and self.expert_parallel))

            def merge_extra_states(
                self,
                state_dict,
                prefix,
                local_metadata,
                strict,
                missing_keys,
                unexpected_keys,
                error_msgs,
            ):
                """
                Merge multiple "_extra_state" into one.
                """
                self.init_fp8_metadata(num_gemms=self.num_gemms)
                # When resume training, loading ckpt is out of fp8_autocast context.
                # So we need to manually detect from the state_dict.
                fp8_checkpoint = any("_extra_state" in str(key) for key in state_dict.keys())

                if not fp8_checkpoint:
                    return

                try:
                    state_list = [
                        state_dict.pop(f"{prefix}_extra_state{i}") for i in range(1, self.num_gemms)
                    ]
                except KeyError:
                    # "_extra_state{i}" only exists for dist-ckpt. Return for torch native ckpt.
                    return

                # Early return conditions:
                # 1. Empty state_dict
                # 2. Empty state_list
                # 3. _extra_state is None
                # 4. _extra_state does not contain any information
                if (
                    not state_dict
                    or not state_list
                    or state_dict.get(f"{prefix}_extra_state") is None
                    or self._decode_extra_state(state_dict[f"{prefix}_extra_state"]) is None
                ):
                    return

                state_list = [state_dict.pop(f"{prefix}_extra_state")] + state_list
                state_list = [self._decode_extra_state(state) for state in state_list]
                extra_fp8_variables = state_list[0]['extra_fp8_variables']
                extra_fp8_variables['num_gemms'] = self.num_gemms
                extra_state = {"extra_fp8_variables": extra_fp8_variables}
                # TE 2.0 adds recipe in extra_state
                if is_te_min_version("2.0.0"):
                    self.fp8_meta["recipe"] = state_list[0]['recipe']
                    extra_state['recipe'] = self.fp8_meta["recipe"]
                # Only delayed scaling has global fp8 meta tensors. We're not using
                # self.fp8_meta["recipe"].delayed() because it's available in TE 2.0 and later.
                if isinstance(self.fp8_meta["recipe"], te.common.recipe.DelayedScaling):
                    extra_state.update(
                        {
                            "scale_fwd": torch.cat(
                                [state['scale_fwd'].view(-1, 1) for state in state_list], dim=1
                            ).view(-1),
                            "amax_history_fwd": torch.cat(
                                [state['amax_history_fwd'].view(-1, 1) for state in state_list],
                                dim=1,
                            ).view(self.fp8_meta["recipe"].amax_history_len, -1),
                            "scale_bwd": torch.cat(
                                [state['scale_bwd'].view(-1, 1) for state in state_list], dim=1
                            ).view(-1),
                            "amax_history_bwd": torch.cat(
                                [state['amax_history_bwd'].view(-1, 1) for state in state_list],
                                dim=1,
                            ).view(self.fp8_meta["recipe"].amax_history_len, -1),
                        }
                    )
                    # TE 2.0 removes scale_inv_fwd and scale_inv_bwd
                    if not is_te_min_version("2.0.0"):
                        extra_state.update(
                            {
                                "scale_inv_fwd": torch.cat(
                                    [state['scale_inv_fwd'].view(-1, 1) for state in state_list],
                                    dim=1,
                                ).view(-1),
                                "scale_inv_bwd": torch.cat(
                                    [state['scale_inv_bwd'].view(-1, 1) for state in state_list],
                                    dim=1,
                                ).view(-1),
                            }
                        )
                state_dict[f"{prefix}_extra_state"] = self._encode_extra_state(extra_state)

            self._register_load_state_dict_pre_hook(merge_extra_states, with_module=True)

        def forward(self, x, m_splits):
            """Forward."""
            _is_first_microbatch = (
                None if self.disable_parameter_transpose_cache else self.is_first_microbatch
            )
            out = super().forward(x, m_splits, is_first_microbatch=_is_first_microbatch)
            self.is_first_microbatch = False

            # TE only returns a tuple when return_bias is True, otherwise
            # it returns a single Tensor, we always want to return two
            # values regardless of the arguments.
            if self.te_return_bias:
                return out
            return out, None

        def _encode_extra_state(self, state):
            # TE 2.0 changed the format of extra_state to be a byte tensor
            if is_te_min_version("2.0.0"):
                torch.cuda.synchronize()
                state_serialized = bytearray(pickle.dumps(state))
                state_serialized = torch.frombuffer(state_serialized, dtype=torch.uint8)
            else:
                state_serialized = io.BytesIO()
                torch.save(state, state_serialized)
            return state_serialized

        def _decode_extra_state(self, state):
            if isinstance(state, torch.Tensor):
                # No FP8 is indicated by an empty tensor we don't need to unpickle.
                if state.numel() == 0:
                    return
                return pickle.loads(state.detach().cpu().numpy().tobytes())
            elif isinstance(state, io.BytesIO):
                state.seek(0)
                return torch.load(state, map_location="cuda")
            else:
                raise RuntimeError("Unsupported checkpoint format.")

        def _split_extra_state(self, state):
            fp8_checkpoint = self.fp8_meta["fp8_checkpoint"] or self.fp8 or self.fp8_calibration

            if not fp8_checkpoint:
                return [state] * self.num_gemms

            state = self._decode_extra_state(state)
            extra_states = []
            extra_fp8_variables = state['extra_fp8_variables']
            extra_fp8_variables['num_gemms'] = 1
            for gemm_idx in range(self.num_gemms):
                tmp_state = {"extra_fp8_variables": extra_fp8_variables}
                # TE 2.0 adds recipe in extra_state
                if is_te_min_version("2.0.0"):
                    tmp_state['recipe'] = state['recipe']
                # Only delayed scaling has global fp8 meta tensors. We're not using
                # self.fp8_meta["recipe"].delayed() because it's available in TE 2.0 and later.
                if isinstance(self.fp8_meta["recipe"], te.common.recipe.DelayedScaling):
                    tmp_state.update(
                        {
                            "scale_fwd": state['scale_fwd'].view(3, -1)[:, gemm_idx],
                            "amax_history_fwd": state['amax_history_fwd'].view(
                                self.fp8_meta["recipe"].amax_history_len, 3, -1
                            )[:, :, gemm_idx],
                            "scale_bwd": state['scale_bwd'].view(2, -1)[:, gemm_idx],
                            "amax_history_bwd": state['amax_history_bwd'].view(
                                self.fp8_meta["recipe"].amax_history_len, 2, -1
                            )[:, :, gemm_idx],
                        }
                    )
                    # TE 2.0 removes scale_inv_fwd and scale_inv_bwd
                    if not is_te_min_version("2.0.0"):
                        tmp_state.update(
                            {
                                "scale_inv_fwd": state['scale_inv_fwd'].view(3, -1)[:, gemm_idx],
                                "scale_inv_bwd": state['scale_inv_bwd'].view(2, -1)[:, gemm_idx],
                            }
                        )
                extra_states.append(self._encode_extra_state(tmp_state))
            return extra_states

        def _sharded_state_dict_grouped(
            self, tp_axis_map, prefix='', sharded_offsets=(), metadata=None
        ):
            """
            prefix should be module_name to make keys identical to sequetial ones.
            """
            sharded_state_dict = {}
            full_state_dict = self.state_dict(prefix='', keep_vars=True)
            num_global_experts = get_expert_model_parallel_world_size() * self.num_gemms
            local_expert_indices_offset = get_expert_model_parallel_rank() * self.num_gemms
            ep_axis = len(sharded_offsets)
            extra_states = self._split_extra_state(full_state_dict['_extra_state'])
            for gemm_idx in range(self.num_gemms):
                state_dict = {
                    f'{gemm_idx}.weight': full_state_dict[f'weight{gemm_idx}'],
                    f'{gemm_idx}._extra_state': extra_states[gemm_idx],
                }
                if self.use_bias:
                    state_dict[f'{gemm_idx}.bias'] = full_state_dict[f'bias{gemm_idx}']
                sub_sd = make_sharded_tensors_for_checkpoint(
                    state_dict,
                    '',
                    tp_axis_map,
                    (
                        *sharded_offsets,
                        (ep_axis, local_expert_indices_offset + gemm_idx, num_global_experts),
                    ),
                )
                # Remove expert layers indexing from sharded keys
                replace_prefix_for_sharding(sub_sd, f'{gemm_idx}.', prefix)
                sharded_state_dict.update(
                    {
                        f'{prefix}weight{gemm_idx}': sub_sd[f'{gemm_idx}.weight'],
                        f'{prefix}_extra_state{"" if gemm_idx == 0 else gemm_idx}': sub_sd[
                            f'{gemm_idx}._extra_state'
                        ],
                    }
                )
                if self.use_bias:
                    sharded_state_dict[f'{prefix}bias{gemm_idx}'] = sub_sd[f'{gemm_idx}.bias']
            # Adjust replica ids - replication along DP modulo EP
            for k, sh_ten in sharded_state_dict.items():
                replica_id = sh_ten.replica_id
                assert (
                    len(replica_id) == 3
                ), f'Expected replica_id for {k} to be in (PP, TP, DP) format, got: {replica_id}'
                if getattr(sh_ten, "is_data_parallel_fully_shard", False):
                    edp_replica_id = 0
                else:
                    edp_replica_id = get_expert_data_parallel_rank()
                sh_ten.replica_id = (*replica_id[:2], edp_replica_id)
            return sharded_state_dict

        def backward_dw(self):
            """
            Compute weight gradients during the backward pass
            if delay_wgrad_compute is enabled.
            """
            if self.config.delay_wgrad_compute:
                super().backward_dw()

    class TEColumnParallelGroupedLinear(TEGroupedLinear):
        """
        Wrapper for the Transformer-Engine's `GroupedLinear` layer but specialized
        to column-parallel style.
        """

        def __init__(
            self,
            num_gemms: int,
            input_size: int,
            output_size: int,
            *,
            config: ModelParallelConfig,
            init_method: Callable,
            bias: bool,
            skip_bias_add: bool,
            is_expert: bool,
            tp_comm_buffer_name: Optional[str] = None,
            tp_group: Optional[torch.distributed.ProcessGroup] = None,
        ):

            super().__init__(
                num_gemms=num_gemms,
                input_size=input_size,
                output_size=output_size,
                parallel_mode="column",
                config=config,
                init_method=condition_init_method(config, init_method),
                bias=bias,
                skip_bias_add=skip_bias_add,
                is_expert=is_expert,
                tp_comm_buffer_name=tp_comm_buffer_name,
                tp_group=tp_group,
            )

        def sharded_state_dict(self, prefix='', sharded_offsets=(), metadata=None):
            """
            For each gemm, sharding along axis 0, bias sharded.
            Assume sharded_offsets[-1] is the expert parallel offset.
            """
            tp_axis_map = {}
            for gemm_idx in range(self.num_gemms):
                tp_axis_map.update({f'{gemm_idx}.weight': 0, f'{gemm_idx}.bias': 0})
            return super()._sharded_state_dict_grouped(
                tp_axis_map, prefix, sharded_offsets, metadata
            )

    class TERowParallelGroupedLinear(TEGroupedLinear):
        """
        Wrapper for the Transformer-Engine's `GroupedLinear` layer but specialized
        to row-parallel style.
        """

        def __init__(
            self,
            num_gemms: int,
            input_size: int,
            output_size: int,
            *,
            config: ModelParallelConfig,
            init_method: Callable,
            bias: bool,
            skip_bias_add: bool,
            is_expert: bool,
            tp_comm_buffer_name: Optional[str] = None,
            tp_group: Optional[torch.distributed.ProcessGroup] = None,
        ):

            super().__init__(
                num_gemms=num_gemms,
                input_size=input_size,
                output_size=output_size,
                parallel_mode="row",
                config=config,
                init_method=condition_init_method(config, init_method),
                bias=bias,
                skip_bias_add=skip_bias_add,
                is_expert=is_expert,
                tp_comm_buffer_name=tp_comm_buffer_name,
                tp_group=tp_group,
            )

        def sharded_state_dict(self, prefix='', sharded_offsets=(), metadata=None):
            """
            For each gemm, sharding along axis 1, bias not sharded.
            Assume sharded_offsets[-1] is the expert parallel offset.
            """
            tp_axis_map = {f'{gemm_idx}.weight': 1 for gemm_idx in range(self.num_gemms)}
            return super()._sharded_state_dict_grouped(
                tp_axis_map, prefix, sharded_offsets, metadata
            )

else:

    TEGroupedLinear = None  # type: ignore[assignment, misc]
    TEColumnParallelGroupedLinear = None  # type: ignore[assignment, misc]
    TERowParallelGroupedLinear = None  # type: ignore[assignment, misc]


if is_te_min_version("1.13.0"):

    class TEFusedMLP(te.pytorch.ops.Sequential):
        """
        A fused MLP implementation using Transformer Engine's operation-based API
        """

        def __init__(
            self,
            config: TransformerConfig,
            *,
            is_expert: bool = False,
            input_size: Optional[int] = None,
            ffn_hidden_size: Optional[int] = None,
            tp_group: Optional[torch.distributed.ProcessGroup] = None,
        ):
            self.config: TransformerConfig = config

            # MoE is not supported
            # Note: This option is for compatibility with MLP class
            if is_expert:
                raise ValueError(
                    'Transformer Engine operation-based API does not support mixture-of-experts'
                )

            # Tensor-parallel group
            tp_group = get_tensor_model_parallel_group_if_none(tp_group)

            # Layer sizes
            if ffn_hidden_size is None:
                warnings.warn(
                    "MLP requires ffn_hidden_size, but it was not provided. Using "
                    "config.ffn_hidden_size by default.",
                    DeprecationWarning,
                    stacklevel=2,
                )
                ffn_hidden_size = config.ffn_hidden_size
            fc1_in_size = input_size if input_size != None else config.hidden_size
            fc1_out_size = 2 * ffn_hidden_size if config.gated_linear_unit else ffn_hidden_size
            fc2_in_size = ffn_hidden_size
            fc2_out_size = fc1_in_size

            # Linear ops
            fc1_op = te.pytorch.ops.Linear(
                in_features=fc1_in_size,
                out_features=fc1_out_size,
                sequence_parallel=config.sequence_parallel,
                tensor_parallel_group=tp_group,
                rng_state_tracker_function=(
                    get_cuda_rng_tracker if get_cuda_rng_tracker().is_initialized() else None
                ),
                bias=config.add_bias_linear,
            )
            fc2_op = te.pytorch.ops.Linear(
                in_features=fc2_in_size,
                out_features=fc2_out_size,
                sequence_parallel=config.sequence_parallel,
                tensor_parallel_group=tp_group,
                rng_state_tracker_function=(
                    get_cuda_rng_tracker if get_cuda_rng_tracker().is_initialized() else None
                ),
                bias=config.add_bias_linear,
            )

            # Normalization op
            norm_type: Type[te.pytorch.ops.FusibleOperation]
            if config.normalization == 'LayerNorm':
                norm_type = te.pytorch.ops.LayerNorm
            elif config.normalization == "RMSNorm":
                norm_type = te.pytorch.ops.RMSNorm
            else:
                raise ValueError(f"Unsupported normalization: {config.normalization}")
            norm_op = norm_type(
                fc1_in_size,
                eps=config.layernorm_epsilon,
                zero_centered_gamma=config.layernorm_zero_centered_gamma,
            )

            # Activation op
            activation_type = {
                (F.gelu, False): te.pytorch.ops.GELU,
                (F.gelu, True): te.pytorch.ops.GEGLU,
                (F.silu, True): te.pytorch.ops.SwiGLU,
                (F.relu, False): te.pytorch.ops.ReLU,
                (F.relu, True): te.pytorch.ops.ReGLU,
            }[config.activation_func, config.gated_linear_unit]
            activation_kwargs = {}
            if is_te_min_version("2.3"):
                activation_kwargs["cache_quantized_input"] = config.activation_func_fp8_input_store
            activation_op = activation_type(**activation_kwargs)

            # Construct layers
            super().__init__(norm_op, fc1_op, activation_op, fc2_op)

        def forward(self, hidden_states: Tensor) -> Tuple[Tensor, Optional[Tensor]]:
            """Forward."""
            out = super().forward(hidden_states)
            bias = self[-1].bias  # Bias from last layer
            return out, bias

        def sharded_state_dict(self, prefix='', sharded_offsets=(), metadata=None):
            """Sharding along axis 0, bias sharded"""
            state_dict = self.state_dict(prefix='', keep_vars=True)
            return make_sharded_tensors_for_checkpoint(
                state_dict, prefix, {'weight': 0, 'bias': 0}, sharded_offsets
            )

else:

    TEFusedMLP = None  # type: ignore[assignment, misc]


class TEDelayedScaling(te.common.recipe.DelayedScaling):
    """
    Wrapper for the Transformer-Engine's `DelayedScaling` layer.
    """

    def __init__(
        self,
        config: ModelParallelConfig,
        fp8_format: int,
        override_linear_precision: tuple = (False, False, False),
    ):
        extra_kwargs = _get_extra_te_kwargs(config)
        if is_te_min_version("1.6.0.dev0"):
            extra_kwargs["fp8_dpa"] = config.fp8_dot_product_attention
            extra_kwargs["fp8_mha"] = config.fp8_multi_head_attention
        if get_te_version() < PkgVersion("1.8.0"):
            extra_kwargs["interval"] = config.fp8_interval
        elif config.fp8_interval != 1:
            warnings.warn("fp8_interval is deprecated and ignored from Transformer-Engine v1.8.0.")

        super().__init__(
            margin=config.fp8_margin,
            fp8_format=fp8_format,
            amax_compute_algo=config.fp8_amax_compute_algo,
            amax_history_len=config.fp8_amax_history_len,
            override_linear_precision=override_linear_precision,
            **extra_kwargs,
        )


class TECudaRNGStatesTracker(te.pytorch.distributed.CudaRNGStatesTracker):
    """Wraps TransformerEngine's CudaRNGStatesTracker so that it is
    interchangeable with Megatron's RNG tracker"""

    def __init__(self, is_inference_rng_tracker=False):
        super().__init__()
        self.reset()
        self.is_inference_rng_tracker = is_inference_rng_tracker

    def is_initialized(self):
        """Checks if the internal RNG state has been set with set_states()."""
        return self._is_initialized

    def reset(self):
        """Reset the internal RNG state."""
        super().reset()
        self._is_initialized = False

    def set_states(self, states):
        """Set the internal RNG state."""
        super().set_states(states)
        self._is_initialized = True

    def add(self, name, seed):
        """Track the rng state."""
        super().add(name, seed)
        self._is_initialized = True


def te_checkpoint(
    forward_func, distribute_saved_activations, get_rng_state_tracker, tp_group, *args, **kwargs
):
    """Checkpointing with Transformer-Engine."""
    from transformer_engine.pytorch.distributed import checkpoint

    if is_te_min_version("1.5.0"):
        return checkpoint(
            forward_func,
            *args,
            distribute_saved_activations=distribute_saved_activations,
            get_rng_state_tracker=get_rng_state_tracker,
            tp_group=tp_group,
            **kwargs,
        )
    else:
        return checkpoint(
            forward_func, distribute_saved_activations, get_rng_state_tracker, tp_group, *args
        )


try:

    from transformer_engine.pytorch.attention import _SplitAlongDim

    SplitAlongDim = _SplitAlongDim.apply

except ImportError:

    SplitAlongDim = None

try:

    from transformer_engine.pytorch.cpu_offload import (
        get_cpu_offload_context as _get_cpu_offload_context,
    )

    def get_cpu_offload_context(
        enabled, num_layers, model_layers, activation_offloading, weight_offloading
    ):
        """Get CPU offload context and sync function."""
        if is_te_min_version("1.10.0.dev0"):
            context, sync_func = _get_cpu_offload_context(
                enabled, num_layers, model_layers, activation_offloading, weight_offloading
            )
        else:
            context, sync_func = _get_cpu_offload_context(
                enabled, num_layers, activation_offloading, weight_offloading
            )

        return context, sync_func

except ImportError:

    get_cpu_offload_context = None  # type: ignore[assignment, misc]

try:

    from transformer_engine.pytorch.attention import apply_rotary_pos_emb

    def fused_apply_rotary_pos_emb(
        t: torch.Tensor,
        freqs: torch.Tensor,
        transpose_output_memory: bool = False,
        interleaved: bool = False,
    ) -> torch.Tensor:
        """Apply rotary positional embedding to input tensor T in `sbhd` format."""
        if transpose_output_memory:
            warnings.warn(
                "transpose_output_memory is not supported by TE's fused RoPE and will be ignored."
            )
        if is_te_min_version("2.3.0.dev0"):
            return apply_rotary_pos_emb(
                t, freqs, tensor_format="sbhd", interleaved=interleaved, fused=True
            )
        else:
            if interleaved:
                raise ValueError("Only TE >= 2.3.0.dev0 supports interleaved fused RoPE.")
            return apply_rotary_pos_emb(t, freqs, tensor_format="sbhd", fused=True)

    def fused_apply_rotary_pos_emb_thd(
        t: torch.Tensor,
        cu_seqlens: torch.Tensor,
        freqs: torch.Tensor,
        cp_size: int = 1,
        cp_rank: int = 0,
    ) -> torch.Tensor:
        """
        Apply rotary positional embedding to input tensor T in `thd` format with CP support.
        """
        if is_te_min_version("1.12.0", check_equality=True):
            return apply_rotary_pos_emb(
                t,
                freqs,
                tensor_format="thd",
                fused=True,
                cu_seqlens=cu_seqlens,
                cp_size=cp_size,
                cp_rank=cp_rank,
            )
        else:
            return apply_rotary_pos_emb(
                t, freqs, tensor_format="thd", fused=True, cu_seqlens=cu_seqlens
            )

except ImportError:

    pass

try:

    from transformer_engine.pytorch import Fp8Padding, Fp8Unpadding  # pylint: disable=unused-import

except ImportError:

    Fp8Padding = None
    Fp8Unpadding = None

try:

    from transformer_engine.pytorch.permutation import (
        moe_permute,
        moe_permute_with_probs,
        moe_sort_chunks_by_index,
        moe_sort_chunks_by_index_with_probs,
        moe_unpermute,
    )

    fused_permute = moe_permute
    fused_permute_with_probs = moe_permute_with_probs
    fused_sort_chunks_by_index = moe_sort_chunks_by_index
    fused_sort_chunks_by_index_with_probs = moe_sort_chunks_by_index_with_probs
    fused_unpermute = moe_unpermute

except ImportError:

    fused_permute = None
    fused_permute_with_probs = None
    fused_sort_chunks_by_index = None
    fused_sort_chunks_by_index_with_probs = None
    fused_unpermute = None

try:

    from transformer_engine.pytorch.cross_entropy import parallel_cross_entropy

    def te_parallel_cross_entropy(
        logits: torch.Tensor, labels: torch.Tensor, tp_group: torch.distributed.ProcessGroup
    ):
        """Wrapper function for TE's Cross Entropy Loss kernel"""
        return parallel_cross_entropy(logits, labels, 0.0, False, tp_group)

except ImportError:

    te_parallel_cross_entropy = None  # type: ignore[assignment, misc]

try:

    from transformer_engine.pytorch.cpp_extensions import general_gemm
    from transformer_engine.pytorch.module.base import get_workspace

    def te_general_gemm(
        A: torch.Tensor,
        B: torch.Tensor,
        out_dtype: Optional[torch.dtype] = None,
        layout: str = "TN",
        out: Optional[torch.Tensor] = None,
        bias: Optional[torch.Tensor] = None,
        grad: bool = False,
    ) -> List[torch.Tensor]:
        """
        Wrapper for TE's general_gemm function.
        It supports fp32, bf16, fp16, and fp8 GEMMs with TN, NN, and NT layouts.
        The output dtype can be specified by `out_dtype`.
        Note: not all combinations of these settings are supported. If not supported,
        cublaslt will throw an error.
        """
        return general_gemm(
            A,
            B,
            workspace=get_workspace(),
            out_dtype=out_dtype,
            quantization_params=None,
            gelu=None,
            gelu_in=None,
            accumulate=False,
            layout=layout,
            out=out,
            bias=bias,
            use_split_accumulator=False,
            grad=grad,
            ub=None,
            ub_type=None,
            extra_output=None,
            bulk_overlap=False,
        )

except ImportError:

    te_general_gemm = None  # type: ignore[assignment, misc]
