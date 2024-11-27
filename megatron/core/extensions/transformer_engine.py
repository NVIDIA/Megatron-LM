# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

import dataclasses
import io
import os
import pickle
import warnings
from typing import Callable

import torch
import transformer_engine as te
from packaging.version import Version as PkgVersion
from torch import Tensor
from torch.nn.parameter import Parameter

from megatron.core import ModelParallelConfig
from megatron.core.dist_checkpointing.utils import replace_prefix_for_sharding
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.parallel_state import (
    get_context_parallel_global_ranks,
    get_context_parallel_group,
    get_expert_data_parallel_rank,
    get_expert_model_parallel_rank,
    get_expert_model_parallel_world_size,
    get_expert_tensor_parallel_group,
    get_expert_tensor_parallel_rank,
    get_expert_tensor_parallel_world_size,
    get_hierarchical_context_parallel_groups,
    get_tensor_model_parallel_group,
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
)
from megatron.core.tensor_parallel import get_cuda_rng_tracker, get_expert_parallel_rng_tracker_name
from megatron.core.tensor_parallel.layers import (
    _initialize_affine_weight_cpu,
    set_tensor_model_parallel_attributes,
)
from megatron.core.tensor_parallel.utils import divide
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.transformer.utils import make_sharded_tensors_for_checkpoint
from megatron.core.utils import get_te_version, is_te_min_version


def _get_extra_te_kwargs(config: TransformerConfig):
    extra_transformer_engine_kwargs = {"params_dtype": config.params_dtype}

    if is_te_min_version("0.12.0"):
        if config.use_cpu_initialization:
            extra_transformer_engine_kwargs["device"] = 'cpu'
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
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        *,
        parallel_mode: str,
        config: ModelParallelConfig,
        init_method: Callable,
        bias: bool,
        skip_bias_add: bool,
        skip_weight_param_allocation: bool,
        tp_comm_buffer_name: str = None,
        is_expert: bool = False,
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
        if skip_weight_param_allocation:
            raise ValueError(
                'Transformer Engine linear layers do not support skip_weight_param_allocation'
            )

        extra_kwargs = _get_extra_te_kwargs(config)

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

        self.expert_parallel = self.config.expert_model_parallel_size > 1
        if is_expert:
            rng_tracker_name = get_expert_parallel_rng_tracker_name()
        else:
            rng_tracker_name = None
        if is_te_min_version("1.7.0"):
            extra_kwargs["rng_tracker_name"] = rng_tracker_name

        # Disable communications in TE when using TP or EP by making TE agnostic of model parallel.
        if is_expert:
            tp_group = get_expert_tensor_parallel_group(check_initialized=False)
            tp_size = get_expert_tensor_parallel_world_size()
        else:
            tp_group = get_tensor_model_parallel_group(check_initialized=False)
            tp_size = get_tensor_model_parallel_world_size()
        explicit_expert_comm = is_expert and (tp_size > 1 or self.expert_parallel)

        if explicit_expert_comm:
            if parallel_mode == "column":
                output_size = divide(output_size, tp_size)
            elif parallel_mode == "row":
                input_size = divide(input_size, tp_size)
            parallel_mode = None
            tp_size = 1
            tp_group = None

        super().__init__(
            in_features=input_size,
            out_features=output_size,
            sequence_parallel=self.config.sequence_parallel,
            fuse_wgrad_accumulation=self.config.gradient_accumulation_fusion,
            tp_group=tp_group,
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
        tp_comm_buffer_name: str = None,
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

        # TE returns a zero length Tensor when bias=False and
        # return_bias=True, but we prefer None.  So in that case we
        # tell TE to not return the bias, and return None
        # ourselves. This way our forward always returns two values
        # and we don't have to deal with the zero length Tensor.
        self.te_return_bias = skip_bias_add and bias
        self.is_first_microbatch = True
        self.disable_parameter_transpose_cache = self.config.disable_parameter_transpose_cache
        extra_kwargs = _get_extra_te_kwargs(config)

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

        super().__init__(
            in_features=input_size,
            out_features=output_size,
            eps=self.config.layernorm_epsilon,
            sequence_parallel=self.config.sequence_parallel,
            fuse_wgrad_accumulation=self.config.gradient_accumulation_fusion,
            tp_group=get_tensor_model_parallel_group(check_initialized=False),
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

        world_size = get_tensor_model_parallel_world_size()
        rank = get_tensor_model_parallel_rank()

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
        tp_comm_buffer_name: str = None,
    ):
        if gather_output:
            raise ValueError('Transformer Engine linear layers do not support gather_output = True')

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
        )

        if config.use_cpu_initialization:
            if is_expert:
                world_size = get_expert_tensor_parallel_world_size()
                rank = get_expert_tensor_parallel_rank()
            else:
                world_size = get_tensor_model_parallel_world_size()
                rank = get_tensor_model_parallel_rank()
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
        tp_comm_buffer_name: str = None,
    ):
        if not input_is_parallel:
            raise ValueError(
                "Transformer Engine linear layers do not support input_is_parallel = False"
            )

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
            skip_weight_param_allocation=False,  # We don't currently use this for row parallel layers # pylint: disable=line-too-long
            is_expert=is_expert,
            tp_comm_buffer_name=tp_comm_buffer_name,
        )
        if config.use_cpu_initialization:
            if is_expert:
                world_size = get_expert_tensor_parallel_world_size()
                rank = get_expert_tensor_parallel_rank()
            else:
                world_size = get_tensor_model_parallel_world_size()
                rank = get_tensor_model_parallel_rank()
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
        attention_dropout: float = None,
        softmax_scale: float = None,
        k_channels: int = None,
        v_channels: int = None,
        cp_comm_type: str = "p2p",
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

        extra_kwargs = {}
        if is_te_min_version("0.11.0"):
            extra_kwargs["num_gqa_groups"] = self.config.num_query_groups
        elif self.config.num_query_groups != self.config.num_attention_heads:
            raise ValueError(
                f"Transformer Engine v{get_te_version()} does not support Grouped Query Attention, "
                f"use a newer version of Transformer Engine. "
                f"(num_query_groups ({self.config.num_query_groups}) != "
                f"num_attention_heads ({self.config.num_attention_heads}))"
            )

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
            extra_kwargs["cp_group"] = get_context_parallel_group(check_initialized=False)
            extra_kwargs["cp_global_ranks"] = get_context_parallel_global_ranks(
                check_initialized=False
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
            tp_group=get_tensor_model_parallel_group(check_initialized=False),
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
            dataclasses.asdict(packed_seq_params) if packed_seq_params is not None else {}
        )
        qkv_format = packed_seq_kwargs.get('qkv_format', self.qkv_format)

        if get_te_version() < PkgVersion("1.3.0"):
            # TE 1.3.0 introduces precomputing max_seqlen to remove unnecessary kernels and D2H
            # copies (#555)
            # These two arguments did not exist prior to 1.3.0
            packed_seq_kwargs.pop("max_seqlen_q", None)
            packed_seq_kwargs.pop("max_seqlen_kv", None)

        if get_te_version() < PkgVersion("1.10.0"):
            # TE 1.8.0 introduces cu_seqlens_padded which is the cu_seqlens with paddings counted
            # in each individual sequence in THD format dataset
            # These two arguments did not exist prior to 1.8.0.Full support added in 1.10.0 (#1012)
            packed_seq_kwargs.pop("cu_seqlens_q_padded", None)
            packed_seq_kwargs.pop("cu_seqlens_kv_padded", None)

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
            parallel_mode: str,
            config: ModelParallelConfig,
            init_method: Callable,
            bias: bool,
            skip_bias_add: bool,
            is_expert: bool = False,
            tp_comm_buffer_name: str = None,
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
            extra_kwargs["ub_name"] = tp_comm_buffer_name

            self.expert_parallel = self.config.expert_model_parallel_size > 1
            if is_expert:
                extra_kwargs["rng_tracker_name"] = get_expert_parallel_rng_tracker_name()

            # The comms between TP and EP group is explicitly handled by MoE token dispatcher.
            # So we disable comms by making TE agnostic of model parallel.
            if is_expert:
                tp_group = get_expert_tensor_parallel_group(check_initialized=False)
                tp_size = get_expert_tensor_parallel_world_size()
            else:
                tp_group = get_tensor_model_parallel_group(check_initialized=False)
                tp_size = get_tensor_model_parallel_world_size()
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
                tp_group=tp_group,
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
                fp8_checkpoint = self.fp8_meta["fp8_checkpoint"] or self.fp8 or self.fp8_calibration

                try:
                    state_list = [
                        state_dict.pop(f"{prefix}_extra_state{i}") for i in range(1, self.num_gemms)
                    ]
                except KeyError:
                    # "_extra_state{i}" only exists for dist-ckpt. Return for torch native ckpt.
                    return

                if not fp8_checkpoint:
                    return
                state_list = [state_dict.pop(f"{prefix}_extra_state")] + state_list
                state_list = [self._decode_extra_state(state) for state in state_list]
                extra_fp8_variables = state_list[0]['extra_fp8_variables']
                extra_fp8_variables['num_gemms'] = self.num_gemms
                extra_state = {
                    "scale_fwd": torch.cat(
                        [state['scale_fwd'].view(-1, 1) for state in state_list], dim=1
                    ).view(-1),
                    "scale_inv_fwd": torch.cat(
                        [state['scale_inv_fwd'].view(-1, 1) for state in state_list], dim=1
                    ).view(-1),
                    "amax_history_fwd": torch.cat(
                        [state['amax_history_fwd'].view(-1, 1) for state in state_list], dim=1
                    ).view(self.fp8_meta["recipe"].amax_history_len, -1),
                    "scale_bwd": torch.cat(
                        [state['scale_bwd'].view(-1, 1) for state in state_list], dim=1
                    ).view(-1),
                    "scale_inv_bwd": torch.cat(
                        [state['scale_inv_bwd'].view(-1, 1) for state in state_list], dim=1
                    ).view(-1),
                    "amax_history_bwd": torch.cat(
                        [state['amax_history_bwd'].view(-1, 1) for state in state_list], dim=1
                    ).view(self.fp8_meta["recipe"].amax_history_len, -1),
                    "extra_fp8_variables": extra_fp8_variables,
                }
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
            state_serialized = io.BytesIO()
            torch.save(state, state_serialized)
            return state_serialized

        def _decode_extra_state(self, state):
            if isinstance(state, torch.Tensor):
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
                tmp_state = {
                    "scale_fwd": state['scale_fwd'].view(3, -1)[:, gemm_idx],
                    "scale_inv_fwd": state['scale_inv_fwd'].view(3, -1)[:, gemm_idx],
                    "amax_history_fwd": state['amax_history_fwd'].view(
                        self.fp8_meta["recipe"].amax_history_len, 3, -1
                    )[:, :, gemm_idx],
                    "scale_bwd": state['scale_bwd'].view(2, -1)[:, gemm_idx],
                    "scale_inv_bwd": state['scale_inv_bwd'].view(2, -1)[:, gemm_idx],
                    "amax_history_bwd": state['amax_history_bwd'].view(
                        self.fp8_meta["recipe"].amax_history_len, 2, -1
                    )[:, :, gemm_idx],
                    "extra_fp8_variables": extra_fp8_variables,
                }
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
                sh_ten.replica_id = (*replica_id[:2], get_expert_data_parallel_rank())
            return sharded_state_dict

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
            tp_comm_buffer_name: str = None,
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
            tp_comm_buffer_name: str = None,
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

    TEGroupedLinear = None
    TEColumnParallelGroupedLinear = None
    TERowParallelGroupedLinear = None


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

    def is_initialized(self):
        """Checks if the internal RNG state has been set wirth set_states()."""
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
    forward_func,
    distribute_saved_activations,
    get_rng_state_tracker,
    tp_group,
    hidden_states,
    attention_mask,
    context,
    context_mask,
    rotary_pos_emb,
):
    """Checkpointing with Transformer-Engine."""
    from transformer_engine.pytorch.distributed import checkpoint

    if is_te_min_version("1.5.0"):
        return checkpoint(
            forward_func,
            hidden_states,
            attention_mask,
            context,
            context_mask,
            rotary_pos_emb,
            distribute_saved_activations=distribute_saved_activations,
            get_rng_state_tracker=get_rng_state_tracker,
            tp_group=tp_group,
        )
    else:
        return checkpoint(
            forward_func,
            distribute_saved_activations,
            get_rng_state_tracker,
            tp_group,
            hidden_states,
            attention_mask,
            context,
            context_mask,
            rotary_pos_emb,
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

    get_cpu_offload_context = None

try:

    from transformer_engine.pytorch.attention import FusedRoPEFunc

    def fused_apply_rotary_pos_emb(t: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
        """Apply rotary positional embedding to input tensor T in `sbhd` format."""
        return FusedRoPEFunc.apply(t, freqs, "sbhd")

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
        if is_te_min_version("1.11.0", check_equality=False):
            return FusedRoPEFunc.apply(t, freqs, "thd", cu_seqlens, cp_size, cp_rank)
        else:
            return FusedRoPEFunc.apply(t, freqs, "thd", cu_seqlens)

except ImportError:

    pass

try:

    from transformer_engine.pytorch import Fp8Padding, Fp8Unpadding  # pylint: disable=unused-import

except ImportError:

    Fp8Padding = None
    Fp8Unpadding = None
