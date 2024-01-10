import os
from importlib.metadata import version
from typing import Callable

import torch
import transformer_engine as te
from pkg_resources import packaging
from torch import Tensor

from megatron.core import ModelParallelConfig
from megatron.core.parallel_state import (
    get_context_parallel_global_ranks,
    get_context_parallel_group,
    get_tensor_model_parallel_group,
)
from megatron.core.tensor_parallel import get_cuda_rng_tracker
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.transformer.utils import make_sharded_tensors_for_checkpoint


def _get_extra_te_kwargs(config: TransformerConfig):
    extra_transformer_engine_kwargs = {
        "params_dtype": config.params_dtype,
    }

    te_version = packaging.version.Version(version("transformer-engine"))
    if te_version >= packaging.version.Version("0.12.0"):
        if config.use_cpu_initialization:
            extra_transformer_engine_kwargs["device"] = 'cpu'
        else:
            extra_transformer_engine_kwargs["device"] = torch.cuda.current_device()
    return extra_transformer_engine_kwargs


class TENorm:
    """
    A conditional wrapper to initialize an instance of Transformer-Engine's
    `LayerNorm` or `RMSNorm` based on input
    """

    # TODO should we ditch normalization config and just use spec to choose LayerNorm vs RMSNorm?
    def __new__(
        cls, config: TransformerConfig, hidden_size: int, eps: float = 1e-5,
    ):
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
    ):
        self.config = config

        # TE returns a zero length Tensor when bias=False and
        # return_bias=True, but we prefer None.  So in that case we
        # tell TE to not return the bias, and return None
        # ourselves. This way our forward always returns two values
        # and we don't have to deal with the zero length Tensor.
        self.te_return_bias = skip_bias_add and bias
        self.is_first_microbatch = True
        if skip_weight_param_allocation:
            raise ValueError(
                'Transformer Engine linear layers do not support skip_weight_param_allocation'
            )

        extra_kwargs = _get_extra_te_kwargs(config)

        te_version = packaging.version.Version(version("transformer-engine"))
        if te_version >= packaging.version.Version("0.8.0"):
            if self.config.tp_comm_overlap:
                extra_kwargs["ub_split_ag"] = self.config.tp_comm_split_ag
                extra_kwargs["ub_split_rs"] = self.config.tp_comm_split_rs
                if te_version > packaging.version.Version("1.0.0"):
                    assert (
                        tp_comm_buffer_name is not None
                    ), "Buffer name should be set to configure communication overlap settings"
                    extra_kwargs["ub_name"] = tp_comm_buffer_name

        super().__init__(
            in_features=input_size,
            out_features=output_size,
            sequence_parallel=self.config.sequence_parallel,
            fuse_wgrad_accumulation=self.config.gradient_accumulation_fusion,
            tp_group=get_tensor_model_parallel_group(check_initialized=False),
            tp_size=self.config.tensor_model_parallel_size,
            get_rng_state_tracker=get_cuda_rng_tracker,
            init_method=init_method,
            bias=bias,
            return_bias=self.te_return_bias,
            parallel_mode=parallel_mode,
            **extra_kwargs,
        )

    def forward(self, x):
        out = super().forward(x, self.is_first_microbatch)
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
        extra_kwargs = _get_extra_te_kwargs(config)

        # Only Transformer-Engine version >= 0.11.0 supports `RMSNorm`
        te_version = packaging.version.Version(version("transformer-engine"))
        if te_version >= packaging.version.Version("0.11.0"):
            extra_kwargs["normalization"] = self.config.normalization
        elif self.config.normalization != "LayerNorm":
            raise ValueError(
                f"Transformer Engine v{te_version} does not support {self.config.normalization}."
            )

        if te_version >= packaging.version.Version("0.8.0"):
            if self.config.tp_comm_overlap:
                extra_kwargs["ub_bulk_wgrad"] = self.config.tp_comm_bulk_wgrad
                extra_kwargs["ub_bulk_dgrad"] = self.config.tp_comm_bulk_dgrad
                extra_kwargs["ub_split_ag"] = self.config.tp_comm_split_ag
                if te_version > packaging.version.Version("1.0.0"):
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
            get_rng_state_tracker=get_cuda_rng_tracker,
            init_method=init_method,
            bias=bias,
            return_bias=self.te_return_bias,
            parallel_mode="column",
            return_layernorm_output=False,
            zero_centered_gamma=self.config.layernorm_zero_centered_gamma,
            **extra_kwargs,
        )

    def forward(self, x):
        out = super().forward(x, self.is_first_microbatch)
        self.is_first_microbatch = False

        # TE only returns a tuple when return_bias is True, otherwise
        # it returns a single Tensor, we always want to return two
        # values regardless of the arguments.
        if self.te_return_bias:
            return out
        return out, None

    def sharded_state_dict(self, prefix='', sharded_key_prefix=None, sharded_offsets=()):
        """ Sharding along axis 0, bias sharded """
        state_dict = self.state_dict(prefix='', keep_vars=True)
        return make_sharded_tensors_for_checkpoint(
            state_dict, prefix, sharded_key_prefix, {'weight': 0, 'bias': 0}, sharded_offsets
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

        if is_expert:
            raise ValueError('Transformer Engine linear layers do not yet support MoE')

        super().__init__(
            input_size=input_size,
            output_size=output_size,
            parallel_mode="column",
            config=config,
            init_method=init_method,
            bias=bias,
            skip_bias_add=skip_bias_add,
            skip_weight_param_allocation=skip_weight_param_allocation,
            tp_comm_buffer_name=tp_comm_buffer_name,
        )

    def sharded_state_dict(self, prefix='', sharded_key_prefix=None, sharded_offsets=()):
        """ Sharding along axis 0, bias sharded """
        state_dict = self.state_dict(prefix='', keep_vars=True)
        return make_sharded_tensors_for_checkpoint(
            state_dict, prefix, sharded_key_prefix, {'weight': 0, 'bias': 0}, sharded_offsets
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

        if is_expert:
            raise ValueError('Transformer Engine linear layers do not yet support MoE')

        super().__init__(
            input_size=input_size,
            output_size=output_size,
            parallel_mode="row",
            config=config,
            init_method=init_method,
            bias=bias,
            skip_bias_add=skip_bias_add,
            skip_weight_param_allocation=False,  # We don't currently use this for row parallel layers
            tp_comm_buffer_name=tp_comm_buffer_name,
        )

    def sharded_state_dict(self, prefix='', sharded_key_prefix=None, sharded_offsets=()):
        """ Sharding along axis 1, bias not sharded """
        state_dict = self.state_dict(prefix='', keep_vars=True)
        return make_sharded_tensors_for_checkpoint(
            state_dict, prefix, sharded_key_prefix, {'weight': 1}, sharded_offsets
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
    ):
        self.config = config
        self.te_forward_mask_type = False

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
        te_version = packaging.version.Version(version("transformer-engine"))
        if te_version >= packaging.version.Version("0.11.0"):
            extra_kwargs["num_gqa_groups"] = self.config.num_query_groups
        elif self.config.num_query_groups != self.config.num_attention_heads:
            raise ValueError(
                f"Transformer Engine v{te_version} does not support Grouped Query Attention, "
                f"use a newer version of Transformer Engine. "
                f"(num_query_groups ({self.config.num_query_groups}) != "
                f"num_attention_heads ({self.config.num_attention_heads}))"
            )

        if te_version >= packaging.version.Version("0.10.0"):
            extra_kwargs["attention_type"] = attention_type
            # older version don't need attention_type

        if te_version > packaging.version.Version("0.12.0"):
            self.te_forward_mask_type = True

        # Only Transformer-Engine version >= 1.0.0 supports context parallelism
        if te_version >= packaging.version.Version("1.0.0"):
            if getattr(TEDotProductAttention, "cp_stream") is None:
                TEDotProductAttention.cp_stream = torch.cuda.Stream()
            extra_kwargs["cp_group"] = get_context_parallel_group(check_initialized=False)
            extra_kwargs["cp_global_ranks"] = get_context_parallel_global_ranks(
                check_initialized=False
            )
            extra_kwargs["cp_stream"] = TEDotProductAttention.cp_stream
        else:
            assert (
                self.config.context_parallel_size == 1
            ), "Only Transformer-Engine version >= 1.0.0 supports context parallelism!"

        if config.window_size is not None:
            # Check version
            assert te_version >= packaging.version.Version(
                "1.2.0"
            ), f"Transformer-Engine version ({str(te_version)}) must be >= 1.2.0 to support sliding window attention."
            extra_kwargs['window_size'] = config.window_size

        super().__init__(
            num_attention_heads=self.config.num_attention_heads,
            kv_channels=self.config.kv_channels,
            attention_dropout=self.config.attention_dropout
            if attention_dropout is None
            else attention_dropout,
            attn_mask_type=attn_mask_type.name,
            sequence_parallel=self.config.sequence_parallel,
            tp_size=self.config.tensor_model_parallel_size,
            get_rng_state_tracker=get_cuda_rng_tracker,
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
    ):
        if self.te_forward_mask_type:
            return super().forward(
                query, key, value, attention_mask, attn_mask_type=attn_mask_type.name
            )
        else:
            return super().forward(query, key, value, attention_mask)


try:

    from transformer_engine.pytorch.attention import _SplitAlongDim

    SplitAlongDim = _SplitAlongDim.apply

except ImportError:

    SplitAlongDim = None
