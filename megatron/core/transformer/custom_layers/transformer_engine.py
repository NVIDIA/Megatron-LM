import os
from importlib.metadata import version
from typing import Callable

import torch
import transformer_engine as te
from pkg_resources import packaging

from megatron.core.parallel_state import (
    get_context_parallel_global_ranks,
    get_context_parallel_group,
    get_tensor_model_parallel_group,
)
from megatron.core.tensor_parallel import get_cuda_rng_tracker
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.transformer_config import TransformerConfig


def _get_extra_te_kwargs(config: TransformerConfig):
    extra_transformer_engine_kwargs = {}
    from importlib.metadata import version

    from pkg_resources import packaging

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

    def __new__(
        cls,
        config: TransformerConfig,
        hidden_size: int,
        eps: float = 1e-5,
        sequence_parallel: bool = False,
        normalization: str = "LayerNorm",
        **kwargs
    ):
        if normalization == "LayerNorm":
            instance = te.pytorch.LayerNorm(
                hidden_size=hidden_size,
                eps=eps,
                sequence_parallel=sequence_parallel,
                **_get_extra_te_kwargs(config),
            )
        elif normalization == "RMSNorm":
            assert hasattr(
                te.pytorch, "RMSNorm"
            ), "Transformer-Engine >= v0.11 required to use this feature"
            instance = te.pytorch.RMSNorm(
                hidden_size=hidden_size,
                eps=eps,
                sequence_parallel=sequence_parallel,
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
        config: TransformerConfig,
        parallel_mode: str,
        init_method: Callable,
        *,
        bias: bool = True,
        skip_bias_add: bool = False,
        **kwargs
    ):
        self.config = config

        # TE returns a zero length Tensor when bias=False and
        # return_bias=True, but we prefer None.  So in that case we
        # tell TE to not return the bias, and return None
        # ourselves. This way our forward always returns two values
        # and we don't have to deal with the zero length Tensor.
        self.te_return_bias = skip_bias_add and bias

        extra_kwargs = _get_extra_te_kwargs(config)

        te_version = packaging.version.Version(version("transformer-engine"))
        if te_version >= packaging.version.Version("0.8.0"):
            extra_kwargs["ub_split_ag"] = self.config.tp_comm_overlap and self.config.tp_comm_split_ag
            )
            extra_kwargs["ub_split_rs"] = self.config.tp_comm_overlap and self.config.tp_comm_split_rs
            )

        super().__init__(
            in_features=input_size,
            out_features=output_size,
            sequence_parallel=self.config.sequence_parallel,
            fuse_wgrad_accumulation=self.config.gradient_accumulation_fusion,
            tp_group=get_tensor_model_parallel_group(check_initialized=False),
            tp_size=self.config.tensor_model_parallel_size,
            get_rng_state_tracker=get_cuda_rng_tracker,
            init_method=init_method,
            params_dtype=self.config.params_dtype,
            parallel_mode=parallel_mode,
            bias=bias,
            return_bias=self.te_return_bias,
            **extra_kwargs,
        )

    def forward(self, x):
        out = super().forward(x)

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
        config: TransformerConfig,
        init_method: Callable,
        bias: bool,
        skip_bias_add: bool,
        **kwargs
    ):
        self.config = config
        # TE returns a zero length Tensor when bias=False and
        # return_bias=True, but we prefer None.  So in that case we
        # tell TE to not return the bias, and return None
        # ourselves. This way our forward always returns two values
        # and we don't have to deal with the zero length Tensor.
        self.te_return_bias = skip_bias_add and bias

        extra_kwargs = _get_extra_te_kwargs(config)

        # Only Transformer-Engine version >= 0.11.0 supports `RMSNorm`
        te_version = packaging.version.Version(version("transformer-engine"))
        if te_version >= packaging.version.Version("0.11.0"):
            kwargs["normalization"] = self.config.normalization

        if te_version >= packaging.version.Version("0.8.0"):
            extra_kwargs["ub_bulk_wgrad"] = self.config.tp_comm_overlap and self.config.tp_comm_bulk_wgrad
            )
            extra_kwargs["ub_bulk_dgrad"] = self.config.tp_comm_overlap and self.config.tp_comm_bulk_dgrad
            )
            extra_kwargs["ub_split_ag"] = self.config.tp_comm_overlap and self.config.tp_comm_split_ag
            )

        super().__init__(
            in_features=input_size,
            out_features=output_size,
            bias=bias,
            sequence_parallel=self.config.sequence_parallel,
            fuse_wgrad_accumulation=self.config.gradient_accumulation_fusion,
            tp_group=get_tensor_model_parallel_group(check_initialized=False),
            tp_size=self.config.tensor_model_parallel_size,
            get_rng_state_tracker=get_cuda_rng_tracker,
            init_method=init_method,
            params_dtype=self.config.params_dtype,
            parallel_mode="column",
            return_bias=self.te_return_bias,
            **extra_kwargs,
        )

    def forward(self, x):
        out = super().forward(x)

        # TE only returns a tuple when return_bias is True, otherwise
        # it returns a single Tensor, we always want to return two
        # values regardless of the arguments.
        if self.te_return_bias:
            return out
        return out, None


class TEColumnParallelLinear(TELinear):
    """
    Wrapper for the Transformer-Engine's `Linear` layer but specialized similar
    to megatron's `ColumnParallelLinear` layer.
    """

    def __init__(self, input_size: int, output_size: int, config: TransformerConfig, **kwargs):
        self.config = config
        super().__init__(
            input_size=input_size,
            output_size=output_size,
            config=self.config,
            parallel_mode="column",
            **kwargs,
        )


class TERowParallelLinear(TELinear):
    """
    Wrapper for the Transformer-Engine's `Linear` layer but specialized similar
    to megatron's `RowParallelLinear` layer.
    """

    def __init__(self, input_size: int, output_size: int, config: TransformerConfig, **kwargs):
        self.config = config
        super().__init__(
            input_size=input_size,
            output_size=output_size,
            config=self.config,
            parallel_mode="row",
            **kwargs,
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
        layer_number: int = 1,
        attn_mask_type: AttnMaskType = AttnMaskType.padding,
        **kwargs
    ):
        self.config = config

        # Only Transformer-Engine version > 0.13.0 supports context parallelism
        te_version = packaging.version.Version(version("transformer-engine"))
        if te_version > packaging.version.Version("0.13.0"):
            if getattr(TEDotProductAttention, "cp_stream") is None:
                TEDotProductAttention.cp_stream = torch.cuda.Stream()
            kwargs["cp_group"] = get_context_parallel_group(check_initialized=False)
            kwargs["cp_global_ranks"] = get_context_parallel_global_ranks(check_initialized=False)
            kwargs["cp_stream"] = TEDotProductAttention.cp_stream
        else:
            assert (
                self.config.context_parallel_size == 1
            ), "Only Transformer-Engine version > 0.13.0 supports context parallelism"

        super().__init__(
            num_attention_heads=self.config.num_attention_heads,
            kv_channels=self.config.kv_channels,
            attention_dropout=self.config.attention_dropout,
            layer_number=layer_number,
            attn_mask_type=attn_mask_type.name,
            sequence_parallel=self.config.sequence_parallel,
            tp_size=self.config.tensor_model_parallel_size,
            get_rng_state_tracker=get_cuda_rng_tracker,
            tp_group=get_tensor_model_parallel_group(check_initialized=False),
            **kwargs,
        )


class TELayerNormMLP(te.pytorch.LayerNormMLP):
    """
    Wrapper for the Transformer-Engine's `LayerNormMLP` layer that combines
    `LayerNorm` and the MLP (2 x feedforward layers) into a single module which
    is performance-efficient as it removes the unnecessary FP8 -> FP32 casts.
    """

    def __init__(self, config: TransformerConfig, **kwargs):
        self.config = config

        # Only Transformer-Engine version >= 0.11.0 supports `RMSNorm`
        te_version = packaging.version.Version(version("transformer-engine"))
        if te_version >= packaging.version.Version("0.11.0"):
            kwargs["normalization"] = self.config.normalization

        super().__init__(
            self.config.hidden_size,
            self.config.ffn_hidden_size,
            self.config.layernorm_epsilon,
            fuse_wgrad_accumulation=self.config.gradient_accumulation_fusion,
            tp_group=get_tensor_model_parallel_group(check_initialized=False),
            tp_size=self.config.tensor_model_parallel_size,
            get_rng_state_tracker=get_cuda_rng_tracker,
            init_method=self.config.init_method,
            params_dtype=self.config.params_dtype,
            return_bias=not self.config.add_bias_linear,
        )

    def forward(self, x):
        out = super().forward(x)

        # TE only returns a tuple when return_bias is True, otherwise
        # it returns a single Tensor, we always want to return two
        # values regardless of the arguments.
        if isinstance(out, (list, tuple)):
            return out
        return out, None
