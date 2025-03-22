# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

from typing import Callable, List, Optional

import torch
import transformer_engine as te

from megatron.core.extensions.transformer_engine import _get_extra_te_kwargs
from megatron.core.model_parallel_config import ModelParallelConfig
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.transformer.transformer_layer import TransformerLayer
from megatron.core.transformer.utils import make_sharded_tensors_for_checkpoint

try:
    import modelopt.torch.quantization as mtq
    from modelopt.torch.quantization.nn import QuantModuleRegistry
    from modelopt.torch.quantization.nn.modules.quant_linear import _QuantLinear

    has_nvidia_modelopt = True
except Exception:
    has_nvidia_modelopt = False


FP8_PER_TENSOR_REAL_QUANT_CFG = {
    "quant_cfg": {
        "*weight_quantizer": {"num_bits": (4, 3), "axis": None, "fake_quant": False},
        "*input_quantizer": {"enable": False},
        "*output_layer*": {"enable": False},
        "default": {"enable": False},
    },
    "algorithm": "max",
}

# FP8 2D blockwise real quantization config for deepseek models
FP8_2D_BLOCKWISE_REAL_QUANT_CFG = {
    "quant_cfg": {
        "*weight_quantizer": {
            "num_bits": (4, 3),
            "block_sizes": {-1: 128, -2: 128},
            "fake_quant": False,
        },
        "*input_quantizer": {"enable": False},
        "*output_layer*": {"enable": False},
        "default": {"enable": False},
    },
    "algorithm": "max",
}


class Norm:
    """
    A conditional wrapper to initialize an instance of Transformer-Engine's
    `LayerNorm` or `RMSNorm` based on input. If there is an additional _extra_state,
    insert _state_dict_hook and _load_state_dict_pre_hook to handle the state_dict
    mismatch issue.
    """

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

        def _state_dict_hook(self, state_dict, prefix, local_metadata):
            if "_extra_state" in state_dict:
                state_dict.pop("_extra_state")

        def _load_state_dict_pre_hook(
            state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
        ):
            state_dict[prefix + "_extra_state"] = None

        if "_extra_state" in instance.state_dict():
            instance._register_state_dict_hook(_state_dict_hook)
            instance._register_load_state_dict_pre_hook(_load_state_dict_pre_hook)

        return instance


class Linear(torch.nn.Linear):
    """Local Linear impl as a replacement of ParallelLinear."""

    def __init__(
        self,
        input_size: int,
        output_size: int,
        *,
        config: ModelParallelConfig,
        init_method: Callable,
        bias: bool = True,
        gather_output: bool = False,
        stride: int = 1,
        keep_master_weight_for_test: bool = False,
        skip_bias_add: bool = False,
        skip_weight_param_allocation: bool = False,
        embedding_activation_buffer: Optional[List[torch.Tensor]] = None,
        grad_output_buffer: Optional[List[torch.Tensor]] = None,
        is_expert: bool = False,
        tp_comm_buffer_name: str = None,  # Not used
        disable_grad_reduce: bool = False,
    ):
        self.config = config

        self._return_bias = skip_bias_add and bias

        if stride != 1:
            raise ValueError('torch.nn.Linear does not support stride != 1')

        if skip_weight_param_allocation:
            raise ValueError('torch.nn.Linear layers do not support skip_weight_param_allocation')

        if embedding_activation_buffer is not None:
            raise ValueError('torch.nn.Linear does not support embedding_activation_buffer != None')

        if grad_output_buffer is not None:
            raise ValueError('torch.nn.Linear does not support grad_output_buffer != None')

        super().__init__(
            in_features=input_size, out_features=output_size, bias=bias, dtype=config.params_dtype
        )

        for param in self.parameters():
            if is_expert:
                # Reduce the gradient on the expert_data_parallel group for expert linear layers
                setattr(param, 'allreduce', self.config.expert_model_parallel_size == 1)
            else:
                # Reduce the gradient on DP group
                setattr(param, 'allreduce', True)
                setattr(param, 'sequence_parallel', self.config.sequence_parallel)

    def sharded_state_dict(self, prefix='', sharded_offsets=(), metadata=None):
        """Sharding along axis 0, bias sharded"""
        state_dict = self.state_dict(prefix='', keep_vars=True)

        for k, v in state_dict.items():
            if "_amax" in k or "_scale" in k:
                if v.ndim == 0:
                    state_dict[k] = v.view(1)
        sharded_state_dict = make_sharded_tensors_for_checkpoint(
            state_dict, prefix, sharded_offsets=sharded_offsets
        )
        return sharded_state_dict

    def forward(self, x):
        """Forward."""
        out = super().forward(x)

        if self._return_bias:
            return out
        return out, None


if has_nvidia_modelopt:
    QuantModuleRegistry.register({Linear: Linear.__class__.__name__})(_QuantLinear)


class RealQuantTransformerLayer(TransformerLayer):
    """Real quantization transformer layer base class.

    This base class iniitialize the default TransformerLayer and immediately
    perform weight-only real quantization via TensorRT Model Optimizer.
    All linear weights (Linear, ColumnParallelLinear, RowParallelLinear) picked
    up will be replaced with low-bit data type (default torch.uint8). If sub-byte
    real_quant_cfg is used, the weight shape will further be half.

    This module cannot be trained (all parameters frozen).
    """

    verbose: bool = False
    real_quant_cfg: str = "None"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if has_nvidia_modelopt and self.real_quant_cfg != "None":

            REAL_QUANT_CFG_CHOICES = {
                "fp8_real_quant": FP8_PER_TENSOR_REAL_QUANT_CFG,
                "fp8_blockwise_real_quant": FP8_2D_BLOCKWISE_REAL_QUANT_CFG,
            }
            mtq_cfg = REAL_QUANT_CFG_CHOICES.get(self.real_quant_cfg, None)
            if mtq_cfg is None:
                raise ValueError(
                    "RealQuantTransformerLayer does not support {}".format(self.real_quant_cfg)
                )

            self._collect_original_tensor_info()

            mtq.quantize(self, mtq_cfg)

            delattr(self, "_modelopt_state")

            # Freeze all parameters since the real-quant linears cannot be trained.
            for param in self.parameters():
                param.requires_grad = False

            if self.verbose:
                self._report_quantize_tensor_info()

    def _collect_original_tensor_info(self):
        self._original_tensor_info = {}
        for k, v in self.state_dict().items():
            if isinstance(v, torch.Tensor):
                self._original_tensor_info[k] = (str(v.dtype), str(v.shape))

    def _report_quantize_tensor_info(self):
        torch.distributed.barrier()
        if torch.distributed.get_rank() == 0:
            for k, v in self.state_dict().items():
                if not isinstance(v, torch.Tensor):
                    continue
                original_dtype, original_shape = self._original_tensor_info.get(k, ("-", "-"))
                print(
                    "{:<64} {:<16} {:<32} {:<16} {:<32}".format(
                        k, original_dtype, original_shape, str(v.dtype), str(v.shape)
                    )
                )
        torch.distributed.barrier()


class FP8WeightTransformerLayer(RealQuantTransformerLayer):
    """FP8 weight transformer layer."""

    real_quant_cfg: str = "fp8_real_quant"


class BlockwiseFP8WeightTransformerLayer(RealQuantTransformerLayer):
    """Blockwise FP8 weight transformer layer."""

    real_quant_cfg: str = "fp8_blockwise_real_quant"
