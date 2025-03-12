# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.

import logging
import math
from copy import deepcopy
from functools import partial
from typing import Tuple, Union

import torch

from megatron.core.extensions.transformer_engine import (
    TEColumnParallelLinear,
    TELayerNormColumnParallelLinear,
    TERowParallelLinear,
)
from megatron.core.tensor_parallel import (
    ColumnParallelLinear,
    RowParallelLinear,
)
from megatron.core.transformer import TransformerConfig
from megatron.core.transformer.custom_layers.synced_linear import SyncedLinear
from megatron.core.transformer.module import MegatronModule


LOGGER = logging.getLogger(__name__)
KAIMING_INIT_METHOD: callable = lambda x: torch.nn.init.kaiming_uniform_(x, a=math.sqrt(5))
LORA_LAYERS_DEFAULT_CONFIG = {
    "bias": False,
    "skip_bias_add": True,
}
COLUMN_PARALLEL_LAYERS = [
    partial(SyncedLinear, init_method=KAIMING_INIT_METHOD),
    partial(ColumnParallelLinear, **LORA_LAYERS_DEFAULT_CONFIG, init_method=torch.nn.init.zeros_),
]
ROW_PARALLEL_LAYERS = [
    partial(RowParallelLinear, **LORA_LAYERS_DEFAULT_CONFIG, init_method=KAIMING_INIT_METHOD, input_is_parallel=True),
    partial(SyncedLinear, init_method=torch.nn.init.zeros_, broadcast_weights=False),
]
TE_COLUMN_PARALLEL_LAYERS = [
    partial(SyncedLinear, init_method=KAIMING_INIT_METHOD),
    partial(TEColumnParallelLinear, **LORA_LAYERS_DEFAULT_CONFIG, init_method=torch.nn.init.zeros_, gather_output=False),
]
TE_ROW_PARALLEL_LAYERS = [
    partial(TERowParallelLinear, **LORA_LAYERS_DEFAULT_CONFIG, init_method=KAIMING_INIT_METHOD, input_is_parallel=True),
    partial(SyncedLinear, init_method=torch.nn.init.zeros_, broadcast_weights=False),
]
LORA_LAYERS_MAPPING = {
    ColumnParallelLinear: COLUMN_PARALLEL_LAYERS,
    RowParallelLinear: ROW_PARALLEL_LAYERS,
    TEColumnParallelLinear: TE_COLUMN_PARALLEL_LAYERS,
    TELayerNormColumnParallelLinear: TE_COLUMN_PARALLEL_LAYERS,
    TERowParallelLinear: TE_ROW_PARALLEL_LAYERS,
}


class LoraAdapter(MegatronModule):
    def __init__(self, base_layer: torch.nn.Module, *, config: TransformerConfig, rank: int, alpha: float, dropout: float, is_expert: bool = False):
        super(LoraAdapter, self).__init__(config)

        if config.sequence_parallel and torch.distributed.get_rank() == 0:
            LOGGER.warning("Sequence parallelism is not fully supported and may slow down the training. Use it at your own risk.")

        self.lora_alpha = alpha
        self.base_layer = base_layer
        self.base_layer.weight.requires_grad = False
        self.lora_a = None
        self.lora_b = None
        self.register_load_state_dict_pre_hook(self._remap_base_layer_for_training)
        self.register_load_state_dict_post_hook(self._ignore_missing_lora_keys_for_training)

        base_layer_class = type(base_layer)
        if base_layer_class not in LORA_LAYERS_MAPPING:
            if torch.distributed.get_rank() == 0:
                LOGGER.warning(f"LoRA is not supported for {base_layer_class}. Freezing weights of {base_layer_class} but skipping addition of LoRA layers")
            return
        
        layer_config = {
            "config": config,
            "is_expert": is_expert,
        }
        output_size, input_size = self.base_layer.weight.shape
        if base_layer_class in [RowParallelLinear, TERowParallelLinear]:
            input_size *= config.tensor_model_parallel_size
        if base_layer_class in [ColumnParallelLinear, TEColumnParallelLinear, TELayerNormColumnParallelLinear]:
            output_size *= config.tensor_model_parallel_size
        lora_a_class, lora_b_class = LORA_LAYERS_MAPPING[base_layer_class]
        self.lora_a = lora_a_class(input_size=input_size, output_size=rank, **layer_config)
        self.lora_b = lora_b_class(input_size=rank, output_size=output_size, **layer_config)
        self.lora_dropout = torch.nn.Dropout(p=dropout, inplace=False)

    def _remap_base_layer_for_training(self, _: torch.nn.Module, state_dict: dict, prefix: str, *args) -> None:
        extra_prefix = "base_layer."
        keys = list(state_dict.keys())
        for key in keys:
            # The model is already finetuned with LoRA
            if extra_prefix in key or "lora_" in key:
                continue
            
            # The model has no adapter layers
            new_key = key.replace(prefix, f"{prefix}{extra_prefix}")
            state_dict[new_key] = state_dict.pop(key)

    def _ignore_missing_lora_keys_for_training(self, _: torch.nn.Module, incompatible_keys: torch.nn.modules.module._IncompatibleKeys) -> None:
        keys = deepcopy(incompatible_keys.missing_keys)
        for key in keys:
            if "lora_" in key:
                incompatible_keys.missing_keys.remove(key)

    def forward(self, input: torch.Tensor, *args, **kwargs) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        output = self.base_layer(input, *args, **kwargs)
        if self.lora_a is None:
            return output

        lora_a_output_parallel, _ = self.lora_a(input)
        lora_b_output_parallel, _ = self.lora_b(lora_a_output_parallel)
        lora_dropout_output_parallel = self.lora_dropout(lora_b_output_parallel)
        lora_output_parallel = self.lora_alpha * lora_dropout_output_parallel

        if type(output) is torch.Tensor:
            return output + lora_output_parallel
        
        output, bias = output
        return output + lora_output_parallel, bias
