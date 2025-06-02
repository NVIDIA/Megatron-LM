# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
import warnings
import torch
from torch import Tensor

from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.transformer.utils import get_linear_layer

try:
    import apex  # pylint: disable=unused-import

    from megatron.core.fusions.fused_layer_norm import FusedLayerNorm

    HAVE_APEX = True
    LNImpl = FusedLayerNorm
except ImportError:

    from megatron.core.transformer.torch_norm import WrappedTorchNorm

    warnings.warn('Apex is not installed. Falling back to Torch Norm')
    LNImpl = WrappedTorchNorm


class BertLMHead(MegatronModule):
    """Masked LM head for Bert.

    Args:
        hidden_size: hidden size
        config (TransformerConfig): TransformerConfig object
    """

    def __init__(self, hidden_size: int, config: TransformerConfig):
        super().__init__(config=config)

        # TODO: Should switch this to TE ?
        self.dense = get_linear_layer(
            hidden_size, hidden_size, config.init_method, config.perform_initialization
        )

        setattr(self.dense.weight, 'sequence_parallel', config.sequence_parallel)
        setattr(self.dense.bias, 'sequence_parallel', config.sequence_parallel)

        self.layer_norm = LNImpl(
            config=config, hidden_size=hidden_size, eps=config.layernorm_epsilon
        )

        self.gelu = torch.nn.functional.gelu

    def forward(self, hidden_states: Tensor) -> Tensor:
        """forward pass"""

        hidden_states = self.dense(hidden_states)
        hidden_states = self.gelu(hidden_states)
        hidden_states = self.layer_norm(hidden_states)
        return hidden_states
