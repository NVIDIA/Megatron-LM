# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
from typing import Optional

import torch
from torch import Tensor

from megatron.core.models.backends import LocalSpecProvider
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.transformer.utils import get_linear_layer
from megatron.core.utils import get_tensor_model_parallel_group_if_none


class BertLMHead(MegatronModule):
    """Masked LM head for Bert.

    Args:
        hidden_size: hidden size
        config (TransformerConfig): TransformerConfig object
        tp_group (torch.distributed.ProcessGroup, optional): Tensor-parallel group the head's
            weights are replicated across, which sets their checkpoint replica ids. Defaults to
            the global tensor-parallel group.
    """

    def __init__(
        self,
        hidden_size: int,
        config: TransformerConfig,
        tp_group: Optional[torch.distributed.ProcessGroup] = None,
    ):
        super().__init__(config=config)
        self.tp_group = get_tensor_model_parallel_group_if_none(tp_group)

        # TODO: Should switch this to TE ?
        self.dense = get_linear_layer(
            hidden_size, hidden_size, config.init_method, config.perform_initialization
        )

        setattr(self.dense.weight, 'sequence_parallel', config.sequence_parallel)
        setattr(self.dense.bias, 'sequence_parallel', config.sequence_parallel)

        # Deliberately the local provider, not whatever ``config.transformer_impl`` names:
        # this norm has never come from the transformer backend. ``--spec local`` selects local
        # BERT layers without touching ``transformer_impl`` (which defaults to
        # transformer_engine), so reading it here would put a TE norm on a local encoder.
        self.layer_norm = LocalSpecProvider().layer_norm(
            rms_norm=config.normalization == "RMSNorm"
        )(config=config, hidden_size=hidden_size, eps=config.layernorm_epsilon)

        self.gelu = torch.nn.functional.gelu

    def forward(self, hidden_states: Tensor) -> Tensor:
        """forward pass"""

        hidden_states = self.dense(hidden_states)
        hidden_states = self.gelu(hidden_states)
        hidden_states = self.layer_norm(hidden_states)
        return hidden_states
