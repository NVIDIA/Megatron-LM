# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
from typing import Optional

import torch
from torch import Tensor

from megatron.core import tensor_parallel
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.transformer.utils import get_linear_layer
from megatron.core.utils import get_tensor_model_parallel_group_if_none


class Pooler(MegatronModule):
    """Pooler layer.

    Pool hidden states of a specific token (for example start of the
    sequence) and add a linear transformation followed by a tanh.

    Args:
        hidden_size (int): The hidden size_
        init_method (callable): weight initialization method for the linear layer. bias is set to
            zero.
        config (TransformerConfig): The transformer configuration
        sequence_parallel (bool): Using squence parallel ? Defaults to False
        tp_group (torch.distributed.ProcessGroup, optional): Tensor-parallel group over which the
            sequence-parallel hidden states are gathered. Defaults to the global tensor-parallel
            group.
    """

    def __init__(
        self,
        hidden_size: int,
        init_method: callable,
        config: TransformerConfig,
        sequence_parallel: bool = False,
        tp_group: Optional[torch.distributed.ProcessGroup] = None,
    ):
        super(Pooler, self).__init__(config)
        # TODO: Shoudl switch this to TE ?
        self.dense = get_linear_layer(
            hidden_size, hidden_size, init_method, config.perform_initialization
        )
        self.sequence_parallel = sequence_parallel
        self.tp_group = get_tensor_model_parallel_group_if_none(tp_group)

    def forward(self, hidden_states: Tensor, sequence_index=0):
        """Pool the hidden state at ``sequence_index`` and apply the dense layer and tanh."""
        # hidden_states: [s, b, h]
        # sequence_index: index of the token to pool.

        # gather data along sequence dimensions
        # same pooler is run on all tensor parallel nodes
        if self.sequence_parallel:
            hidden_states = tensor_parallel.gather_from_sequence_parallel_region(
                hidden_states, tensor_parallel_output_grad=False, group=self.tp_group
            )

        pooled = hidden_states[sequence_index, :, :]
        pooled = self.dense(pooled)
        pooled = torch.tanh(pooled)
        return pooled
