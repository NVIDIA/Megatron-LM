# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
import torch
from torch import Tensor

from megatron.core import tensor_parallel
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.transformer.utils import get_linear_layer


class Pooler(MegatronModule):
    """Pooler layer.

    Pool hidden states of a specific token (for example start of the
    sequence) and add a linear transformation followed by a tanh.

    Args:
        hidden_size (int): The hidden size_
        init_method (callable): Weight initializer for the linear layer. Its bias starts at zero.
        config (TransformerConfig): The transformer configuration
        sequence_parallel (bool): Using squence parallel ? Defaults to False
        tp_group: Tensor-parallel group for gathering sequence-parallel hidden states.
    """

    def __init__(
        self,
        hidden_size: int,
        init_method: callable,
        config: TransformerConfig,
        sequence_parallel: bool = False,
        tp_group: torch.distributed.ProcessGroup | None = None,
    ):
        super(Pooler, self).__init__(config)
        # TODO: Shoudl switch this to TE ?
        self.dense = get_linear_layer(
            hidden_size, hidden_size, init_method, config.perform_initialization
        )
        self.sequence_parallel = sequence_parallel
        self.tp_group = tp_group

    def forward(self, hidden_states: Tensor, sequence_index=0):
        """Pool the token at sequence_index from [sequence, batch, hidden] states."""

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
