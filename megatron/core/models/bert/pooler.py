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
        init_method (callable): weight initialization method for the linear layer. bias is set to zero.
        config (TransformerConfig): The transformer configuration
        sequence_parallel (bool): Using sequence parallel ? Defaults to False
    """

    def __init__(
        self,
        hidden_size: int,
        init_method: callable,
        config: TransformerConfig,
        sequence_parallel: bool = False,
    ):
        super(Pooler, self).__init__(config)
        # TODO: Should switch this to TE ?
        self.dense = get_linear_layer(
            hidden_size, hidden_size, init_method, config.perform_initialization
        )
        self.sequence_parallel = sequence_parallel

    def forward(self, hidden_states: Tensor, sequence_index=0):
        # hidden_states: [s, b, h]
        # sequence_index: index of the token to pool.

        # gather data along sequence dimensions
        # same pooler is run on all tensor parallel nodes
        if self.sequence_parallel:
            hidden_states = tensor_parallel.gather_from_sequence_parallel_region(
                hidden_states, tensor_parallel_output_grad=False
            )

        pooled = hidden_states[sequence_index, :, :]
        pooled = self.dense(pooled)
        pooled = torch.tanh(pooled)
        return pooled
