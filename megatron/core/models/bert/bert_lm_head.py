import torch
from torch import Tensor

from megatron.core import tensor_parallel
from megatron.core.fusions.fused_layer_norm import FusedLayerNorm
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.transformer.utils import erf_gelu, get_linear_layer, openai_gelu


class BertLMHead(MegatronModule):
    """Masked LM head for Bert

    Args:
        hidden_size: hidden size
        config (TransformerConfig): TransformerConfig object
        parallel_output (bool): Do not gather the outputs, keep them split across tensor parallel ranks
        vocab_size(int): The vocabulary size
        share_embeddings_and_output_weights (bool): When True, input embeddings and output logit weights are shared. Defaults to False
        pre_process (bool): Include embedding layer (used with pipeline parallelism)
    """

    def __init__(
        self,
        hidden_size: int,
        config: TransformerConfig,
        parallel_output: bool,
        vocab_size: int,
        pre_process: bool,
        share_embeddings_and_output_weights: bool = False,
    ):
        super().__init__(config=config)

        self.vocab_size = vocab_size
        self.parallel_output = parallel_output

        # TODO: Should switch this to TE ?
        self.dense = get_linear_layer(
            hidden_size, hidden_size, config.init_method, config.perform_initialization
        )

        setattr(self.dense.weight, 'sequence_parallel', config.sequence_parallel)
        setattr(self.dense.bias, 'sequence_parallel', config.sequence_parallel)

        self.layernorm = FusedLayerNorm(
            config=config,
            hidden_size=hidden_size,
            eps=config.layernorm_epsilon,
            sequence_parallel=config.sequence_parallel,
        )

        self.gelu = torch.nn.functional.gelu
        # TODO Use activation_func in config to determine what to use
        # if config.openai_gelu: # Dont have these configs in transformer config yet
        #    self.gelu = openai_gelu
        # elif config.onnx_safe: # Dont have these configs in transformer config yet
        #   self.gelu = erf_gelu

        self.output_layer = tensor_parallel.ColumnParallelLinear(
            config.hidden_size,
            self.vocab_size,
            config=config,
            init_method=config.init_method,
            bias=True,
            skip_bias_add=False,
            gather_output=not self.parallel_output,
            skip_weight_param_allocation=pre_process and share_embeddings_and_output_weights,
        )

    def forward(self, hidden_states: Tensor, word_embeddings_weight: Tensor) -> Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.gelu(hidden_states)
        hidden_states = self.layernorm(hidden_states)
        logits, _ = self.output_layer(hidden_states, weight=word_embeddings_weight)
        return logits
