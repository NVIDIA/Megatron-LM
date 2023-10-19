import torch

from megatron.core import tensor_parallel
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.utils import erf_gelu, get_linear_layer, openai_gelu
from megatron.model import LayerNorm


class BertLMHead(MegatronModule):
    """Masked LM head for Bert

    Arguments:
        config: TransformerConfig object
        mpu_vocab_size: model parallel size of vocabulary.
        hidden_size: hidden size
        parallel_output: whether output logits being distributed or not.
    """

    def __init__(
        self,
        mpu_vocab_size,
        hidden_size,
        config,
        parallel_output,
        vocab_size,
        pre_process,
        share_embeddings_and_output_weights,
    ):
        super().__init__(config=config)

        self.vocab_size = vocab_size
        self.bias = torch.nn.Parameter(torch.zeros(mpu_vocab_size))
        tensor_parallel.set_tensor_model_parallel_attributes(self.bias, True, 0, 1)
        self.parallel_output = parallel_output

        # TODO: Shoudl switch this to TELinear ? Or club this sand the LayerNorm to TELayerNormColumnParallelLinear ?
        self.dense = get_linear_layer(hidden_size, hidden_size, config.init_method)

        setattr(self.dense.weight, 'sequence_parallel', config.sequence_parallel)
        setattr(self.dense.bias, 'sequence_parallel', config.sequence_parallel)

        self.layernorm = LayerNorm(
            hidden_size, eps=config.layernorm_epsilon, sequence_parallel=config.sequence_parallel
        )

        self.gelu = torch.nn.functional.gelu
        # TODO Use activation_func in config to etermine what to use
        # if config.openai_gelu: # Dont have these configs in transfomer config yet
        #    self.gelu = openai_gelu
        # elif config.onnx_safe: # Dont have these configs in transfomer config yet
        #   self.gelu = erf_gelu

        self.output_layer = tensor_parallel.ColumnParallelLinear(
            config.hidden_size,
            self.vocab_size,
            config=config,
            init_method=config.init_method,
            bias=False,
            skip_bias_add=False,
            gather_output=not self.parallel_output,
            skip_weight_param_allocation=pre_process and share_embeddings_and_output_weights,
        )

    def forward(self, hidden_states, word_embeddings_weight):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.gelu(hidden_states)
        hidden_states = self.layernorm(hidden_states)
        logits, _ = self.output_layer(hidden_states, weight=word_embeddings_weight)
        return logits
