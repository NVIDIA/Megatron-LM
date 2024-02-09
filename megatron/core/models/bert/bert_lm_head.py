import torch
from torch import Tensor

from megatron.core import tensor_parallel, parallel_state
from megatron.core.fusions.fused_layer_norm import FusedLayerNorm
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.transformer.utils import erf_gelu, get_linear_layer, make_sharded_tensors_for_checkpoint, openai_gelu
from megatron.core.utils import make_tp_sharded_tensor_for_checkpoint

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
        self.share_embeddings_and_output_weights = share_embeddings_and_output_weights
        # TODO: Shoudl switch this to TE ?
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
        # if config.openai_gelu: # Dont have these configs in transfomer config yet
        #    self.gelu = openai_gelu
        # elif config.onnx_safe: # Dont have these configs in transfomer config yet
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
    
    def sharded_state_dict(self, prefix=''):
        sharded_state_dict = {}

        dense_prefix = f'{prefix}dense.'
        state_dict = self.dense.state_dict()
        #TODO need to check fi this dictionary of weight and bias is required
        dense_layer_sharded_state_dict = make_sharded_tensors_for_checkpoint(state_dict, dense_prefix,  {'weight': 0, 'bias': 0})
        sharded_state_dict.update(dense_layer_sharded_state_dict)

        output_layer_prefix = f'{prefix}output'

        #if share embeddings is enabled it is stored in the bert_model class itself in sharded_state_dict function
        if not self.share_embeddings_and_output_weights:     
            output_layer_sharded_state_dict = self.output_layer.sharded_state_dict(prefix=output_layer_prefix)
            sharded_state_dict.update(output_layer_sharded_state_dict)

        return sharded_state_dict
