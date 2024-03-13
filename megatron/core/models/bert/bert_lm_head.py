import torch
from torch import Tensor

from megatron.core import tensor_parallel, parallel_state
from megatron.core.fusions.fused_layer_norm import FusedLayerNorm
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.transformer.utils import erf_gelu, get_linear_layer, make_sharded_tensors_for_checkpoint, openai_gelu
from megatron.core.utils import make_tp_sharded_tensor_for_checkpoint

class BertLMHead(MegatronModule):
    """Masked LM head for Bert. 

    Args:
        hidden_size: hidden size
        config (TransformerConfig): TransformerConfig object
     """

    def __init__(
        self,
        hidden_size: int,
        config: TransformerConfig,
    ):
        super().__init__(config=config)

        # TODO: Should switch this to TE ?
        self.dense = get_linear_layer(
            hidden_size, hidden_size, config.init_method, config.perform_initialization
        )

        setattr(self.dense.weight, 'sequence_parallel', config.sequence_parallel)
        setattr(self.dense.bias, 'sequence_parallel', config.sequence_parallel)

        self.layer_norm = FusedLayerNorm(
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

    def forward(self, hidden_states: Tensor) -> Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.gelu(hidden_states)
        hidden_states = self.layer_norm(hidden_states)
        return hidden_states
    
    def sharded_state_dict(self, prefix=''):
        sharded_state_dict = {}

        dense_prefix = f'{prefix}dense.'
        state_dict = self.dense.state_dict(keep_vars=True)
        dense_layer_sharded_state_dict = make_sharded_tensors_for_checkpoint(state_dict, dense_prefix)
        sharded_state_dict.update(dense_layer_sharded_state_dict)

        layer_norm_sharded_state_dict = self.layer_norm.sharded_state_dict(prefix=prefix)
        sharded_state_dict.update(layer_norm_sharded_state_dict)

        return sharded_state_dict
