import torch
from megatron.core.transformer import TransformerConfig

class WrappedTorchLayerNorm(torch.nn.LayerNorm):
    
    def __init__(
        self,
        config: TransformerConfig, ## unused
        hidden_size: int,
        eps: float = 1e-5,
        persist_layer_norm: bool = True,
        zero_centered_gamma: bool = False,
        normalization: str = "LayerNorm",  # included to match TE interface
    ):
        super().__init__(
            normalized_shape = hidden_size, ## applied to last len(normalized_shape.size) dimensions
            eps = eps,
        )
