from dataclass import dataclass
from megatron.core.transformer import MLATransformerConfig

@dataclass
class DeepSeekV2TransformerConfig(MLATransformerConfig):
    moe_ffn_hidden_size: int = None
    moe_layer_freq: int = None