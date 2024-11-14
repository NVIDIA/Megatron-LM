from dataclasses import dataclass
from megatron.core.transformer import TransformerConfig


@dataclass
class DeepSeekV2TransformerConfig(TransformerConfig):

    moe_ffn_hidden_size: int = None

    enable_shared_expert: bool = False

    q_lora_rank: int = None

    kv_lora_rank: int = None

    qk_nope_head_dim: int = None

    qk_rope_head_dim: int = None

    v_head_dim: int = None

    num_shared_experts: int = None

    moe_layer_freq: int = None

    rotary_base: int = None

    rotary_scaling_factor: int = None

    max_position_embeddings: int = None

    moe_aux_loss_coeff: float = 0.0