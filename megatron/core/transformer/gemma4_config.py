# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

from dataclasses import dataclass, fields

from megatron.core.transformer.transformer_config import TransformerConfig


@dataclass
class Gemma4TransformerConfig(TransformerConfig):
    """Configuration object for Gemma 4 transformers.

    Gemma 4 is heterogeneous: layers alternate between local *sliding* attention
    (``head_dim`` 256, RoPE base 1e4, sliding window 512) and global *full*
    attention (``head_dim`` 512, RoPE base 1e6, proportional RoPE). The per-layer
    differences are produced by :meth:`get_config_for_layer`, mirroring the
    ``HeterogeneousTransformerConfig.get_config_for_layer`` precedent. The base
    config holds the sliding defaults; full layers override them.

    All Gemma4-specific forward-only knobs (softcap, embedding scaling, PLE) live
    on this subclass so the base ``TransformerConfig`` is untouched.
    """

    # Layer indices (0-based) that use global full attention; the rest are sliding.
    # E4B: every i where (i+1) % 6 == 0 over 42 layers -> {5,11,17,23,29,35,41}.
    full_attention_layers: tuple[int, ...] = (5, 11, 17, 23, 29, 35, 41)

    # Sliding (local) attention parameters; full layers override via get_config_for_layer.
    sliding_kv_channels: int = 256
    full_kv_channels: int = 512
    sliding_rotary_base: float = 1e4
    full_rotary_base: float = 1e6
    sliding_window: int = 512

    # Gemma4 forward-only flags (not folded into weights).
    final_logit_softcapping: float = 30.0
    apply_embedding_scaling: bool = True
    hidden_size_per_layer_input: int = 256
    vocab_size_per_layer_input: int = 262144

    def __post_init__(self):
        # Sliding layers are the common case -> make them the base defaults.
        self.kv_channels = self.sliding_kv_channels
        self.rotary_base = self.sliding_rotary_base
        self.softmax_scale = 1.0
        self.window_size = (self.sliding_window, 0)
        self.heterogeneous_block_specs = True
        super().__post_init__()

    def is_full_attention(self, global_layer_number: int) -> bool:
        """Whether the given 1-based global layer number uses full attention."""
        return (global_layer_number - 1) in self.full_attention_layers

    def get_config_for_layer(self, global_layer_number: int) -> TransformerConfig:
        """Return a ``TransformerConfig`` specialized for the given layer.

        ``global_layer_number`` is 1-based (Megatron convention). Full-attention
        layers swap in the 512 head_dim / 1e6 RoPE base and drop the sliding
        window; sliding layers inherit this config's defaults.
        """
        keys_to_update = {}
        if self.is_full_attention(global_layer_number):
            keys_to_update['kv_channels'] = self.full_kv_channels
            keys_to_update['rotary_base'] = self.full_rotary_base
            keys_to_update['window_size'] = None

        transformer_config_field_names = {f.name for f in fields(TransformerConfig)}
        transformer_config_dict = {
            f.name: getattr(self, f.name) for f in fields(self) if f.name in transformer_config_field_names
        }
        transformer_config_dict.update(keys_to_update)

        return TransformerConfig(**transformer_config_dict)
