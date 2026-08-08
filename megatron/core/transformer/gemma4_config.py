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

    # Number of trailing layers that borrow K/V from an earlier producer of the same
    # type (HF ``num_kv_shared_layers``). E4B: layers 24..41 borrow.
    num_kv_shared_layers: int = 18

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
        # Gemma4 is RMSNorm-only (HF Gemma4RMSNorm). The local spec hard-wires
        # Gemma4RMSNorm via its builders and ignores this field, but the TE spec's
        # TENorm dispatches on config.normalization: the base TransformerConfig
        # default "LayerNorm" would make every TENorm a (mean-subtracting)
        # te.pytorch.LayerNorm, which is structurally wrong for Gemma. Force RMSNorm
        # so TENorm builds te.pytorch.RMSNorm with plain weight
        # (layernorm_zero_centered_gamma=False), matching the converted HF weights.
        self.normalization = "RMSNorm"
        self.heterogeneous_block_specs = True
        # Per-layer attention type ("sliding"/"full"), 0-based, used by the KV bus and
        # rope/mask selection. Derived from full_attention_layers.
        self.layer_types = tuple(
            "full" if i in self.full_attention_layers else "sliding"
            for i in range(self.num_layers)
        )
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
        # Only kv_channels (head_dim 256->512) and window_size (drop sliding window)
        # are base TransformerConfig fields that differ per layer. rotary_base is NOT
        # a TransformerConfig field and is not needed per-layer: Gemma4RotaryEmbedding
        # bakes in both thetas (1e4/1e6) and selects by layer_type. The sliding/full
        # rope bases are carried as gemma4-only fields (sliding_rotary_base/
        # full_rotary_base) for the rope module to read.
        keys_to_update = {}
        if self.is_full_attention(global_layer_number):
            keys_to_update['kv_channels'] = self.full_kv_channels
            keys_to_update['window_size'] = None

        base_field_names = {f.name for f in fields(TransformerConfig)}
        transformer_config_dict = {
            f.name: getattr(self, f.name) for f in fields(self) if f.name in base_field_names
        }
        transformer_config_dict.update(keys_to_update)

        # Build a base TransformerConfig (not Gemma4TransformerConfig) so its
        # __post_init__ does not re-clobber the per-layer kv_channels/rotary_base/
        # window_size overrides above.
        layer_config = TransformerConfig(**transformer_config_dict)
        # Carry ALL gemma4-only fields the per-layer modules need (PLE dims, softcap,
        # KV-bus role, rope/mask selection). They are not base TransformerConfig
        # fields, so the dict round-trip above drops them; re-attach generically.
        for f in fields(self):
            if f.name not in base_field_names:
                setattr(layer_config, f.name, getattr(self, f.name))
        # layer_types is derived in __post_init__ (not a dataclass field), so it is
        # not covered by the fields() loop above; carry it explicitly.
        layer_config.layer_types = self.layer_types
        return layer_config
