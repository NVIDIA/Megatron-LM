# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.

"""Central registry of per-encoder defaults.

One EncoderSpec per `vision_model_type` carries everything callers need to
instantiate the encoder without hard-coding numbers in multiple places:

  * patch_dim, default image size, class_token_len, native-spatial-merge flag
  * implementation model type, default converted checkpoint directory
  * pixel_mean / pixel_std (ImageNet-style normalisation)
  * the full TransformerConfig arch (num_layers, hidden_size, ffn_hidden_size,
    activation, normalisation, bias flags, RoPE flags, ...)

Consumers:
  - examples/multimodal/config.py::get_vision_model_config (via apply_to_config)
  - examples/multimodal/v3/energon_multimodal_provider.py (pixel statistics)
  - examples/multimodal/multimodal_args.py::resolve_multimodal_encoder_args

Adding a new encoder = one entry here.
"""

from dataclasses import dataclass
from typing import Callable, Dict, Optional, Tuple, Union

_CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
_CLIP_STD = (0.26862954, 0.26130258, 0.27577711)
_IN_MEAN = (0.485, 0.456, 0.406)
_IN_STD = (0.229, 0.224, 0.225)
_HALF_MEAN = (0.5, 0.5, 0.5)
_HALF_STD = (0.5, 0.5, 0.5)


ActivationFunc = Union[str, Callable]


def _gelu_tanh(x):
    import torch

    return torch.nn.functional.gelu(x, approximate='tanh')


def _resolve_activation(activation_func: ActivationFunc) -> Callable:
    if callable(activation_func):
        return activation_func
    if activation_func == "gelu":
        import torch

        return torch.nn.functional.gelu
    if activation_func == "silu":
        import torch

        return torch.nn.functional.silu
    if activation_func == "gelu_tanh":
        return _gelu_tanh
    if activation_func == "fast_gelu":
        from megatron.core.activations import fast_gelu

        return fast_gelu
    if activation_func == "quick_gelu":
        from megatron.core.activations import quick_gelu

        return quick_gelu
    raise ValueError(f"unknown activation function {activation_func!r}")


@dataclass(frozen=True)
class EncoderSpec:
    """Per-encoder defaults: image geometry, pixel stats, transformer arch."""

    # ---- Image geometry / data-loader ----
    name: str
    patch_dim: int
    default_img_h: int
    default_img_w: int
    class_token_len: int = 0
    has_native_spatial_merge: bool = False
    model_type: Optional[str] = None
    checkpoint_dir: Optional[str] = None
    dynamic_resolution: bool = False
    pixel_shuffle: bool = False
    conv_merging: bool = False
    use_tiling: bool = False
    max_num_tiles: int = 1
    use_thumbnail: bool = False
    dynamic_resolution_max_patches: int = 0
    dynamic_resolution_max_side: Optional[int] = None
    radio_force_eval_mode: bool = False
    radio_hf_resolution: bool = False
    pixel_mean: Tuple[float, float, float] = _CLIP_MEAN
    pixel_std: Tuple[float, float, float] = _CLIP_STD

    # ---- Transformer architecture (None → keep TransformerConfig default) ----
    num_layers: Optional[int] = None
    hidden_size: Optional[int] = None
    num_attention_heads: Optional[int] = None
    num_query_groups: Optional[int] = None  # None → mirrors num_attention_heads
    ffn_hidden_size: Optional[int] = None
    kv_channels: Optional[int] = None  # None → hidden // heads (mcore default)
    gated_linear_unit: bool = False
    activation_func: ActivationFunc = "gelu"
    add_bias_linear: bool = True
    add_qkv_bias: bool = True
    normalization: str = 'LayerNorm'
    layernorm_epsilon: Optional[float] = None  # None → TransformerConfig default
    qk_layernorm: Optional[bool] = None  # None → don't touch
    rotary_interleaved: bool = False
    # internvit rounds its 24 heads up to the next multiple of TP at build time.
    tp_round_up_heads: bool = False

    def apply_to_config(self, config, apply_query_key_layer_scaling: bool = False):
        """Write this spec onto an existing TransformerConfig and return it."""
        if self.num_layers is not None:
            config.num_layers = self.num_layers
        if self.hidden_size is not None:
            config.hidden_size = self.hidden_size
        if self.num_attention_heads is not None:
            if self.tp_round_up_heads:
                tp = config.tensor_model_parallel_size
                config.num_attention_heads = (self.num_attention_heads // tp + 1) * tp
            else:
                config.num_attention_heads = self.num_attention_heads
        if self.num_query_groups is not None:
            config.num_query_groups = self.num_query_groups
        elif self.num_attention_heads is not None:
            config.num_query_groups = config.num_attention_heads
        if self.ffn_hidden_size is not None:
            config.ffn_hidden_size = self.ffn_hidden_size
        if self.kv_channels is not None:
            config.kv_channels = self.kv_channels
        config.add_bias_linear = self.add_bias_linear
        config.add_qkv_bias = self.add_qkv_bias
        config.gated_linear_unit = self.gated_linear_unit
        config.activation_func = _resolve_activation(self.activation_func)
        config.normalization = self.normalization
        if self.layernorm_epsilon is not None:
            config.layernorm_epsilon = self.layernorm_epsilon
        if self.qk_layernorm is not None:
            config.qk_layernorm = self.qk_layernorm
        config.rotary_interleaved = self.rotary_interleaved
        # Defaults that are uniform across every encoder in this repo.
        config.hidden_dropout = 0.0
        config.attention_dropout = 0.0
        config.layernorm_zero_centered_gamma = False
        config.apply_query_key_layer_scaling = apply_query_key_layer_scaling
        config.bias_activation_fusion = False
        config.bias_dropout_fusion = False
        config.attention_softmax_in_fp32 = True
        config.apply_rope_fusion = False
        return config


def _radio_h_spec(
    name: str,
    checkpoint_dir: str,
    *,
    dynamic_resolution: bool,
    pixel_shuffle: bool,
    use_tiling: bool,
    max_num_tiles: int = 1,
    use_thumbnail: bool = False,
    radio_hf_resolution: bool = False,
) -> EncoderSpec:
    return EncoderSpec(
        name=name,
        model_type="radio",
        checkpoint_dir=checkpoint_dir,
        patch_dim=16,
        default_img_h=512,
        default_img_w=512,
        class_token_len=10,
        radio_force_eval_mode=True,
        dynamic_resolution=dynamic_resolution,
        pixel_shuffle=pixel_shuffle,
        use_tiling=use_tiling,
        max_num_tiles=max_num_tiles,
        use_thumbnail=use_thumbnail,
        radio_hf_resolution=radio_hf_resolution,
        num_layers=32,
        hidden_size=1280,
        num_attention_heads=16,
        ffn_hidden_size=5120,
        kv_channels=80,
        activation_func="fast_gelu",
        layernorm_epsilon=1e-6,
        qk_layernorm=False,
    )


REGISTRY: Dict[str, EncoderSpec] = {
    "clip": EncoderSpec(
        name="clip",
        patch_dim=16,
        default_img_h=512,
        default_img_w=512,
        num_layers=24,
        hidden_size=1024,
        num_attention_heads=16,
        ffn_hidden_size=4096,
        kv_channels=64,
        activation_func="quick_gelu",
    ),
    "siglip": EncoderSpec(
        name="siglip",
        patch_dim=16,
        default_img_h=512,
        default_img_w=512,
        num_layers=27,
        hidden_size=1152,
        num_attention_heads=16,
        ffn_hidden_size=4304,
        kv_channels=72,
        activation_func="fast_gelu",
        layernorm_epsilon=1e-6,
        qk_layernorm=False,
    ),
    "internvit": EncoderSpec(
        name="internvit",
        patch_dim=16,
        default_img_h=512,
        default_img_w=512,
        num_layers=45,
        hidden_size=3200,
        num_attention_heads=24,
        tp_round_up_heads=True,
        ffn_hidden_size=12800,
        activation_func="gelu",
        add_qkv_bias=False,
        normalization='RMSNorm',
        layernorm_epsilon=1e-6,
    ),
    "internvit300M": EncoderSpec(
        name="internvit300M",
        patch_dim=16,
        default_img_h=512,
        default_img_w=512,
        num_layers=24,
        hidden_size=1024,
        num_attention_heads=16,
        ffn_hidden_size=4096,
        kv_channels=64,
        activation_func="gelu",
        layernorm_epsilon=1e-6,
        qk_layernorm=False,
    ),
    "radio": _radio_h_spec(
        "radio",
        "c_radio_vit_h",
        dynamic_resolution=True,
        pixel_shuffle=False,
        use_tiling=False,
        radio_hf_resolution=True,
    ),
    "post-c-radio-omni": _radio_h_spec(
        "post-c-radio-omni",
        "post-c-radio-omni",
        dynamic_resolution=True,
        pixel_shuffle=True,
        use_tiling=False,
    ),
    "radio-g": EncoderSpec(
        name="radio-g",
        patch_dim=16,
        default_img_h=512,
        default_img_w=512,
        num_layers=40,
        hidden_size=1536,
        num_attention_heads=24,
        ffn_hidden_size=4096,
        kv_channels=64,
        gated_linear_unit=True,
        activation_func="silu",
        layernorm_epsilon=1e-6,
        qk_layernorm=False,
    ),
    "cradio-g": EncoderSpec(
        name="cradio-g",
        patch_dim=16,
        default_img_h=512,
        default_img_w=512,
        class_token_len=10,
        num_layers=40,
        hidden_size=1536,
        num_attention_heads=24,
        ffn_hidden_size=6144,
        kv_channels=64,
        activation_func="fast_gelu",
        layernorm_epsilon=1e-6,
        qk_layernorm=False,
    ),
    # Pixtral-12B: SwiGLU, RMSNorm, Mistral-native interleaved 2D RoPE, no bias, no CLS
    "pixtral-vit": EncoderSpec(
        name="pixtral-vit",
        patch_dim=16,
        default_img_h=512,
        default_img_w=512,
        num_layers=24,
        hidden_size=1024,
        num_attention_heads=16,
        ffn_hidden_size=4096,
        kv_channels=64,
        gated_linear_unit=True,
        activation_func="silu",
        add_bias_linear=False,
        add_qkv_bias=False,
        normalization='RMSNorm',
        layernorm_epsilon=1e-5,
        rotary_interleaved=True,
    ),
    # Pixtral-Large (Mistral-Large-3-675B): 48L/1664h/8192ffn + 2×2 patch merger
    "pixtral-vit-large": EncoderSpec(
        name="pixtral-vit-large",
        patch_dim=14,
        default_img_h=1540,
        default_img_w=1540,
        has_native_spatial_merge=True,
        checkpoint_dir="pixtral_large",
        dynamic_resolution=True,
        conv_merging=True,
        dynamic_resolution_max_patches=12100,
        dynamic_resolution_max_side=1540,
        num_layers=48,
        hidden_size=1664,
        num_attention_heads=16,
        ffn_hidden_size=8192,
        kv_channels=104,
        gated_linear_unit=True,
        activation_func="silu",
        add_bias_linear=False,
        add_qkv_bias=False,
        normalization='RMSNorm',
        layernorm_epsilon=1e-5,
        rotary_interleaved=True,
    ),
    # Qwen3.5-MoE VL: GELU-tanh MLP, LayerNorm, 2D RoPE + learned pos, bias
    "qwen-vl": EncoderSpec(
        name="qwen-vl",
        patch_dim=16,
        default_img_h=768,
        default_img_w=768,
        has_native_spatial_merge=True,
        checkpoint_dir="qwen35vl_moe",
        dynamic_resolution=True,
        conv_merging=True,
        dynamic_resolution_max_patches=4096,
        dynamic_resolution_max_side=768,
        pixel_mean=_HALF_MEAN,
        pixel_std=_HALF_STD,
        num_layers=27,
        hidden_size=1152,
        num_attention_heads=16,
        ffn_hidden_size=4304,
        kv_channels=72,
        activation_func="gelu_tanh",
        layernorm_epsilon=1e-6,
    ),
    # Kimi-K2: GELU-tanh MLP, LayerNorm, interleaved 2D RoPE + learned pos, bias
    "kimi-vit": EncoderSpec(
        name="kimi-vit",
        patch_dim=14,
        default_img_h=896,
        default_img_w=896,
        has_native_spatial_merge=True,
        checkpoint_dir="kimi_k26",
        dynamic_resolution=True,
        conv_merging=True,
        dynamic_resolution_max_patches=8192,
        dynamic_resolution_max_side=896,
        pixel_mean=_HALF_MEAN,
        pixel_std=_HALF_STD,
        num_layers=27,
        hidden_size=1152,
        num_attention_heads=16,
        ffn_hidden_size=4304,
        kv_channels=72,
        activation_func="gelu_tanh",
        layernorm_epsilon=1e-5,
        rotary_interleaved=True,
    ),
}


def get_spec(vision_model_type: str) -> EncoderSpec:
    """Return the spec for `vision_model_type`, or raise KeyError with a list of known types."""
    try:
        return REGISTRY[vision_model_type]
    except KeyError:
        raise KeyError(
            f"Unknown vision_model_type {vision_model_type!r}. " f"Known: {sorted(REGISTRY)}"
        )
