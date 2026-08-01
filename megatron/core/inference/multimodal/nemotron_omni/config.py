# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

"""Nemotron Omni configuration: HF `config.json` -> mcore configs.

Three traps in this checkpoint fail silently rather than loudly, so they are handled
explicitly here rather than left to defaults:

1. `class_token_len` is `num_cls_tokens + num_registers` = 10, not mcore's default of 8.
   Getting it wrong drops or keeps 10 extra rows per image with no error.
2. The language model's `expand: 2` is stored but unused -- the real Mamba inner size is
   `mamba_num_heads * mamba_head_dim` = 4096, not `expand * hidden_size` = 5376. mcore only
   derives `mamba_num_heads` when it is left as None, so it is always set explicitly.
3. `projector_hidden_size` is 20480, four times the language model's own `d_ff`.

The RADIO tower's `args` blob in the checkpoint is serialized timm *training* state
(optimizer, augmentation, distillation bookkeeping). Only the handful of fields read below
are meaningful; everything else there is inert.
"""

from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

import torch

from megatron.core.activations import squared_relu
from megatron.core.transformer.transformer_config import TransformerConfig

# timm architecture name -> (hidden_size, num_layers, num_heads, ffn_hidden_size).
# C-RADIOv4-H is vit_huge_patch16_224.
_TIMM_VIT_DIMS: Dict[str, Tuple[int, int, int, int]] = {
    "vit_base_patch16_224": (768, 12, 12, 3072),
    "vit_large_patch16_224": (1024, 24, 16, 4096),
    "vit_huge_patch16_224": (1280, 32, 16, 5120),
}

# RADIO input conditioner values. Applied on the host during preprocessing rather than as a
# device module, which is why `input_conditioner.*` checkpoint entries are skipped on load.
OPENAI_CLIP_MEAN = (0.4815, 0.4578, 0.4082)
OPENAI_CLIP_STD = (0.2686, 0.2613, 0.2758)


@dataclass
class RadioVisionConfig:
    """RADIO vision tower geometry, resolved from the HF `vision_config` block."""

    timm_model: str = "vit_huge_patch16_224"
    patch_size: int = 16
    preferred_resolution: Tuple[int, int] = (768, 768)
    cpe_max_size: int = 2048

    # Cropped positional embedding grid. mcore derives max_num_rows/cols from these.
    max_img_h: int = 2048
    max_img_w: int = 2048

    # num_cls_tokens comes from the number of distinct teachers; num_registers is padded so
    # that (num_cls_tokens + num_registers) is a multiple of `register_multiple`.
    num_cls_tokens: int = 4
    register_multiple: int = 10

    # Video tubelet depth. > 1 activates RADIOViTModel's temporal grouping path, which also
    # changes forward() to return (features, imgs_sizes, num_frames).
    video_temporal_patch_size: int = 2
    separate_video_embedder: bool = True

    # Host-side tiling budget, in *patches* (4 patches per post-shuffle token).
    min_num_patches: int = 1024
    max_num_patches: int = 13312

    # Video framing.
    video_target_num_patches: int = 1024
    video_maintain_aspect_ratio: bool = True

    # RadioConfig defaults that the checkpoint does not override. Pinned so a future default
    # change upstream cannot silently alter numerics.
    qkv_bias: bool = True
    qk_normalization: bool = False
    layer_norm_eps: float = 1.0e-6

    norm_mean: Tuple[float, float, float] = OPENAI_CLIP_MEAN
    norm_std: Tuple[float, float, float] = OPENAI_CLIP_STD

    @property
    def num_registers(self) -> int:
        """Register tokens appended after the CLS tokens.

        Padded so the combined prefix is a multiple of `register_multiple`; for the shipped
        checkpoint that is `10 - (4 % 10) = 6`.
        """
        remainder = self.num_cls_tokens % self.register_multiple
        return self.register_multiple - remainder if remainder else 0

    @property
    def class_token_len(self) -> int:
        """Total non-patch prefix tokens per tile, i.e. mcore's `class_token_len`.

        mcore stores CLS and register tokens as one `[class_token_len, hidden]` parameter; the
        reference implementation splits them but concatenates in the same order.
        """
        return self.num_cls_tokens + self.num_registers

    @property
    def downsample_ratio(self) -> float:
        """Spatial reduction applied by pixel shuffle, per axis."""
        return 0.5

    @property
    def dims(self) -> Tuple[int, int, int, int]:
        """(hidden_size, num_layers, num_heads, ffn_hidden_size) for the timm architecture."""
        assert (
            self.timm_model in _TIMM_VIT_DIMS
        ), f"unknown RADIO timm model {self.timm_model!r}; known: {sorted(_TIMM_VIT_DIMS)}"
        return _TIMM_VIT_DIMS[self.timm_model]

    @property
    def hidden_size(self) -> int:
        """Vision tower hidden size."""
        return self.dims[0]

    @property
    def projector_input_size(self) -> int:
        """Vision projector input width.

        Pixel shuffle folds a 2x2 patch neighbourhood into channels, so the projector sees
        four times the tower's hidden size.
        """
        return self.hidden_size * 4

    @classmethod
    def from_hf(cls, vision_config: Dict[str, Any]) -> "RadioVisionConfig":
        """Build from an HF `vision_config` dict."""
        args = vision_config.get("args", {})
        teachers = args.get("teachers", [])
        # One CLS token per distinct teacher when cls_token_per_teacher is set.
        num_cls_tokens = 1
        if args.get("cls_token_per_teacher", False) and teachers:
            num_cls_tokens = len({t.get("name", i) for i, t in enumerate(teachers)})

        resolution = vision_config.get("preferred_resolution", [768, 768])
        cpe_max_size = args.get("cpe_max_size", 2048)

        return cls(
            timm_model=args.get("model", "vit_huge_patch16_224"),
            patch_size=vision_config.get("patch_size", 16),
            preferred_resolution=(int(resolution[0]), int(resolution[1])),
            cpe_max_size=cpe_max_size,
            max_img_h=cpe_max_size,
            max_img_w=cpe_max_size,
            num_cls_tokens=num_cls_tokens,
            register_multiple=args.get("register_multiple", 10),
            video_temporal_patch_size=vision_config.get("video_temporal_patch_size", 2),
            separate_video_embedder=vision_config.get("separate_video_embedder", True),
            min_num_patches=args.get("min_num_patches", 1024),
            max_num_patches=args.get("max_num_patches", 13312),
            video_target_num_patches=vision_config.get("video_target_num_patches", 1024),
            video_maintain_aspect_ratio=vision_config.get("video_maintain_aspect_ratio", True),
            qkv_bias=vision_config.get("qkv_bias", True),
            qk_normalization=vision_config.get("qk_normalization", False),
            layer_norm_eps=vision_config.get("layer_norm_eps", 1.0e-6),
        )

    def to_transformer_config(self, base: TransformerConfig) -> TransformerConfig:
        """Derive the ViT `TransformerConfig` from the language model's config.

        Args:
            base (TransformerConfig): Language model config, used only for dtype and
                parallelism; every architectural field is overridden.

        Return:
            (TransformerConfig) Config for `RADIOViTModel`.
        """
        from megatron.core.activations import fast_gelu

        hidden, layers, heads, ffn = self.dims
        config = deepcopy(base)
        config.num_layers = layers
        config.hidden_size = hidden
        config.num_attention_heads = heads
        config.num_query_groups = heads
        config.kv_channels = hidden // heads
        config.ffn_hidden_size = ffn
        config.gated_linear_unit = False
        config.activation_func = fast_gelu
        config.add_bias_linear = True
        config.add_qkv_bias = self.qkv_bias
        config.qk_layernorm = self.qk_normalization
        config.normalization = "LayerNorm"
        config.layernorm_epsilon = self.layer_norm_eps
        config.layernorm_zero_centered_gamma = False
        config.apply_rope_fusion = False
        config.bias_activation_fusion = False
        config.bias_dropout_fusion = False
        config.attention_softmax_in_fp32 = True
        config.attention_dropout = 0.0
        config.hidden_dropout = 0.0
        # Triggers TransformerBlock's final_layernorm allocation.
        config.mtp_num_layers = 0
        # The tower inherits none of the language model's MoE / Mamba / hybrid settings.
        config.num_moe_experts = None
        config.moe_ffn_hidden_size = None
        config.moe_shared_expert_intermediate_size = None
        config.moe_grouped_gemm = False
        config.moe_router_fusion = False
        config.moe_permute_fusion = False
        config.moe_shared_expert_overlap = False
        config.is_hybrid_model = False
        config.use_fused_weighted_squared_relu = False
        config.hybrid_override_pattern = None
        # The tower runs on a packed variable-length sequence; SP would shard it against the
        # cu_seqlens the varlen attention kernel is given.
        config.sequence_parallel = False
        config.variable_seq_lengths = True
        # The checkpoint quantizes only the language model: the NVFP4 variant's
        # quantized_layers list has no entry under vision_model.*, sound_encoder.*,
        # sound_projection.*, or mlp1.*. Keep the towers in params_dtype.
        config.fp8 = None
        config.fp4 = None
        return config


@dataclass
class SoundConfig:
    """Parakeet Conformer audio tower geometry, from the HF `sound_config` block."""

    hidden_size: int = 1024
    num_attention_heads: int = 8
    num_hidden_layers: int = 24
    intermediate_size: int = 4096
    conv_kernel_size: int = 9
    convolution_bias: bool = False
    subsampling_conv_channels: int = 256
    subsampling_conv_kernel_size: int = 3
    subsampling_factor: int = 8

    # Mel front-end. `feat_in` in the checkpoint is a stale default of 80 and is deliberately
    # not read: the real filterbank width is num_mel_bins.
    num_mel_bins: int = 128
    sampling_rate: int = 16000
    hop_length: int = 160
    win_length: int = 400
    n_fft: int = 512
    preemphasis: float = 0.97
    clip_seconds: float = 30.0

    @property
    def ms_per_audio_token(self) -> float:
        """Audio duration each encoder output row covers.

        `hop_length / sampling_rate * subsampling_factor` = 80 ms, i.e. 12.5 tokens/second.
        """
        return 1000.0 * self.hop_length * self.subsampling_factor / self.sampling_rate

    @classmethod
    def from_hf(cls, sound_config: Dict[str, Any]) -> "SoundConfig":
        """Build from an HF `sound_config` dict."""
        return cls(
            hidden_size=sound_config.get("hidden_size", 1024),
            num_attention_heads=sound_config.get("num_attention_heads", 8),
            num_hidden_layers=sound_config.get("num_hidden_layers", 24),
            intermediate_size=sound_config.get("intermediate_size", 4096),
            conv_kernel_size=sound_config.get("conv_kernel_size", 9),
            convolution_bias=sound_config.get("convolution_bias", False),
            subsampling_conv_channels=sound_config.get("subsampling_conv_channels", 256),
            subsampling_conv_kernel_size=sound_config.get("subsampling_conv_kernel_size", 3),
            subsampling_factor=sound_config.get("subsampling_factor", 8),
            num_mel_bins=sound_config.get("num_mel_bins", 128),
            sampling_rate=sound_config.get("sampling_rate", 16000),
        )


@dataclass
class NemotronOmniConfig:
    """Top-level Nemotron Omni configuration.

    Holds the two tower configs, the projector widths, the placeholder token strings, and the
    handful of policy knobs where this port deliberately diverges from vLLM.
    """

    vision: RadioVisionConfig = field(default_factory=RadioVisionConfig)
    sound: SoundConfig = field(default_factory=SoundConfig)

    # Language model.
    hidden_size: int = 2688
    max_sequence_length: int = 131072

    # Both projectors are RMSNorm -> Linear -> ReLU^2 -> Linear, bias-free. 20480 is 4x the
    # language model's own d_ff, so do not assume it is small.
    projector_hidden_size: int = 20480
    projector_norm_eps: float = 1.0e-5

    # Placeholder markers. The reference implementation hard-codes these strings and resolves
    # ids through the tokenizer, so the strings -- not the ids -- are the contract.
    img_start_token: str = "<img>"
    img_end_token: str = "</img>"
    img_context_token: str = "<image>"
    video_token: str = "<video>"
    audio_start_token: str = "<so_start>"
    audio_context_token: str = "<so_embedding>"
    audio_end_token: str = "<so_end>"

    # Image token budget. The reference implementation derives the per-image patch budget from
    # the engine's max_model_len, which makes image *resolution* a function of a serving flag.
    # Pinning it to the processor's value instead keeps output reproducible across
    # deployments; set to None to recover the flag-dependent behaviour.
    image_budget_sequence_length: Optional[int] = 16384

    # Fraction of video tokens to prune via efficient video sampling. The checkpoint requests
    # 0.7 and the HF reference honours it, while vLLM reads it only from a CLI flag and
    # therefore defaults to no pruning at all. Honouring the checkpoint is the divergence.
    video_pruning_rate: Optional[float] = 0.7

    @classmethod
    def from_hf(cls, hf_config: Dict[str, Any]) -> "NemotronOmniConfig":
        """Build from a parsed Nemotron Omni HF `config.json`.

        Args:
            hf_config (Dict[str, Any]): Parsed top-level `config.json`.

        Return:
            (NemotronOmniConfig) Resolved configuration.
        """
        text_config = hf_config.get("text_config") or hf_config.get("llm_config") or {}
        vision = RadioVisionConfig.from_hf(hf_config.get("vision_config", {}))
        sound = SoundConfig.from_hf(hf_config.get("sound_config", {}))

        return cls(
            vision=vision,
            sound=sound,
            hidden_size=text_config.get("hidden_size", 2688),
            max_sequence_length=hf_config.get("max_sequence_length", 131072),
            projector_hidden_size=hf_config.get(
                "projector_hidden_size", vision.projector_input_size * 4
            ),
            video_pruning_rate=hf_config.get("video_pruning_rate", 0.7),
        )

    def projector_transformer_config(
        self, base: TransformerConfig, input_size: int
    ) -> TransformerConfig:
        """Config for one of the two `RMSNorm -> Linear -> ReLU^2 -> Linear` projectors.

        The norm is folded into fc1 via `TELayerNormColumnParallelLinear`, which is why
        `normalization` is set here even though the projector has no attention.

        Args:
            base (TransformerConfig): Language model config, for dtype and parallelism.
            input_size (int): Projector input width (`vision.projector_input_size` for the
                vision projector, `sound.hidden_size` for the audio projector).

        Return:
            (TransformerConfig) Config to pass to `MultimodalProjector`.
        """
        config = deepcopy(base)
        config.hidden_size = self.hidden_size
        config.ffn_hidden_size = self.projector_hidden_size
        config.gated_linear_unit = False
        config.activation_func = squared_relu
        config.add_bias_linear = False
        config.add_qkv_bias = False
        config.normalization = "RMSNorm"
        config.layernorm_epsilon = self.projector_norm_eps
        config.num_moe_experts = None
        config.moe_ffn_hidden_size = None
        config.moe_shared_expert_intermediate_size = None
        config.moe_grouped_gemm = False
        config.is_hybrid_model = False
        config.hybrid_override_pattern = None
        config.sequence_parallel = False
        config.fp8 = None
        config.fp4 = None
        # Unused by the projector but read during MLP construction.
        config.num_layers = 1
        return config

    def language_model_overrides(self) -> Dict[str, Any]:
        """Mamba/MoE fields that must be set explicitly on the language model config.

        `expand: 2` in the checkpoint is inert: the real inner size is
        `mamba_num_heads * mamba_head_dim` = 4096, whereas mcore's fallback derivation
        (`hidden_size * expand // mamba_head_dim`, used only when `mamba_num_heads is None`)
        would build a 5376-wide mixer that cannot load the checkpoint.

        The `hybrid_override_pattern` string transfers verbatim -- mcore already defines
        `M` (Mamba), `E` (MoE), and `*` (attention) in
        `megatron.core.models.hybrid.hybrid_layer_allocation.Symbols`.
        """
        return {
            "mamba_num_heads": 64,
            "mamba_head_dim": 64,
            "mamba_state_dim": 128,
            "mamba_num_groups": 8,
            # vLLM's fp32-SSM-state config hook is keyed to a non-Omni architecture string and
            # never fires for this checkpoint. Force it here rather than inheriting that bug.
            "mamba_training_ssm_states_dtype": torch.float32,
        }
