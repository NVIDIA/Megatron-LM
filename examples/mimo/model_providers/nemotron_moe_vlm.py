# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Nemotron6-MoE VLM model provider for hetero MIMO examples.

Declares the MIMO/vision provider args, applies the Nemotron architecture
preset, and builds the language ``MambaModel`` (hybrid Mamba/MoE) and the
vision ``RADIOViTModel`` + ``MultimodalProjector`` ``ModuleSpec`` s.
"""

from __future__ import annotations

import argparse
from contextlib import nullcontext
from copy import deepcopy
from typing import Optional

import torch

from megatron.core.activations import fast_gelu, squared_relu
from megatron.core.hyper_comm_grid import HyperCommGrid
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_with_transformer_engine_spec
from megatron.core.models.gpt.gpt_model import GPTModel
from megatron.core.models.mamba.mamba_layer_specs import mamba_stack_spec
from megatron.core.models.mamba.mamba_model import MambaModel
from megatron.core.models.mimo.submodules.vision import VisionModalitySubmodules
from megatron.core.models.multimodal.llava_model import pixel_shuffle
from megatron.core.models.vision.multimodal_projector import MultimodalProjector
from megatron.core.models.vision.radio import RADIOViTModel
from megatron.core.models.vision.vit_layer_specs import get_vit_layer_with_transformer_engine_spec
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.tensor_parallel import ColumnParallelLinear
from megatron.core.transformer.enums import AttnBackend
from megatron.core.transformer.mlp import MLP, MLPSubmodules
from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.transformer.utils import sharded_state_dict_default

try:
    from megatron.core.extensions.transformer_engine import TERowParallelLinear
except ImportError:  # pragma: no cover - TE always present in the CI container
    TERowParallelLinear = None

MOCK_MODEL_PROVIDER = "mock"
NEMOTRON_MODEL_PROVIDER = "nemotron-moe-vlm"
NEMOTRON_IMAGE_SEQ_PER_TILE = 256
NEMOTRON_MAX_NUM_TILES = 12
NEMOTRON_DEFAULT_STAGE = "stage2"
NEMOTRON_SEQ_LENGTH = 8192
NEMOTRON_VISION_ENCODER_KEY = "radio_encoder"


# Process-group / grid helpers.
def get_grid_dim_size(grid: HyperCommGrid, dim: str) -> int:
    """Return the size of ``dim`` in a HyperCommGrid, or 1 if absent."""
    try:
        return int(grid.shape[grid.dim_names.index(dim)])
    except (ValueError, AttributeError):
        return 1


def get_group_size_or(pg, fallback: int) -> int:
    """Return ``pg``'s world size when joinable, else ``fallback``."""
    import torch.distributed as dist

    if pg is None:
        return fallback
    return dist.get_world_size(group=pg)


def get_group_rank_or(pg, fallback: int = 0) -> int:
    """Return this rank's index within ``pg``, else ``fallback``."""
    import torch.distributed as dist

    if pg is None:
        return fallback
    rank = dist.get_rank(group=pg)
    return rank if rank >= 0 else fallback


def is_process_group_member(pg) -> bool:
    """Whether the current rank belongs to ``pg``."""
    import torch.distributed as dist

    return pg is not None and dist.get_rank(group=pg) >= 0


def is_nemotron_moe_vlm(args: argparse.Namespace) -> bool:
    """Return whether the Nemotron6-MoE VLM provider is active."""
    return getattr(args, "model_provider", None) == NEMOTRON_MODEL_PROVIDER


def add_model_provider_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Register new model-provider args for hetero MIMO examples.

    Only MIMO/vision/provider knobs are declared here; stock ``arguments.py``
    owns the ``TransformerConfig`` field flags.
    """
    provider = parser.add_argument_group("mimo model provider")
    provider.add_argument(
        "--model-provider",
        choices=[MOCK_MODEL_PROVIDER, NEMOTRON_MODEL_PROVIDER],
        default=MOCK_MODEL_PROVIDER,
        help="Which MIMO model provider/preset to build.",
    )
    # Vision / MIMO-specific knobs.
    provider.add_argument("--image-seq-length", type=int, default=None,
                          help="Total image tokens emitted by the encoder per sample.")
    provider.add_argument("--image-token-id", type=int, default=511,
                          help="Vocab id of the placeholder image token.")
    provider.add_argument("--pad-token-id", type=int, default=0)
    provider.add_argument("--image-token", type=str, default="<image>")
    provider.add_argument("--tokenizer-prompt-format", type=str, default="nemotron6-moe")
    provider.add_argument("--image-tag-type", type=str, default="")
    provider.add_argument("--force-system-message", action="store_true")
    provider.add_argument("--class-token-len", type=int, default=8)
    provider.add_argument(
        "--num-image-tiles",
        "--max-num-tiles",
        dest="num_image_tiles",
        type=int,
        default=NEMOTRON_MAX_NUM_TILES,
    )
    provider.add_argument("--vision-model-type", type=str, default="radio")
    provider.add_argument("--pixel-shuffle", action="store_true")
    provider.add_argument("--disable-vision-class-token", action="store_true")
    provider.add_argument("--use-tiling", action="store_true")
    provider.add_argument("--use-thumbnail", action="store_true")
    provider.add_argument(
        "--dynamic-resolution",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Patchify each image at its native aspect ratio with a token budget instead of "
            "fixed-tile resize. Enabled by default for Nemotron6-MoE VLM providers. "
            "Pass --no-dynamic-resolution to disable."
        ),
    )
    provider.add_argument(
        "--dynamic-resolution-min-patches",
        type=int,
        default=4,
        help="Lower bound on per-image patch count under dynamic resolution.",
    )
    provider.add_argument(
        "--dynamic-resolution-max-patches",
        type=int,
        default=0,
        help="Upper bound on per-image patch count under dynamic resolution; 0 = uncapped.",
    )
    provider.add_argument("--freeze-lm", action="store_true")
    provider.add_argument("--freeze-vit", action="store_true")
    provider.add_argument("--freeze-projection", action="store_true")
    provider.add_argument(
        "--vision-projection-type",
        type=str,
        choices=["mlp", "affine"],
        default="affine",
        help="Projection module from frozen vision features to language hidden size.",
    )
    provider.add_argument(
        "--training-stage", choices=["stage1", "stage2", "stage3"], default=None
    )
    provider.add_argument("--fp32", action="store_true", help="Use float32 instead of bfloat16.")
    return parser


def prepare_model_provider_args(args: argparse.Namespace) -> None:
    """Apply the post-parse Nemotron preset + derived tokenizer/vision settings.

    Call after ``parse_args`` and before stock ``validate_args``.
    """
    apply_model_provider_defaults(args)
    apply_training_stage(args)
    resolve_image_token_id(args)
    args.vision_encoder_key = NEMOTRON_VISION_ENCODER_KEY
    args.vision_input_mode = "pixels" if is_nemotron_moe_vlm(args) else "hidden_states"


def apply_model_provider_defaults(args: argparse.Namespace) -> None:
    """Set the vision/data-path knobs the dataloaders read off ``args``.

    Architecture (num_layers, hidden_size, moe_*, mamba_*, hybrid pattern, ...)
    comes from stock args via ``core_transformer_config_from_args``; the run
    script passes those flags. This only sets the vision/data-path fields.
    """
    if not is_nemotron_moe_vlm(args):
        return

    args.pixel_shuffle = True
    args.disable_vision_class_token = True
    args.image_seq_length = NEMOTRON_IMAGE_SEQ_PER_TILE * args.num_image_tiles
    if getattr(args, "dynamic_resolution", None) is None:
        args.dynamic_resolution = True
    # Dynamic resolution patchifies natively (no fixed-tile resize / thumbnail).
    args.use_tiling = not args.dynamic_resolution
    args.use_thumbnail = not args.dynamic_resolution


def apply_training_stage(args: argparse.Namespace) -> None:
    """Set the freeze flags for the requested stage (the runtime reads the flags)."""
    if not is_nemotron_moe_vlm(args):
        return

    stage = args.training_stage or NEMOTRON_DEFAULT_STAGE
    if stage == "stage1":
        args.freeze_vit = True
        args.freeze_lm = True
    elif stage == "stage2":
        args.freeze_vit = True
    elif stage != "stage3":
        raise ValueError(f"unsupported Nemotron VLM training stage: {stage}")
    args.training_stage = stage


def resolve_image_token_id(args: argparse.Namespace) -> None:
    """Resolve image, pad, and vocab ids from the configured tokenizer."""
    if not is_nemotron_moe_vlm(args) or not getattr(args, "tokenizer_model", None):
        return

    from megatron.core.tokenizers.vision.libraries.multimodal_tokenizer import (
        MegatronMultimodalTokenizer,
    )

    tokenizer = MegatronMultimodalTokenizer(
        path=args.tokenizer_model,
        prompt_format=args.tokenizer_prompt_format,
        special_tokens=[args.image_token],
        image_tag_type=args.image_tag_type,
        force_system_message=args.force_system_message,
    )
    image_token_id = tokenizer.convert_tokens_to_ids(args.image_token)
    if image_token_id is None:
        raise RuntimeError(
            f"tokenizer at {args.tokenizer_model} did not produce an id for {args.image_token}"
        )
    args.image_token_id = int(image_token_id)
    if tokenizer.pad is not None:
        args.pad_token_id = int(tokenizer.pad)


def _vocab_size(args: argparse.Namespace) -> int:
    """Resolve the vocabulary size from stock args (``padded_vocab_size`` / ``vocab_size``)."""
    for attr in ("padded_vocab_size", "vocab_size"):
        value = getattr(args, attr, None)
        if value:
            return int(value)
    raise ValueError(
        "vocab size unresolved: set --vocab-size / a tokenizer, or padded_vocab_size"
    )


def validate_model_provider_args(args: argparse.Namespace) -> None:
    """Validate derived model-provider arguments.

    Call after stock ``validate_args`` so ``padded_vocab_size`` is populated.
    """
    vocab_size = _vocab_size(args)
    if not 0 <= args.image_token_id < vocab_size:
        raise ValueError("--image-token-id must be within the vocabulary")
    if not 0 <= args.pad_token_id < vocab_size:
        raise ValueError("--pad-token-id must be within the vocabulary")


def _pixel_shuffle_dynamic_res(x, imgs_sizes, patch_dim, scale_factor=0.5, version=2):
    """Pixel shuffle for dynamic resolution (variable tile sizes).

    Splits the packed sequence by per-tile lengths, applies pixel shuffle to each
    tile, then re-concatenates.
    """
    seq_lens = torch.prod(imgs_sizes // patch_dim, dim=-1)
    splits = torch.split(x, seq_lens.tolist(), dim=-2)

    out = []
    for i, sv in enumerate(splits):
        h = imgs_sizes[i][0] // patch_dim
        w = imgs_sizes[i][1] // patch_dim
        sv = sv.reshape(sv.shape[0], h, w, -1)

        n, h, w, c = sv.size()
        sv = sv.view(n, h, int(w * scale_factor), int(c / scale_factor))
        sv = sv.permute(0, 2, 1, 3).contiguous()
        sv = sv.view(
            n,
            int(w * scale_factor),
            int(h * scale_factor),
            int(c / (scale_factor * scale_factor)),
        )

        if version == 2:
            sv = sv.permute(0, 2, 1, 3).contiguous()

        sv = sv.reshape(sv.shape[0], -1, sv.shape[-1])
        out.append(sv)

    return torch.cat(out, dim=-2)


class RADIOEncoderWrapper(torch.nn.Module):
    """RADIO encoder wrapper matching the Nemotron6-MoE VLM provider."""

    def __init__(
        self,
        transformer_config: TransformerConfig,
        transformer_layer_spec: ModuleSpec,
        pg_collection: Optional[ProcessGroupCollection],
        img_h: int,
        img_w: int,
        patch_dim: int,
        class_token_len: int,
        drop_class_token: bool = True,
        apply_pixel_shuffle: bool = True,
        force_eval_mode: bool = False,
        dynamic_resolution: bool = False,
    ) -> None:
        super().__init__()
        self.class_token_len = class_token_len
        self.drop_class_token = drop_class_token
        self.apply_pixel_shuffle = apply_pixel_shuffle
        self.force_eval_mode = force_eval_mode
        self.dynamic_resolution = dynamic_resolution
        self.radio_model = RADIOViTModel(
            transformer_config=transformer_config,
            transformer_layer_spec=transformer_layer_spec,
            patch_dim=patch_dim,
            img_h=img_h,
            img_w=img_w,
            class_token_len=class_token_len,
            add_class_token=True,
            max_img_h=2048,
            max_img_w=2048,
            has_cpe=True,
            embedder_bias=False,
            dynamic_resolution=dynamic_resolution,
            force_eval_mode=force_eval_mode,
            pg_collection=pg_collection,
        )

    @property
    def config(self):
        """Expose the underlying RADIO config for DDP wrapping."""
        return self.radio_model.config

    def forward(
        self,
        x: torch.Tensor,
        imgs_sizes: Optional[torch.Tensor] = None,
        packed_seq_params=None,
    ) -> torch.Tensor:
        """Run RADIO, drop class tokens, and apply pixel shuffle."""
        context = torch.no_grad() if self.force_eval_mode else nullcontext()
        with context:
            x = x.to(dtype=self.radio_model.embedder.weight.dtype)
            embeddings = self.radio_model(
                x, imgs_sizes=imgs_sizes, packed_seq_params=packed_seq_params
            )
        if self.drop_class_token:
            if self.dynamic_resolution and imgs_sizes is not None and self.class_token_len > 0:
                # Class tokens are interleaved between tiles; build mask to remove them.
                remove_mask = torch.full(
                    (embeddings.shape[-2],), True, dtype=torch.bool, device=embeddings.device
                )
                patch_dim = self.radio_model.patch_dim
                if torch.is_tensor(imgs_sizes):
                    seq_lens = torch.prod(imgs_sizes // patch_dim, dim=-1)
                else:
                    seq_lens = torch.tensor(
                        [(h // patch_dim) * (w // patch_dim) for h, w in imgs_sizes]
                    )
                current_length = 0
                for sl in seq_lens:
                    remove_mask[current_length : current_length + self.class_token_len] = False
                    current_length += int(sl) + self.class_token_len
                embeddings = embeddings[:, remove_mask, :]
            else:
                embeddings = embeddings[:, self.class_token_len :, :]
        if self.apply_pixel_shuffle:
            if self.dynamic_resolution and imgs_sizes is not None:
                embeddings = _pixel_shuffle_dynamic_res(
                    embeddings, imgs_sizes, self.radio_model.patch_dim
                )
            else:
                embeddings = pixel_shuffle(embeddings, scale_factor=0.5)
        return embeddings

    def sharded_state_dict(self, prefix="", sharded_offsets=(), metadata=None):
        """Delegate checkpoint sharding to the wrapped RADIO model."""
        sharded_sd = {}
        for name, child in self.named_children():
            sharded_sd.update(
                sharded_state_dict_default(child, f"{prefix}{name}.", sharded_offsets, metadata)
            )
        return sharded_sd


def get_vision_encoder_module(args: argparse.Namespace, vision_submodule):
    """Return the provider-owned encoder module used for DDP config and freezing."""
    return vision_submodule.encoders[NEMOTRON_VISION_ENCODER_KEY]


def iter_vision_projection_modules(vision_submodule):
    """Return the provider-owned projection modules used for freeze-stage policy."""
    return iter(vision_submodule.input_projections)


def nemotron_projection_layer_spec() -> ModuleSpec:
    """Return the Nemotron VLM RADIO-to-language projector layer spec."""
    if TERowParallelLinear is None:
        raise RuntimeError("TERowParallelLinear is required")
    # MultimodalProjector's affine path builds fc1 with gather_output=True, which
    # TE column-parallel linears reject; use core ColumnParallelLinear for fc1.
    return ModuleSpec(
        module=MLP,
        submodules=MLPSubmodules(
            linear_fc1=ColumnParallelLinear, linear_fc2=TERowParallelLinear
        ),
    )


def _dtype(args: argparse.Namespace):
    """Resolve params/pipeline dtype: bf16 unless --fp32/--fp16."""
    bf16 = not getattr(args, "fp32", False) and not getattr(args, "fp16", False)
    return bf16, (torch.bfloat16 if bf16 else torch.float32)


def _base_config(args: argparse.Namespace) -> TransformerConfig:
    """Stock config from CLI args; the per-tower override helpers deepcopy this."""
    from megatron.training.argument_utils import core_transformer_config_from_args

    return core_transformer_config_from_args(args)


def nemotron_language_config(
    args: argparse.Namespace, tp_size: int, pp_size: int, ep_size: int, expt_tp_size: int
) -> TransformerConfig:
    """Nemotron6-MoE language config: stock from-args base + model-specific overrides."""
    config = deepcopy(_base_config(args))
    bf16, dtype = _dtype(args)
    # Code-only fields (not cleanly arg-expressible) + hetero parallelism pins.
    config.attention_backend = AttnBackend.flash
    config.use_cpu_initialization = True
    config.variable_seq_lengths = True
    config.cross_entropy_fusion_impl = "te"
    # moe_token_dispatcher_type / moe_flex_dispatcher_backend come from CLI flags
    # so the base config validates at construction (shared-expert-overlap requires
    # the alltoall/flex dispatcher); the run script defaults them from env.
    config.params_dtype = dtype
    config.pipeline_dtype = dtype
    config.bf16 = bf16
    config.expert_model_parallel_size = ep_size
    config.expert_tensor_parallel_size = expt_tp_size
    config.tensor_model_parallel_size = tp_size
    config.pipeline_model_parallel_size = pp_size
    config.sequence_parallel = tp_size > 1
    config.position_embedding_type = "none"
    config.seq_length = NEMOTRON_SEQ_LENGTH
    config.max_position_embeddings = NEMOTRON_SEQ_LENGTH
    return config


def require_per_token_loss(config: TransformerConfig) -> None:
    """The hetero MIMO loop scales both language and vision grads by real LM tokens."""
    if not config.calculate_per_token_loss:
        raise ValueError("hetero MIMO training requires calculate_per_token_loss=True")


def radio_vision_config(args: argparse.Namespace, tp_size: int, pp_size: int) -> TransformerConfig:
    """RADIO vision config: stock from-args base + RADIO-specific overrides."""
    config = deepcopy(_base_config(args))
    bf16, dtype = _dtype(args)
    config.num_layers = 32
    config.hidden_size = 1280
    config.num_attention_heads = 16
    config.kv_channels = 80
    config.num_query_groups = 16
    config.ffn_hidden_size = 5120
    config.gated_linear_unit = False
    config.activation_func = fast_gelu
    config.add_bias_linear = True
    config.add_qkv_bias = True
    config.normalization = "LayerNorm"
    config.layernorm_epsilon = 1.0e-6
    config.layernorm_zero_centered_gamma = False
    config.apply_rope_fusion = False
    config.qk_layernorm = False
    config.bias_activation_fusion = False
    config.bias_dropout_fusion = False
    config.attention_softmax_in_fp32 = True
    config.attention_dropout = 0.0
    config.hidden_dropout = 0.0
    config.mtp_num_layers = 0  # Trigger TransformerBlock's final_layernorm allocation.
    config.use_cpu_initialization = True
    _make_dense_non_hybrid(config)  # ViT inherits no MoE/Mamba/hybrid settings.
    config.params_dtype = dtype
    config.pipeline_dtype = dtype
    config.bf16 = bf16
    config.tensor_model_parallel_size = tp_size
    config.pipeline_model_parallel_size = pp_size
    config.sequence_parallel = False
    return config


def nemotron_projection_config(args: argparse.Namespace, tp_size: int) -> TransformerConfig:
    """RADIO-to-Nemotron projection config: stock from-args base + overrides."""
    config = deepcopy(_base_config(args))
    bf16, dtype = _dtype(args)
    config.num_layers = 1
    config.hidden_size = _llm_hidden_size(args)
    config.num_attention_heads = 1
    config.ffn_hidden_size = 4 * 5120
    config.bias_activation_fusion = False
    config.bias_dropout_fusion = False
    config.add_bias_linear = False
    config.activation_func = squared_relu
    config.normalization = "RMSNorm"
    config.use_cpu_initialization = True
    _make_dense_non_hybrid(config)  # Projection inherits no MoE/Mamba/hybrid settings.
    config.params_dtype = dtype
    config.pipeline_dtype = dtype
    config.bf16 = bf16
    config.tensor_model_parallel_size = tp_size
    config.sequence_parallel = False
    return config


def _llm_hidden_size(args: argparse.Namespace) -> int:
    """Language hidden size the projection maps into (from stock --hidden-size)."""
    return int(args.hidden_size)


def _make_dense_non_hybrid(config: TransformerConfig) -> None:
    """Strip language-only MoE/Mamba/hybrid settings inherited from the base config."""
    config.num_moe_experts = None
    config.moe_ffn_hidden_size = None
    config.moe_shared_expert_intermediate_size = None
    config.moe_grouped_gemm = False
    config.moe_router_fusion = False
    config.moe_permute_fusion = False
    config.moe_shared_expert_overlap = False
    config.is_hybrid_model = False
    config.use_fused_weighted_squared_relu = False


def language_model_spec(
    args: argparse.Namespace,
    pg_collection: Optional[ProcessGroupCollection],
    llm_grid: HyperCommGrid,
) -> ModuleSpec:
    """Create the language ``ModuleSpec`` for the local language grid.

    ``pg_collection`` is the per-module ProcessGroupCollection built by
    ``examples/mimo/training/topology.py`` (``None`` on ranks not in the language
    grid). ``llm_grid`` is the language ``HyperCommGrid`` used only for fallback
    dim sizes when a group is missing.
    """
    pp_pg = getattr(pg_collection, "pp", None) if pg_collection is not None else None
    tp_pg = getattr(pg_collection, "tp", None) if pg_collection is not None else None
    ep_pg = getattr(pg_collection, "ep", None) if pg_collection is not None else None
    expt_tp_pg = getattr(pg_collection, "expt_tp", None) if pg_collection is not None else None

    fallback_tp_size = get_grid_dim_size(llm_grid, "tp")
    pp_rank = get_group_rank_or(pp_pg)
    pp_size = get_group_size_or(pp_pg, get_grid_dim_size(llm_grid, "pp"))
    tp_size = get_group_size_or(tp_pg, fallback_tp_size)
    ep_size = get_group_size_or(ep_pg, getattr(args, "llm_ep", 1))
    expt_tp_size = get_group_size_or(
        expt_tp_pg, getattr(args, "llm_expt_tp", None) or 1
    )

    if is_nemotron_moe_vlm(args):
        config = nemotron_language_config(args, tp_size, pp_size, ep_size, expt_tp_size)
        require_per_token_loss(config)
        return ModuleSpec(
            module=MambaModel,
            params={
                "config": config,
                "mamba_stack_spec": mamba_stack_spec,
                "vocab_size": _vocab_size(args),
                "max_sequence_length": NEMOTRON_SEQ_LENGTH,
                "pre_process": pp_rank == 0,
                "post_process": pp_rank == pp_size - 1,
                "hybrid_layer_pattern": args.hybrid_layer_pattern,
                "position_embedding_type": "none",
                "share_embeddings_and_output_weights": False,
                "scatter_embedding_sequence_parallel": False,
                "pg_collection": pg_collection,
            },
        )

    # Non-Nemotron fallback (mock-style dense/MoE GPT). Uses stock args directly.
    num_experts = getattr(args, "num_experts", None)
    bf16 = not getattr(args, "fp32", False) and not getattr(args, "fp16", False)
    moe_kwargs = {}
    if num_experts:
        moe_kwargs = {
            "num_moe_experts": num_experts,
            "moe_router_topk": args.moe_router_topk,
            "moe_router_pre_softmax": args.moe_router_topk == 1,
            "expert_model_parallel_size": ep_size,
            "expert_tensor_parallel_size": expt_tp_size,
            "moe_grouped_gemm": getattr(args, "moe_grouped_gemm", False),
        }

    config = TransformerConfig(
        num_layers=args.num_layers,
        hidden_size=args.hidden_size,
        num_attention_heads=args.num_attention_heads,
        use_cpu_initialization=True,
        variable_seq_lengths=True,
        tensor_model_parallel_size=tp_size,
        pipeline_model_parallel_size=pp_size,
        pipeline_dtype=torch.bfloat16 if bf16 else torch.float32,
        bf16=bf16,
        calculate_per_token_loss=True,
        cross_entropy_loss_fusion=True,
        cross_entropy_fusion_impl="te",
        **moe_kwargs,
    )
    require_per_token_loss(config)
    return ModuleSpec(
        module=GPTModel,
        params={
            "config": config,
            "transformer_layer_spec": get_gpt_layer_with_transformer_engine_spec(
                num_experts=num_experts, moe_grouped_gemm=getattr(args, "moe_grouped_gemm", False)
            ),
            "vocab_size": _vocab_size(args),
            "max_sequence_length": args.seq_length,
            "pre_process": pp_rank == 0,
            "post_process": pp_rank == pp_size - 1,
            "pg_collection": pg_collection,
        },
    )


def vision_submodules_spec(
    args: argparse.Namespace,
    pg_collection: Optional[ProcessGroupCollection],
    encoder_grid: HyperCommGrid,
) -> ModuleSpec:
    """Create the vision ``ModuleSpec`` for the local encoder grid."""
    pp_pg = getattr(pg_collection, "pp", None) if pg_collection is not None else None
    tp_pg = getattr(pg_collection, "tp", None) if pg_collection is not None else None
    tp_size = get_group_size_or(tp_pg, get_grid_dim_size(encoder_grid, "tp"))
    pp_size = get_group_size_or(pp_pg, get_grid_dim_size(encoder_grid, "pp"))

    if not is_nemotron_moe_vlm(args):
        raise NotImplementedError(
            "vision_submodules_spec on stock args currently supports only the "
            "Nemotron6-MoE VLM providers; use the mock provider's own builder otherwise."
        )

    vision_config = radio_vision_config(args, tp_size, pp_size)
    vision_encoder_spec = ModuleSpec(
        module=RADIOEncoderWrapper,
        params={
            "transformer_config": vision_config,
            "transformer_layer_spec": get_vit_layer_with_transformer_engine_spec(),
            "pg_collection": pg_collection,
            "img_h": args.img_h,
            "img_w": args.img_w,
            "patch_dim": args.patch_dim,
            "class_token_len": args.class_token_len,
            "drop_class_token": True,
            "apply_pixel_shuffle": True,
            "force_eval_mode": args.freeze_vit,
            "dynamic_resolution": bool(getattr(args, "dynamic_resolution", False)),
        },
    )
    vision_projection_spec = ModuleSpec(
        module=MultimodalProjector,
        params={
            "config": nemotron_projection_config(args, tp_size),
            "submodules": nemotron_projection_layer_spec().submodules,
            "projector_type": args.vision_projection_type,
            "input_size": 5120,
            "tp_group": tp_pg if is_process_group_member(tp_pg) else None,
        },
    )
    return ModuleSpec(
        module=VisionModalitySubmodules,
        params={"pg_collection": pg_collection},
        submodules={
            "encoders": {NEMOTRON_VISION_ENCODER_KEY: vision_encoder_spec},
            "input_projections": [vision_projection_spec],
        },
    )
