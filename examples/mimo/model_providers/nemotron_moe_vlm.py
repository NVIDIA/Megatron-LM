# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Nemotron6-MoE VLM model provider for hetero MIMO examples (stock-args edition).

This is PR-E1 of the NMFW-516 hetero-MIMO upstreaming effort. It ports the
prototype provider (sasatheesh/pre-vlm-07) onto Megatron's *stock* argument
system (``megatron.training.arguments``).

The crucial difference from the prototype: stock ``arguments.py`` already
auto-generates CLI flags for every ``TransformerConfig`` dataclass field
(``--hidden-size``, ``--num-layers``, ``--num-attention-heads``,
``--num-query-groups``, ``--ffn-hidden-size``, ``--kv-channels``,
``--moe-router-topk``, ``--moe-grouped-gemm``,
``--moe-shared-expert-intermediate-size``, ``--moe-ffn-hidden-size``, etc.) via
``ArgumentGroupFactory(TransformerConfig, ...)``, plus ``--num-experts``,
``--seq-length``, ``--max-position-embeddings``, ``--normalization``,
``--bf16``/``--fp16``, ``--seed``, ``--micro-batch-size``, ``--img-h``,
``--img-w``, ``--patch-dim``, ``--tokenizer-model``, ``--tokenizer-type``,
``--squared-relu``, ``--hybrid-layer-pattern``, etc.

Therefore this module:
  * declares ONLY genuinely-new MIMO/vision/provider args via
    :func:`add_model_provider_args` (suitable as a stock ``extra_args_provider``);
  * applies the ``nemotron-moe-vlm-20l`` architecture preset *post-parse* via
    :func:`prepare_model_provider_args`, with an explicit respect-or-override
    policy (see module docstring table in the PR description); and
  * builds the language ``MambaModel`` (hybrid Mamba/MoE) and the vision
    ``RADIOViTModel`` + ``MultimodalProjector`` ``ModuleSpec`` s, ported faithfully
    from the prototype.

NOTE on the preset: rather than reconstructing every architecture knob from
hard-coded constants the way the prototype's ``apply_model_provider_defaults``
does, the build functions here construct the canonical Nemotron config directly
(see :func:`nemotron_language_config`). The post-parse preset only *force-sets*
the small set of args that downstream code (data loaders, validate, the MIMO
runtime) reads off ``args`` directly, and otherwise *fills* user-omitted
architecture args so an ``--print-args`` dump is faithful.
"""

from __future__ import annotations

import argparse
import os
from contextlib import nullcontext
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
    from megatron.core.extensions.transformer_engine import (
        TEColumnParallelLinear,
        TELayerNormColumnParallelLinear,
        TERowParallelLinear,
    )
except ImportError:  # pragma: no cover - TE always present in the CI container
    TEColumnParallelLinear = None
    TELayerNormColumnParallelLinear = None
    TERowParallelLinear = None

MOCK_MODEL_PROVIDER = "mock"
NEMOTRON_20L_MODEL_PROVIDER = "nemotron-moe-vlm-20l"
NEMOTRON_54L_MODEL_PROVIDER = "nemotron-moe-vlm-54l"
NEMOTRON_20L_IMAGE_SEQ_PER_TILE = 256
NEMOTRON_20L_MAX_NUM_TILES = 12
NEMOTRON_20L_DEFAULT_STAGE = "stage2"
NEMOTRON_VISION_ENCODER_KEY = "radio_encoder"

# Canonical Nemotron6-MoE architecture (the preset). These are the values the
# build functions construct directly and the preset force-fills onto ``args``.
_NEMOTRON_HIDDEN_SIZE = 2688
_NEMOTRON_NUM_ATTENTION_HEADS = 32
_NEMOTRON_NUM_QUERY_GROUPS = 8
_NEMOTRON_FFN_HIDDEN_SIZE = 1856
_NEMOTRON_KV_CHANNELS = 128
_NEMOTRON_NUM_MOE_EXPERTS = 128
_NEMOTRON_MOE_ROUTER_TOPK = 6
_NEMOTRON_MOE_SHARED_EXPERT = 3712
_NEMOTRON_SEQ_LENGTH = 8192
_NEMOTRON_20L_LAYERS = 20
_NEMOTRON_54L_LAYERS = 54
_NEMOTRON_20L_HYBRID_PATTERN = "MEMEM*EMEMEM*EMEMEM*"
_NEMOTRON_54L_HYBRID_PATTERN = (
    "MEMEM*EMEM*EMEM*EMEM*EMEMEM*EMEMEM*EMEMEM*EMEMEM*EMEME"
)


# ---------------------------------------------------------------------------
# Small process-group / grid helpers.
#
# The prototype imported these from ``examples.mimo.utils.hetero``, which is not
# present on main yet (it arrives with a later hetero PR). They are vendored
# here so this provider has no dependency outside the model_providers package.
# ---------------------------------------------------------------------------
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


def is_nemotron_20l(args: argparse.Namespace) -> bool:
    """Return whether the Nemotron6-MoE VLM 20L provider is active."""
    return getattr(args, "model_provider", None) == NEMOTRON_20L_MODEL_PROVIDER


def is_nemotron_moe_vlm(args: argparse.Namespace) -> bool:
    """Return whether a Nemotron6-MoE VLM provider is active."""
    return getattr(args, "model_provider", None) in (
        NEMOTRON_20L_MODEL_PROVIDER,
        NEMOTRON_54L_MODEL_PROVIDER,
    )


def add_model_provider_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Register *new* model-provider args for hetero MIMO examples.

    Stock ``arguments.py`` already owns every ``TransformerConfig`` field flag
    plus ``--num-experts``, ``--seq-length``, ``--max-position-embeddings``,
    ``--img-h``, ``--img-w``, ``--patch-dim``, ``--tokenizer-model``,
    ``--tokenizer-type``, ``--squared-relu``, ``--hybrid-layer-pattern``, etc.
    Declaring those here would raise an argparse "conflicting option" error.
    Only genuinely-new MIMO/vision/provider knobs are added below.

    Designed to be passed straight to stock ``pretrain(extra_args_provider=...)``;
    returns the parser per the stock hook contract.
    """
    provider = parser.add_argument_group("mimo model provider")
    # NB: stock train.py already declares --model-provider as a free-form str.
    # If this provider is wired in as the *only* extra_args_provider it must own
    # the flag; when composed with the existing MIMO add_mimo_args the caller
    # must register exactly one. We expose the choices-constrained variant and
    # leave de-duplication to the entrypoint (see open questions).
    provider.add_argument(
        "--model-provider",
        choices=[MOCK_MODEL_PROVIDER, NEMOTRON_20L_MODEL_PROVIDER, NEMOTRON_54L_MODEL_PROVIDER],
        default=MOCK_MODEL_PROVIDER,
        help="Which MIMO model provider/preset to build.",
    )
    # Vision / MIMO-specific knobs (none of these exist as stock args).
    provider.add_argument("--image-seq-length", type=int, default=None,
                          help="Total image tokens emitted by the encoder per sample.")
    provider.add_argument("--image-token-id", type=int, default=511,
                          help="Vocab id of the placeholder image token.")
    provider.add_argument("--pad-token-id", type=int, default=0)
    provider.add_argument("--image-token", type=str, default="<image>")
    provider.add_argument("--tokenizer-prompt-format", type=str, default="nemotron6-moe")
    provider.add_argument("--image-tag-type", type=str, default="")
    provider.add_argument("--force-system-message", action="store_true")
    # NB: --moe-router-force-load-balancing is auto-generated by stock arguments.py
    # from the TransformerConfig field; redeclaring it raises an argparse conflict.
    # nemotron_language_config reads it via getattr(args, "moe_router_force_load_balancing").
    provider.add_argument("--class-token-len", type=int, default=8)
    provider.add_argument(
        "--num-image-tiles",
        "--max-num-tiles",
        dest="num_image_tiles",
        type=int,
        default=NEMOTRON_20L_MAX_NUM_TILES,
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
    # Stock declares --bf16/--fp16 but not --fp32; the provider/data path uses fp32
    # as the "force full precision" toggle (matches the prototype). Default bf16.
    provider.add_argument("--fp32", action="store_true", help="Use float32 instead of bfloat16.")
    # NB: --llm-ep / --llm-expt-tp are *topology* knobs and are declared by the
    # hetero grid-arg provider (examples/mimo/training/args.py::add_hetero_grid_args),
    # not here -- declaring them in both groups would raise an argparse
    # "conflicting option" error when the two providers compose. The build
    # functions below still read them via getattr(args, "llm_ep"/"llm_expt_tp", ...)
    # fallbacks, which tolerate their absence (default 1 / None).
    return parser


def prepare_model_provider_args(args: argparse.Namespace) -> None:
    """Apply the post-parse Nemotron preset + derived tokenizer/vision settings.

    Call this AFTER ``parse_args`` and BEFORE stock ``validate_args`` so the
    preset-derived sizes flow into validation. See module docstring for the
    respect-or-override policy.
    """
    apply_model_provider_defaults(args)
    apply_training_stage(args)
    resolve_image_token_id(args)
    args.vision_encoder_key = NEMOTRON_VISION_ENCODER_KEY
    args.vision_input_mode = "pixels" if is_nemotron_moe_vlm(args) else "hidden_states"


def _force_set(args: argparse.Namespace, name: str, value) -> None:
    """Force-override a stock arg the runtime/dataloaders read off ``args``."""
    setattr(args, name, value)


def _fill_if_default(args: argparse.Namespace, name: str, value, stock_default) -> None:
    """Fill an architecture arg only when the user left the stock default.

    This keeps an ``--print-args`` dump faithful to the constructed config while
    still respecting an explicit user override (which would then mismatch the
    hard-coded build config -- see :func:`validate_model_provider_args`).
    """
    current = getattr(args, name, stock_default)
    if current == stock_default:
        setattr(args, name, value)


def apply_model_provider_defaults(args: argparse.Namespace) -> None:
    """Apply Nemotron6-MoE VLM architecture preset onto stock args.

    Force-overrides (runtime/dataloader/validate read these directly):
        image_seq_length, pixel_shuffle, disable_vision_class_token,
        use_tiling, use_thumbnail, dynamic_resolution, num_layers,
        hybrid_layer_pattern.

    Fill-if-default (kept faithful for arg dumps; build funcs use constants):
        hidden_size, num_attention_heads, num_query_groups, ffn_hidden_size,
        kv_channels, num_experts, moe_router_topk, moe_grouped_gemm,
        moe_shared_expert_intermediate_size, seq_length, max_position_embeddings.
    """
    if not is_nemotron_moe_vlm(args):
        return

    num_layers = (
        _NEMOTRON_54L_LAYERS
        if args.model_provider == NEMOTRON_54L_MODEL_PROVIDER
        else _NEMOTRON_20L_LAYERS
    )
    hybrid_pattern = (
        _NEMOTRON_54L_HYBRID_PATTERN
        if args.model_provider == NEMOTRON_54L_MODEL_PROVIDER
        else _NEMOTRON_20L_HYBRID_PATTERN
    )

    # --- Force-overrides --------------------------------------------------
    _force_set(args, "num_layers", num_layers)
    _force_set(args, "hybrid_layer_pattern", hybrid_pattern)
    _force_set(args, "pixel_shuffle", True)
    _force_set(args, "disable_vision_class_token", True)
    _force_set(
        args,
        "image_seq_length",
        NEMOTRON_20L_IMAGE_SEQ_PER_TILE * args.num_image_tiles,
    )
    if getattr(args, "dynamic_resolution", None) is None:
        args.dynamic_resolution = True
    if args.dynamic_resolution:
        # Dynamic-resolution strategy reads use_thumbnail inside
        # DynamicResolutionImageTilingStrategy and emits an extra thumbnail tile
        # when True. use_tiling is inert in this branch; pin both False for
        # args-dump parity (matches the prototype).
        _force_set(args, "use_tiling", False)
        _force_set(args, "use_thumbnail", False)
    else:
        _force_set(args, "use_tiling", True)
        _force_set(args, "use_thumbnail", True)

    # --- Fill-if-default --------------------------------------------------
    # Stock dataclass defaults for these architecture fields. The build
    # functions construct the canonical config directly from constants, so
    # these fills exist only so an args dump matches the built model and so
    # downstream tokenizer/data code sees consistent sizes.
    _fill_if_default(args, "hidden_size", _NEMOTRON_HIDDEN_SIZE, None)
    _fill_if_default(args, "num_attention_heads", _NEMOTRON_NUM_ATTENTION_HEADS, None)
    _fill_if_default(args, "num_query_groups", _NEMOTRON_NUM_QUERY_GROUPS, None)
    _fill_if_default(args, "ffn_hidden_size", _NEMOTRON_FFN_HIDDEN_SIZE, None)
    _fill_if_default(args, "kv_channels", _NEMOTRON_KV_CHANNELS, None)
    _fill_if_default(args, "num_experts", _NEMOTRON_NUM_MOE_EXPERTS, None)
    _fill_if_default(args, "moe_router_topk", _NEMOTRON_MOE_ROUTER_TOPK, 2)
    _fill_if_default(args, "moe_grouped_gemm", True, False)
    _fill_if_default(
        args,
        "moe_shared_expert_intermediate_size",
        _NEMOTRON_MOE_SHARED_EXPERT,
        None,
    )
    _fill_if_default(args, "seq_length", _NEMOTRON_SEQ_LENGTH, None)
    _fill_if_default(args, "max_position_embeddings", _NEMOTRON_SEQ_LENGTH, None)


def apply_training_stage(args: argparse.Namespace) -> None:
    """Apply stage-specific freeze flags for the Nemotron VLM recipe."""
    if not is_nemotron_moe_vlm(args):
        return

    stage = args.training_stage or NEMOTRON_20L_DEFAULT_STAGE
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
    # Stock derives padded_vocab_size from the tokenizer at validate time; we
    # only need padded_vocab_size for the build functions, read lazily below.


def _vocab_size(args: argparse.Namespace) -> int:
    """Resolve the vocabulary size from stock args.

    Stock sets ``padded_vocab_size`` during ``validate_args`` from the tokenizer
    (or ``--vocab-size`` when provided via tokenizer-type=NullTokenizer). The
    build functions consume this; tests may set it directly.
    """
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
    tile, then re-concatenates. Vendored from the prototype to avoid touching the
    upstream-owned ``llava_model.py``.
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
            pg_collection=pg_collection,
        )
        if self.force_eval_mode:
            self.radio_model.eval()

    def train(self, mode: bool = True):
        """Keep frozen RADIO in eval mode while allowing the projection to train."""
        super().train(mode)
        if self.force_eval_mode:
            self.radio_model.eval()
        return self

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
    # MultimodalProjector's "affine" path instantiates linear_fc1 with
    # gather_output=True, which TE column-parallel linears reject. Use the core
    # ColumnParallelLinear for fc1 (matches pre-vlm-07, whose MultimodalProjector
    # hardcoded ColumnParallelLinear for the affine projector).
    return ModuleSpec(
        module=MLP,
        submodules=MLPSubmodules(
            linear_fc1=ColumnParallelLinear, linear_fc2=TERowParallelLinear
        ),
    )


def nemotron_language_config(
    args: argparse.Namespace, tp_size: int, pp_size: int, ep_size: int, expt_tp_size: int
) -> TransformerConfig:
    """Build the Nemotron6-MoE language TransformerConfig (canonical preset)."""
    bf16 = not getattr(args, "fp32", False) and not getattr(args, "fp16", False)
    dtype = torch.bfloat16 if bf16 else torch.float32
    config = TransformerConfig(
        num_layers=_NEMOTRON_54L_LAYERS
        if args.model_provider == NEMOTRON_54L_MODEL_PROVIDER
        else _NEMOTRON_20L_LAYERS,
        hidden_size=_NEMOTRON_HIDDEN_SIZE,
        num_attention_heads=_NEMOTRON_NUM_ATTENTION_HEADS,
        attention_backend=AttnBackend.flash,
        num_query_groups=_NEMOTRON_NUM_QUERY_GROUPS,
        ffn_hidden_size=_NEMOTRON_FFN_HIDDEN_SIZE,
        kv_channels=_NEMOTRON_KV_CHANNELS,
        activation_func=squared_relu,
        gated_linear_unit=False,
        attention_dropout=0.0,
        hidden_dropout=0.0,
        normalization="RMSNorm",
        add_bias_linear=False,
        init_method_std=0.0173,
        use_cpu_initialization=True,
        variable_seq_lengths=True,
        tensor_model_parallel_size=tp_size,
        pipeline_model_parallel_size=pp_size,
        expert_model_parallel_size=ep_size,
        expert_tensor_parallel_size=expt_tp_size,
        sequence_parallel=tp_size > 1,
        params_dtype=dtype,
        pipeline_dtype=dtype,
        bf16=bf16,
        calculate_per_token_loss=True,
        cross_entropy_loss_fusion=True,
        cross_entropy_fusion_impl="te",
        bias_activation_fusion=False,
        masked_softmax_fusion=True,
        persist_layer_norm=True,
        bias_dropout_fusion=True,
        moe_ffn_hidden_size=_NEMOTRON_FFN_HIDDEN_SIZE,
        num_moe_experts=_NEMOTRON_NUM_MOE_EXPERTS,
        moe_router_topk=_NEMOTRON_MOE_ROUTER_TOPK,
        moe_grouped_gemm=True,
        moe_router_score_function="sigmoid",
        moe_router_topk_scaling_factor=2.5,
        moe_router_enable_expert_bias=True,
        moe_router_dtype="fp32",
        moe_router_load_balancing_type="seq_aux_loss",
        moe_router_force_load_balancing=getattr(args, "moe_router_force_load_balancing", False),
        moe_router_fusion=True,
        moe_aux_loss_coeff=1.0e-4,
        moe_shared_expert_intermediate_size=_NEMOTRON_MOE_SHARED_EXPERT,
        moe_shared_expert_overlap=True,
        moe_token_dispatcher_type=os.environ.get("MOE_TOKEN_DISPATCHER_TYPE", "alltoall"),
        moe_flex_dispatcher_backend=os.environ.get("MOE_FLEX_DISPATCHER_BACKEND", "deepep"),
        moe_permute_fusion=True,
        use_fused_weighted_squared_relu=True,
        is_hybrid_model=True,
        mamba_num_heads=64,
        mamba_head_dim=64,
        mamba_num_groups=8,
        mamba_state_dim=128,
        linear_conv_kernel_dim=4,
    )
    config.position_embedding_type = "none"
    config.seq_length = _NEMOTRON_SEQ_LENGTH
    config.max_position_embeddings = _NEMOTRON_SEQ_LENGTH
    mtp_layers = int(getattr(args, "mtp_num_layers", 0) or 0)
    if mtp_layers > 0:
        config.mtp_num_layers = mtp_layers
        config.mtp_loss_scaling_factor = float(getattr(args, "mtp_loss_scaling_factor", 0.1))
        if getattr(args, "mtp_use_repeated_layer", False):
            config.mtp_use_repeated_layer = True
    return config


def require_per_token_loss(config: TransformerConfig) -> None:
    """The hetero MIMO loop scales both language and vision grads by real LM tokens."""
    if not config.calculate_per_token_loss:
        raise ValueError("hetero MIMO training requires calculate_per_token_loss=True")


def radio_vision_config(args: argparse.Namespace, tp_size: int, pp_size: int) -> TransformerConfig:
    """Build the RADIO vision TransformerConfig from the 20L reference provider."""
    bf16 = not getattr(args, "fp32", False) and not getattr(args, "fp16", False)
    dtype = torch.bfloat16 if bf16 else torch.float32
    config = TransformerConfig(
        num_layers=32,
        hidden_size=1280,
        num_attention_heads=16,
        use_cpu_initialization=True,
        tensor_model_parallel_size=tp_size,
        pipeline_model_parallel_size=pp_size,
        params_dtype=dtype,
        pipeline_dtype=dtype,
        bf16=bf16,
    )
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
    # Trigger TransformerBlock's final_layernorm allocation.
    config.mtp_num_layers = 0
    return config


def nemotron_projection_config(args: argparse.Namespace, tp_size: int) -> TransformerConfig:
    """Build the RADIO-to-Nemotron projection config."""
    bf16 = not getattr(args, "fp32", False) and not getattr(args, "fp16", False)
    dtype = torch.bfloat16 if bf16 else torch.float32
    config = TransformerConfig(
        num_layers=1,
        hidden_size=_NEMOTRON_HIDDEN_SIZE,
        num_attention_heads=1,
        use_cpu_initialization=True,
        params_dtype=dtype,
        pipeline_dtype=dtype,
        bf16=bf16,
    )
    config.tensor_model_parallel_size = tp_size
    config.ffn_hidden_size = 4 * 5120
    config.bias_activation_fusion = False
    config.bias_dropout_fusion = False
    config.add_bias_linear = False
    config.activation_func = squared_relu
    config.normalization = "RMSNorm"
    return config


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
                "max_sequence_length": _NEMOTRON_SEQ_LENGTH,
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
