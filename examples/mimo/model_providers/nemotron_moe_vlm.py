# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Nemotron6-MoE VLM model provider for hetero MIMO examples.

Declares the MIMO/vision provider args and builds the language ``MambaModel``
(hybrid Mamba/MoE) and the vision ``RADIOViTModel`` + ``MultimodalProjector``
``ModuleSpec`` s. Image-token / tokenizer / data-path knobs live with the data
pipeline, not here.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
from typing import Optional

from examples.mimo.model_providers.radio_encoder import (
    _base_config,
    _dtype,
    _make_dense_non_hybrid,
    add_radio_encoder_args,
    radio_vision_config,
    radio_vision_encoder_spec,
)
from examples.mimo.utils.hetero import (
    get_grid_dim_size,
    get_group_rank_or,
    get_group_size_or,
    is_process_group_member,
)
from megatron.core.activations import squared_relu
from megatron.core.hyper_comm_grid import HyperCommGrid
from megatron.core.models.mamba.mamba_layer_specs import mamba_stack_spec
from megatron.core.models.mamba.mamba_model import MambaModel
from megatron.core.models.mimo.submodules.vision import VisionModalitySubmodules
from megatron.core.models.vision.multimodal_projector import MultimodalProjector
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.tensor_parallel import ColumnParallelLinear
from megatron.core.transformer.enums import AttnBackend
from megatron.core.transformer.mlp import MLP, MLPSubmodules
from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core.transformer.transformer_config import TransformerConfig

try:
    from megatron.core.extensions.transformer_engine import TERowParallelLinear
except ImportError:  # pragma: no cover - TE always present in the CI container
    TERowParallelLinear = None

NEMOTRON_MODEL_PROVIDER = "nemotron-moe-vlm"
NEMOTRON_VISION_ENCODER_KEY = "radio_encoder"


def is_nemotron_moe_vlm(args: argparse.Namespace) -> bool:
    """Return whether the Nemotron6-MoE VLM provider is active."""
    return getattr(args, "model_provider", None) == NEMOTRON_MODEL_PROVIDER


def add_model_provider_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Register the model-provider args for hetero MIMO examples.

    Only the provider/vision knobs this PR consumes are declared here; stock
    ``arguments.py`` owns the ``TransformerConfig`` field flags and
    ``radio_encoder`` owns the RADIO-encoder knobs.
    """
    add_radio_encoder_args(parser)
    provider = parser.add_argument_group("mimo model provider")
    provider.add_argument(
        "--model-provider",
        choices=[NEMOTRON_MODEL_PROVIDER],
        default=NEMOTRON_MODEL_PROVIDER,
        help="Which MIMO model provider/preset to build.",
    )
    provider.add_argument(
        "--dynamic-resolution",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Patchify each image at its native aspect ratio with a token budget instead of "
            "fixed-tile resize. Pass --no-dynamic-resolution to disable."
        ),
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
    provider.add_argument("--fp32", action="store_true", help="Use float32 instead of bfloat16.")
    return parser


def _vocab_size(args: argparse.Namespace) -> int:
    """Resolve the vocabulary size from stock args (``padded_vocab_size`` / ``vocab_size``)."""
    for attr in ("padded_vocab_size", "vocab_size"):
        value = getattr(args, attr, None)
        if value:
            return int(value)
    raise ValueError(
        "vocab size unresolved: set --vocab-size / a tokenizer, or padded_vocab_size"
    )


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
    # seq_length / max_position_embeddings come from stock --seq-length /
    # --max-position-embeddings via the from-args base (no silent override).
    return config


def require_per_token_loss(config: TransformerConfig) -> None:
    """The hetero MIMO loop scales both language and vision grads by real LM tokens."""
    if not config.calculate_per_token_loss:
        raise ValueError("hetero MIMO training requires calculate_per_token_loss=True")


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

    config = nemotron_language_config(args, tp_size, pp_size, ep_size, expt_tp_size)
    require_per_token_loss(config)
    return ModuleSpec(
        module=MambaModel,
        params={
            "config": config,
            "mamba_stack_spec": mamba_stack_spec,
            "vocab_size": _vocab_size(args),
            "max_sequence_length": args.seq_length,
            "pre_process": pp_rank == 0,
            "post_process": pp_rank == pp_size - 1,
            "hybrid_layer_pattern": args.hybrid_layer_pattern,
            "position_embedding_type": "none",
            "share_embeddings_and_output_weights": False,
            "scatter_embedding_sequence_parallel": False,
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
    vision_encoder_spec = radio_vision_encoder_spec(args, vision_config, pg_collection)
    # projector_type="affine" uses only submodules.linear_fc1 (a single affine
    # projection); fc2 from MLPSubmodules is ignored unless type is "mlp".
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
