# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""MIMO provider pairing a hybrid language model with a 2D-RoPE ViT and an MLP projector.

The ViT follows the Pixtral design: RMSNorm, gated SiLU MLP, no biases, interleaved 2D RoPE,
a pre-transformer norm, and a 2x2 patch merger, followed by a two-layer GELU MLP projector to
the language hidden size. Its sizes come from the --mimo-vision-* arguments.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
from functools import partial
from typing import TYPE_CHECKING

import torch

from examples.mimo.model_providers import MimoProvider
from examples.mimo.model_providers.nemotron_moe_vlm import language_model_spec
from examples.mimo.model_providers.radio_encoder import (
    _base_config,
    _disable_gtp,
    _dtype,
    _make_dense_non_hybrid,
)
from examples.mimo.utils.hetero import get_grid_dim_size
from megatron.core.extensions.transformer_engine import TEColumnParallelLinear, TERowParallelLinear
from megatron.core.hyper_comm_grid import HyperCommGrid
from megatron.core.models.mimo.config.role import MIMO_LANGUAGE_MODULE_KEY
from megatron.core.models.mimo.submodules.vision import VisionModalitySubmodules
from megatron.core.models.vision.multimodal_projector import MultimodalProjector
from megatron.core.models.vision.vit_layer_specs import get_vit_layer_with_transformer_engine_spec
from megatron.core.models.vision.vit_model import ViTModel
from megatron.core.pipeline_parallel.multimodule_communicator import MultiModulePipelineCommunicator
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.transformer.mlp import MLPSubmodules
from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.utils import get_pg_size

if TYPE_CHECKING:
    from examples.mimo.training.topology import HeteroTopology

ROPE2D_VIT_VLM_MODEL_PROVIDER = "rope2d-vit-vlm"
# The ViT merges 2x2 patches into one token before the projection.
_SPATIAL_MERGE_SIZE = 2
_REQUIRED_VISION_ARGS = (
    "mimo_vision_num_layers",
    "mimo_vision_hidden_size",
    "mimo_vision_ffn_hidden_size",
    "mimo_vision_num_attention_heads",
)


def _unfused_fast_gelu(x: torch.Tensor) -> torch.Tensor:
    """Fast GELU without the global MCore torch.compile decorator."""
    return 0.5 * x * (1.0 + torch.tanh(x * 0.7978845608 * (1.0 + 0.044715 * x * x)))


def _check_vision_args(args: argparse.Namespace) -> None:
    missing = [name for name in _REQUIRED_VISION_ARGS if getattr(args, name, None) is None]
    if missing:
        flags = ", ".join("--" + name.replace("_", "-") for name in missing)
        raise ValueError(f"rope2d-vit-vlm requires {flags}")
    if args.mimo_vision_hidden_size % args.mimo_vision_num_attention_heads:
        raise ValueError(
            "--mimo-vision-hidden-size must be divisible by --mimo-vision-num-attention-heads"
        )


class Rope2dViTModel(ViTModel):
    """ViT adapted to the MIMO encoder contract.

    MegatronModule.sharded_state_dict falls back to the global tensor-parallel group when a
    module has no tp_group; under hetero MIMO the encoder grid's group differs from it.
    MIMO batches pass images as x (like for RADIO), while ViTModel uses pixel_values.
    """

    def __init__(self, *args, pg_collection=None, **kwargs):
        super().__init__(*args, pg_collection=pg_collection, **kwargs)
        if pg_collection is not None:
            self.tp_group = pg_collection.tp

    def forward(self, x, imgs_sizes=None, packed_seq_params=None):
        # MIMO moves the whole batch to CUDA; ViTModel needs the per-image patch grid on the host.
        if torch.is_tensor(imgs_sizes) and imgs_sizes.is_cuda:
            imgs_sizes = imgs_sizes.cpu()
        return super().forward(x, imgs_sizes=imgs_sizes, packed_seq_params=packed_seq_params)


def _finalize_tower_config(config: TransformerConfig, args, tp_size: int, pp_size: int) -> None:
    """Apply runtime policy shared by the ViT and projector: dtype and parallelism."""
    bf16, dtype = _dtype(args)
    _make_dense_non_hybrid(config)
    config.params_dtype = dtype
    config.pipeline_dtype = dtype
    config.bf16 = bf16
    config.tensor_model_parallel_size = tp_size
    config.pipeline_model_parallel_size = pp_size
    config.context_parallel_size = 1
    _disable_gtp(config)
    config.sequence_parallel = False


def rope2d_vision_config(
    args: argparse.Namespace, tp_size: int, pp_size: int
) -> TransformerConfig:
    """Build the ViT TransformerConfig from the --mimo-vision-* sizes."""
    config = deepcopy(_base_config(args))
    config.num_layers = args.mimo_vision_num_layers
    config.hidden_size = args.mimo_vision_hidden_size
    config.ffn_hidden_size = args.mimo_vision_ffn_hidden_size
    config.num_attention_heads = args.mimo_vision_num_attention_heads
    config.num_query_groups = args.mimo_vision_num_attention_heads
    config.kv_channels = args.mimo_vision_hidden_size // args.mimo_vision_num_attention_heads
    config.normalization = "RMSNorm"
    config.layernorm_epsilon = 1.0e-5
    config.activation_func = torch.nn.functional.silu
    config.gated_linear_unit = True
    config.add_bias_linear = False
    config.add_qkv_bias = False
    config.rotary_interleaved = True
    config.qk_layernorm = False
    config.layernorm_zero_centered_gamma = False
    config.hidden_dropout = 0.0
    config.attention_dropout = 0.0
    config.bias_activation_fusion = False
    config.bias_dropout_fusion = False
    config.attention_softmax_in_fp32 = True
    config.apply_rope_fusion = False
    config.mtp_num_layers = None
    if args.mimo_vision_encoder_attention_backend is not None:
        config.attention_backend = args.mimo_vision_encoder_attention_backend
    if args.mimo_vision_encoder_flash_attention_version is not None:
        config.flash_attention_version = args.mimo_vision_encoder_flash_attention_version
    _finalize_tower_config(config, args, tp_size, pp_size)
    return config


def rope2d_projection_config(args: argparse.Namespace, tp_size: int) -> TransformerConfig:
    """Build W2(FastGELU(W1(z))) with both hidden widths matching the language model."""
    config = deepcopy(_base_config(args))
    config.num_layers = 1
    config.hidden_size = int(args.hidden_size)
    config.ffn_hidden_size = int(args.hidden_size)
    config.num_attention_heads = 1
    config.activation_func = _unfused_fast_gelu
    config.gated_linear_unit = False
    config.bias_activation_fusion = False
    config.bias_dropout_fusion = False
    config.add_bias_linear = False
    _finalize_tower_config(config, args, tp_size, 1)
    return config


def rope2d_vision_submodules_spec(
    args: argparse.Namespace,
    pg_collection: ProcessGroupCollection | None,
    encoder_grid: HyperCommGrid,
) -> ModuleSpec:
    """Build the ViT encoder and its projection to the language hidden size."""
    if args.mimo_run_input_projections_on_llm_ranks:
        raise ValueError("rope2d-vit-vlm runs its vision projection on the encoder ranks")
    if pg_collection is None:
        tp_size = get_grid_dim_size(encoder_grid, "tp")
        pp_size = get_grid_dim_size(encoder_grid, "pp")
    else:
        assert all(
            getattr(pg_collection, name, None) is not None for name in ("pp", "tp")
        ), "encoder pg_collection is missing the required pp/tp group"
        tp_size = get_pg_size(pg_collection.tp)
        pp_size = get_pg_size(pg_collection.pp)

    vision_config = rope2d_vision_config(args, tp_size, pp_size)
    encoder = ModuleSpec(
        module=Rope2dViTModel,
        params={
            "transformer_config": vision_config,
            "transformer_layer_spec": get_vit_layer_with_transformer_engine_spec(),
            "patch_dim": args.patch_dim,
            "img_h": args.img_h,
            "img_w": args.img_w,
            "add_class_token": False,
            "class_token_len": 0,
            "ln_pre": True,
            "pos_emb_type": "rope2d",
            "use_merger": True,
            "spatial_merge_size": _SPATIAL_MERGE_SIZE,
            "pg_collection": pg_collection,
        },
    )
    projection = ModuleSpec(
        module=MultimodalProjector,
        params={
            "config": rope2d_projection_config(args, tp_size),
            "submodules": MLPSubmodules(
                linear_fc1=TEColumnParallelLinear, linear_fc2=TERowParallelLinear
            ),
            "projector_type": "mlp",
            "input_size": vision_config.hidden_size,
            "pg_collection": pg_collection,
        },
    )
    return ModuleSpec(
        module=VisionModalitySubmodules,
        params={"pg_collection": pg_collection},
        submodules={
            "encoders": {args.mimo_vision_encoder_name: encoder},
            "input_projections": [projection],
        },
    )


def _bridge_recv_shape(batch: dict, encoder_name: str, hidden_size: int) -> tuple[int, int]:
    """Derive the local projected-vision receive shape from precomputed token indices."""
    return batch["modality_token_indices"][encoder_name].numel(), hidden_size


def build_rope2d_vit_vlm_communicator(
    args: argparse.Namespace, topology: "HeteroTopology", *, encoder_name: str
) -> MultiModulePipelineCommunicator:
    """Wire the vision encoder to the language grid."""
    language_grid = topology.grids[MIMO_LANGUAGE_MODULE_KEY]
    language_config = language_model_spec(args, None, language_grid).params["config"]
    if encoder_name not in topology.grids:
        # LLM-only topology has no encoder or cross-module bridge metadata.
        return MultiModulePipelineCommunicator(
            topology.grids,
            {MIMO_LANGUAGE_MODULE_KEY: []},
            language_config,
            dim_mapping={"s": 0, "h": 2, "b": 1},
        )
    return MultiModulePipelineCommunicator(
        topology.grids,
        {encoder_name: [MIMO_LANGUAGE_MODULE_KEY], MIMO_LANGUAGE_MODULE_KEY: []},
        language_config,
        dim_mapping={"s": 0, "h": 2, "b": 1},
        module_output_ndim={encoder_name: 2},
        bridge_comm_dtypes={encoder_name: language_config.params_dtype},
        bridge_recv_shape_fns=(
            {
                encoder_name: partial(
                    _bridge_recv_shape,
                    encoder_name=encoder_name,
                    hidden_size=int(language_config.hidden_size),
                )
            }
            if args.mimo_bridge_skip_shape_exchange
            else None
        ),
    )


def rope2d_vit_vlm_provider(args: argparse.Namespace) -> MimoProvider:
    """Return the provider for the ViT sized by the --mimo-vision-* arguments."""
    _check_vision_args(args)
    name = args.mimo_vision_encoder_name
    return MimoProvider(
        encoder_module_names=(name,),
        language_spec=language_model_spec,
        encoder_specs={name: rope2d_vision_submodules_spec},
        special_token_ids=lambda args: {name: args.image_token_id},
        build_communicator=partial(build_rope2d_vit_vlm_communicator, encoder_name=name),
    )
