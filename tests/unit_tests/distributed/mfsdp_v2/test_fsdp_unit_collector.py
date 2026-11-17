# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Default FSDP unit selection for real MCore layers."""

from functools import partial

import pytest
import torch
from torch import nn

from megatron.core.distributed.fsdp.fsdp_unit_collector import FsdpUnitCollector
from megatron.core.models.gpt.gpt_layer_specs import (
    get_gpt_decoder_block_spec,
    get_gpt_decoder_layer_specs,
    get_gpt_layer_with_transformer_engine_spec,
    get_gpt_mtp_block_spec,
)
from megatron.core.models.gpt.gpt_model import GPTModel
from megatron.core.models.hybrid.hybrid_layer_specs import hybrid_stack_spec
from megatron.core.models.hybrid.hybrid_model import HybridModel
from megatron.core.models.hybrid.shortcut_block import ShortcutMoEBlock
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.ssm.mamba_layer import MambaLayer
from megatron.core.ssm.mlp_layer_config import MLPLayerConfig
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.attention_layer_config import AttentionLayerConfig
from megatron.core.transformer.enums import AttnBackend
from megatron.core.transformer.moe.experts import SequentialMLP, TEGroupedMLP
from megatron.core.transformer.multi_token_prediction import MultiTokenPredictionLayer
from megatron.core.transformer.transformer_block import TransformerBlock
from megatron.core.transformer.transformer_config import MLATransformerConfig, TransformerConfig
from megatron.core.transformer.transformer_layer import TransformerLayer
from tests.unit_tests.distributed.mfsdp_v2.mcore_utils import destroy_model_parallel
from tests.unit_tests.test_utilities import Utils


@pytest.fixture
def process_group_collection(distributed_setup):
    """Initialize the process groups used by the real model constructors."""
    if not torch.cuda.is_available():
        pytest.skip("MCore model construction requires CUDA.")
    Utils.initialize_model_parallel(1, 1)
    model_parallel_cuda_manual_seed(1234)
    yield ProcessGroupCollection.use_mpu_process_groups()
    destroy_model_parallel()


@pytest.mark.parametrize("grouped_gemm", [True, False], ids=["grouped", "sequential"])
def test_collects_routed_experts_before_transformer_layer(process_group_collection, grouped_gemm):
    """Transformer layers select routed experts directly and own their remaining internals."""
    config = TransformerConfig(
        num_layers=1,
        hidden_size=64,
        num_attention_heads=4,
        num_moe_experts=2,
        moe_ffn_hidden_size=128,
        moe_shared_expert_intermediate_size=128,
        moe_grouped_gemm=grouped_gemm,
        moe_token_dispatcher_type="alltoall",
        add_bias_linear=False,
        gradient_accumulation_fusion=False,
        attention_backend=AttnBackend.unfused,
    )
    layer = TransformerLayer(
        config=config,
        submodules=get_gpt_layer_with_transformer_engine_spec(
            num_experts=2, moe_grouped_gemm=grouped_gemm
        ).submodules,
        pg_collection=process_group_collection,
        add_layer_offset=False,
    )
    experts = layer.mlp.experts
    assert isinstance(experts, TEGroupedMLP if grouped_gemm else SequentialMLP)
    assert layer.mlp.shared_experts is not None
    root = nn.Sequential(layer)
    collector = FsdpUnitCollector()

    assert collector.collect(root) == [experts, layer, root]


@pytest.mark.parametrize("pattern", ["*E*", "MEM"], ids=["attention", "mamba"])
def test_shortcut_owns_helper_called_layers(process_group_collection, pattern):
    """Wrapper units keep helper-called layers and leave expert units independent."""
    config = TransformerConfig(
        num_layers=3,
        hidden_size=256,
        num_attention_heads=4,
        num_moe_experts=2,
        moe_ffn_hidden_size=512,
        moe_grouped_gemm=True,
        moe_token_dispatcher_type="alltoall",
        moe_shortcut_connection=True,
        add_bias_linear=False,
        gradient_accumulation_fusion=False,
        attention_backend=AttnBackend.unfused,
    )
    model = HybridModel(
        config=config,
        hybrid_stack_spec=hybrid_stack_spec,
        vocab_size=128,
        max_sequence_length=8,
        hybrid_layer_pattern=pattern,
        pg_collection=process_group_collection,
    )
    block, sibling = model.decoder.layers
    assert isinstance(block, ShortcutMoEBlock)
    assert isinstance(sibling, TransformerLayer if pattern == "*E*" else MambaLayer)
    collector = FsdpUnitCollector()

    assert collector.collect(model) == [block.moe_layer.mlp.experts, block, sibling, model]


_SELECTIVE = dict(recompute_granularity="selective")
_FULL = dict(recompute_granularity="full", recompute_method="uniform", recompute_num_layers=1)
_MTP = dict(mtp_num_layers=1)
_MLA = dict(
    multi_latent_attention=True,
    q_lora_rank=32,
    kv_lora_rank=32,
    qk_head_dim=16,
    qk_pos_emb_head_dim=16,
    v_head_dim=16,
    rope_type="rope",
)
# A dense layer followed by a MoE layer with shared experts, as in DeepSeek-V3.
_MOE = dict(
    num_moe_experts=4,
    moe_layer_freq=[0, 1],
    moe_router_topk=2,
    moe_grouped_gemm=True,
    moe_ffn_hidden_size=128,
    moe_shared_expert_intermediate_size=128,
    moe_token_dispatcher_type="alltoall",
)
_SHORTCUT = dict(**_MOE, num_layers=3, moe_shortcut_connection=True)


def _build_config(**kwargs) -> TransformerConfig:
    config_kwargs = dict(
        num_layers=2,
        hidden_size=128,
        num_attention_heads=4,
        ffn_hidden_size=256,
        add_bias_linear=False,
    )
    config_kwargs.update(kwargs)
    config_class = (
        MLATransformerConfig if kwargs.get("multi_latent_attention") else TransformerConfig
    )
    return config_class(**config_kwargs)


def _build_gpt(config: TransformerConfig) -> GPTModel:
    """Build a GPTModel as gpt_builder does."""
    mtp_block_spec = None
    if config.mtp_num_layers:
        mtp_layer_spec = get_gpt_decoder_layer_specs(config, use_transformer_engine=True)[-1]
        mtp_block_spec = get_gpt_mtp_block_spec(config, mtp_layer_spec, use_transformer_engine=True)
    return GPTModel(
        config=config,
        transformer_layer_spec=get_gpt_decoder_block_spec(config, use_transformer_engine=True),
        mtp_block_spec=mtp_block_spec,
        vocab_size=128,
        max_sequence_length=32,
    )


def _build_hybrid(config: TransformerConfig, **hybrid_kwargs) -> HybridModel:
    """Build a HybridModel as hybrid_builder does, with a layer pattern or config list."""
    return HybridModel(
        config=config,
        hybrid_stack_spec=hybrid_stack_spec,
        vocab_size=128,
        max_sequence_length=32,
        **hybrid_kwargs,
    )


def _build_hybrid_with_layer_recompute(config: TransformerConfig) -> HybridModel:
    """Build a HybridModel whose MLP layer's own config, unlike the root config, recomputes."""
    mlp_layer_config = MLPLayerConfig.from_config(config)
    mlp_layer_config.recompute_granularity = "selective"
    mlp_layer_config.recompute_modules = ["mlp"]
    return _build_hybrid(
        config,
        hybrid_layer_config_list=[AttentionLayerConfig.from_config(config), mlp_layer_config],
    )


# Attention layers with an MTP layer; "/" separates the decoder and MTP patterns.
_build_hybrid_mtp = partial(_build_hybrid, hybrid_layer_pattern="*/*")
# A ShortcutMoEBlock wrapping an attention layer and a MoE layer, then an attention layer.
_build_hybrid_shortcut = partial(_build_hybrid, hybrid_layer_pattern="*E*")


def _partially_recomputed_unit_names(model: nn.Module, collector: FsdpUnitCollector) -> set[str]:
    """Return the names of collected units that collector marks as partially recomputed."""
    units = collector.collect(model)
    return {
        name
        for name, module in model.named_modules()
        if module in units and collector.is_partially_recomputed(module)
    }


@pytest.mark.parametrize(
    "build_model, config_kwargs, fsdp_unit_modules, expected_units",
    [
        pytest.param(_build_gpt, _MTP, None, set(), id="no_recompute"),
        # Only the dense layer recomputes its MLP; the MoE layers, including MTP's, don't.
        pytest.param(
            _build_gpt,
            dict(**_MOE, **_MTP, **_SELECTIVE, recompute_modules=["mlp"]),
            None,
            {"decoder.layers.0"},
            id="selective_mlp",
        ),
        # Attention layers' MLPs are parameterless IdentityOps.
        pytest.param(
            _build_hybrid_mtp,
            dict(**_MTP, **_SELECTIVE, recompute_modules=["mlp"]),
            None,
            set(),
            id="selective_mlp_identity",
        ),
        pytest.param(
            _build_gpt,
            dict(**_MLA, **_SELECTIVE, recompute_modules=["mla_up_proj"]),
            None,
            {"decoder.layers.0", "decoder.layers.1"},
            id="selective_mla_up_proj",
        ),
        # Only the MLP layer's own config recomputes; the root config does not.
        pytest.param(
            _build_hybrid_with_layer_recompute,
            {},
            None,
            {"decoder.layers.1"},
            id="selective_layer_config",
        ),
        pytest.param(
            _build_gpt,
            dict(**_MOE, **_SELECTIVE, recompute_modules=["moe"]),
            None,
            {"decoder.layers.1"},
            id="selective_moe",
        ),
        pytest.param(
            _build_gpt,
            dict(**_MOE, **_SELECTIVE, recompute_modules=["shared_experts"]),
            None,
            {"decoder.layers.1"},
            id="selective_shared_experts",
        ),
        # The shortcut block owns its helper-called MoE layer.
        pytest.param(
            _build_hybrid_shortcut,
            dict(**_SHORTCUT, **_SELECTIVE, recompute_modules=["shared_experts"]),
            None,
            {"decoder.layers.0"},
            id="selective_shared_experts_shortcut",
        ),
        # Only the root owns eh_proj, which the MTP checkpoint reruns.
        pytest.param(_build_gpt, dict(**_MTP, **_FULL), None, {""}, id="full"),
        # Without layer units, the block's checkpoint reruns layers owned by the root or block.
        pytest.param(_build_gpt, _FULL, [], {""}, id="full_no_layer_units"),
        pytest.param(_build_gpt, _FULL, [TransformerBlock], {"decoder"}, id="full_block_unit"),
        # The MTP layer checkpoints inside its own forward, so it needs the opt-in as a unit too.
        pytest.param(
            _build_gpt,
            dict(**_MTP, **_FULL),
            [TransformerLayer, MultiTokenPredictionLayer],
            {"mtp.layers.0"},
            id="full_mtp_layer_unit",
        ),
        # Hybrid MTP layers do not checkpoint eh_proj; their inner layers are units.
        pytest.param(_build_hybrid_mtp, dict(**_MTP, **_FULL), None, set(), id="full_hybrid_mtp"),
    ],
)
def test_marks_units_partially_recomputed_by_inner_checkpoints(
    process_group_collection, build_model, config_kwargs, fsdp_unit_modules, expected_units
):
    """Units are marked exactly when an inner checkpoint reruns their own parameters."""
    model = build_model(_build_config(**config_kwargs))

    collector = FsdpUnitCollector(fsdp_unit_modules)
    assert _partially_recomputed_unit_names(model, collector) == expected_units
