# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Default FSDP unit selection for real MCore layers."""

import pytest
import torch
from torch import nn

from megatron.core.distributed.fsdp.fsdp_unit_collector import FsdpUnitCollector
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_with_transformer_engine_spec
from megatron.core.models.hybrid.hybrid_layer_specs import hybrid_stack_spec
from megatron.core.models.hybrid.hybrid_model import HybridModel
from megatron.core.models.hybrid.shortcut_block import ShortcutMoEBlock
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.ssm.mamba_layer import MambaLayer
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.enums import AttnBackend
from megatron.core.transformer.moe.experts import SequentialMLP, TEGroupedMLP
from megatron.core.transformer.transformer_config import TransformerConfig
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
