# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
"""Integration tests for branchwise wide residual around a real MoE layer."""

import copy
import os

import pytest
import torch

from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_local_submodules
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.tensor_parallel.random import initialize_rng_tracker
from megatron.core.transformer.identity_op import IdentityFuncOp, IdentityOp
from megatron.core.transformer.spec_utils import ModuleSpec, build_module
from megatron.core.transformer.transformer_config import TransformerConfig, WideResidualConfig
from megatron.core.transformer.transformer_layer import MoETransformerLayer
from megatron.core.transformer.wide_residual_layer import (
    StreamwiseSigmoidWideResidualConnection,
    specialize_wide_residual_layer_spec,
)
from megatron.training.initialize import _set_random_seed
from tests.unit_tests.test_utilities import Utils


def _config(
    *,
    tensor_model_parallel_size: int = 1,
    expert_model_parallel_size: int = 1,
    sequence_parallel: bool = False,
) -> TransformerConfig:
    return TransformerConfig(
        num_layers=1,
        hidden_size=64,
        num_attention_heads=4,
        num_moe_experts=4,
        moe_ffn_hidden_size=128,
        moe_shared_expert_intermediate_size=96,
        moe_latent_size=32,
        moe_token_dispatcher_type="alltoall",
        moe_router_load_balancing_type="none",
        moe_router_topk=1,
        moe_router_pre_softmax=True,
        moe_aux_loss_coeff=0.0,
        add_bias_linear=False,
        hidden_dropout=0.0,
        bias_dropout_fusion=False,
        tensor_model_parallel_size=tensor_model_parallel_size,
        expert_model_parallel_size=expert_model_parallel_size,
        sequence_parallel=sequence_parallel,
        use_cpu_initialization=True,
        wide_residual=WideResidualConfig(
            num_streams=3,
            streamwise_sigmoid_init_scale=0.01,
            learned_retention=True,
            retention_init=0.999,
            retention_max_forget=0.10,
        ),
    )


def _build_layer(config: TransformerConfig) -> MoETransformerLayer:
    submodules = copy.deepcopy(
        get_gpt_layer_local_submodules(num_experts=config.num_moe_experts, moe_grouped_gemm=False)
    )
    submodules.input_layernorm = IdentityOp
    submodules.self_attention = IdentityOp
    submodules.self_attn_bda = IdentityFuncOp
    layer_spec = specialize_wide_residual_layer_spec(
        ModuleSpec(module=MoETransformerLayer, submodules=submodules), config
    )
    return build_module(
        layer_spec,
        config=config,
        layer_number=1,
        add_layer_offset=False,
        pg_collection=ProcessGroupCollection.use_mpu_process_groups(),
        name="wide_moe_layer",
    ).cuda()


def _assert_branchwise_geometry(layer: MoETransformerLayer, config: TransformerConfig) -> None:
    wide_hidden_size = config.wide_residual.num_streams * config.hidden_size
    connection = layer.residual_connection_mlp
    assert isinstance(connection, StreamwiseSigmoidWideResidualConnection)
    assert layer.residual_stream_hidden_size == wide_hidden_size
    assert connection.residual_stream_hidden_size == wide_hidden_size
    assert connection.branch_hidden_size == config.hidden_size
    assert layer.mlp.router.weight.shape[-1] == config.hidden_size
    assert layer.mlp.shared_experts.config.hidden_size == config.hidden_size


@pytest.mark.internal
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_branchwise_wide_moe_keeps_all_moe_paths_at_backbone_width():
    Utils.initialize_model_parallel(1, 1)
    try:
        _set_random_seed(seed_=123, data_parallel_random_init=False)
        config = _config()
        layer = _build_layer(config)
        wide_hidden_size = config.wide_residual.num_streams * config.hidden_size
        residual_stream = torch.randn(
            16, 2, wide_hidden_size, device=torch.cuda.current_device(), requires_grad=True
        )

        output, context = layer(hidden_states=residual_stream, attention_mask=None)

        assert context is None
        assert output.shape == residual_stream.shape
        _assert_branchwise_geometry(layer, config)
        output.float().square().mean().backward()
        assert residual_stream.grad is not None
        assert layer.mlp.router.weight.grad is not None
        assert any(parameter.grad is not None for parameter in layer.mlp.experts.parameters())
        assert any(
            parameter.grad is not None for parameter in layer.mlp.shared_experts.parameters()
        )
        assert layer.residual_connection_mlp.read_map.logit.grad is not None
        assert layer.residual_connection_mlp.write_map.logit.grad is not None
        assert layer.residual_connection_mlp.retention.retention_logit.grad is not None
    finally:
        Utils.destroy_model_parallel()


@pytest.mark.internal
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.skipif(
    int(os.environ.get("WORLD_SIZE", "1")) != 4,
    reason="Run this test with torchrun --nproc-per-node=4.",
)
def test_branchwise_wide_moe_runs_with_tensor_and_expert_parallelism():
    tensor_parallel_size = 2
    expert_parallel_size = 2
    Utils.initialize_model_parallel(
        tensor_model_parallel_size=tensor_parallel_size,
        expert_model_parallel_size=expert_parallel_size,
    )
    try:
        initialize_rng_tracker(force_reset=True)
        _set_random_seed(seed_=123, data_parallel_random_init=False)
        config = _config(
            tensor_model_parallel_size=tensor_parallel_size,
            expert_model_parallel_size=expert_parallel_size,
            sequence_parallel=True,
        )
        layer = _build_layer(config)
        wide_hidden_size = config.wide_residual.num_streams * config.hidden_size
        residual_stream = torch.randn(
            16, 2, wide_hidden_size, device=torch.cuda.current_device(), requires_grad=True
        )

        output, context = layer(hidden_states=residual_stream, attention_mask=None)

        assert context is None
        assert output.shape == residual_stream.shape
        _assert_branchwise_geometry(layer, config)
        output.float().square().mean().backward()
        assert residual_stream.grad is not None
        assert layer.residual_connection_mlp.read_map.logit.grad is not None
        assert layer.residual_connection_mlp.write_map.logit.grad is not None
        assert layer.residual_connection_mlp.retention.retention_logit.grad is not None
    finally:
        Utils.destroy_model_parallel()
