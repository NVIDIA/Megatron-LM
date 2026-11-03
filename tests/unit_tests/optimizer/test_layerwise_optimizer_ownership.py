# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Muon whole-parameter ownership and Adam sharding are independent of layout."""

import copy

import pytest
import torch

import megatron.core.optimizer as optimizer_module
from megatron.core.distributed import DistributedDataParallel, DistributedDataParallelConfig
from megatron.core.optimizer import OptimizerConfig, get_megatron_optimizer
from megatron.core.optimizer.distrib_optimizer import DistributedOptimizer
from megatron.core.optimizer.layer_wise_optimizer import (
    LayerWiseDistributedOptimizer,
    tag_params_for_buffer_routing,
)
from megatron.core.optimizer.optimizer import ChainedOptimizer
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.transformer import TransformerConfig
from megatron.training.training import wrap_model_chunks_with_ddp
from tests.unit_tests.test_utilities import Utils

pytestmark = [
    pytest.mark.launch_on_gb200,
    pytest.mark.skipif(
        not optimizer_module.HAVE_EMERGING_OPTIMIZERS, reason="requires emerging-optimizers"
    ),
]


def _build_model_and_config(layout, expert_bias, overlap, seed, direct_core=False, sinkhorn=False):
    torch.manual_seed(seed)
    pg = ProcessGroupCollection.use_mpu_process_groups()
    model = torch.nn.Module()
    # Exercise both whole-matrix Muon ownership and sharded scalar/embedding Adam state.
    model.matrices = torch.nn.ParameterList(
        [
            torch.nn.Parameter(torch.randn(64, 64, device="cuda", dtype=torch.bfloat16))
            for _ in range(16)
        ]
    )
    model.biases = torch.nn.ParameterList(
        [
            torch.nn.Parameter(torch.randn(64, device="cuda", dtype=torch.bfloat16))
            for _ in range(16)
        ]
    )
    model.embedding = torch.nn.Parameter(torch.randn(512, 128, device="cuda", dtype=torch.bfloat16))
    model.embedding.is_embedding_or_output_parameter = True
    if pg.ep.size() > 1:
        model.experts = torch.nn.ParameterList(
            [
                torch.nn.Parameter(torch.randn(64, 64, device="cuda", dtype=torch.bfloat16))
                for _ in range(8)
            ]
        )
        for param in model.experts:
            param.allreduce = False
        if expert_bias:
            model.expert_biases = torch.nn.ParameterList(
                [
                    torch.nn.Parameter(torch.randn(64, device="cuda", dtype=torch.bfloat16))
                    for _ in range(8)
                ]
            )
            for param in model.expert_biases:
                param.allreduce = False

    config = TransformerConfig(
        num_layers=1,
        hidden_size=64,
        num_attention_heads=1,
        bf16=True,
        params_dtype=torch.bfloat16,
        num_moe_experts=2 if pg.ep.size() > 1 else None,
        expert_model_parallel_size=pg.ep.size(),
        expert_tensor_parallel_size=1,
    )
    ddp_config = DistributedDataParallelConfig(
        grad_reduce_in_fp32=True, overlap_param_gather=overlap
    )
    if direct_core:
        ddp_config.use_distributed_optimizer = True
        ddp_config.use_layer_wise_param_layout = layout
        tag_params_for_buffer_routing([model], use_sinkhorn_for_embeddings=sinkhorn)
        ddp = DistributedDataParallel(config, ddp_config, model, pg_collection=pg)
        assert ddp_config.use_distributed_optimizer, "DDP must preserve the caller's configuration"
    else:
        ddp = wrap_model_chunks_with_ddp(
            [model],
            config,
            ddp_config,
            use_layer_wise_distributed_optimizer=True,
            use_layer_wise_param_layout=layout,
            use_sinkhorn_for_embeddings=sinkhorn,
            pg_collection=pg,
        )[0]
    optimizer_config = OptimizerConfig(
        optimizer="muon",
        lr=0.001,
        bf16=True,
        clip_grad=0.0,
        weight_decay=0.0,
        use_layer_wise_distributed_optimizer=True,
        use_layer_wise_param_layout=layout,
        overlap_param_gather=overlap,
        muon_split_qkv=False,
        muon_tp_mode="duplicated",
        muon_use_sinkhorn=sinkhorn,
    )
    return ddp, optimizer_config, pg


def _make_optimizer(ddp, config, pg):
    return get_megatron_optimizer(config, [ddp], pg_collection=pg, use_gloo_process_groups=False)


@pytest.mark.parametrize("layout", [False, True])
@pytest.mark.parametrize("direct_core", [False, True])
@pytest.mark.parametrize("expert_bias", [False, True])
@pytest.mark.parametrize("sinkhorn", [False, True])
def test_layerwise_optimizer_ownership(layout, direct_core, expert_bias, sinkhorn):
    """Both layouts shard Adam; only compact Muon uses unpadded all-reduce buffers."""
    if sinkhorn:
        pytest.importorskip('emerging_optimizers.orthogonalized_optimizers.sinkhorn_utils')
    Utils.initialize_model_parallel(expert_model_parallel_size=2, expert_tensor_parallel_size=1)
    try:
        ddp, config, pg = _build_model_and_config(
            layout, expert_bias, True, 1234, direct_core=direct_core, sinkhorn=sinkhorn
        )
        if expert_bias:
            with pytest.raises(AssertionError, match="Non-emerging expert-parallel param groups"):
                _make_optimizer(ddp, config, pg)
            return
        optimizer = _make_optimizer(ddp, config, pg)
        assert type(optimizer) is ChainedOptimizer
        layerwise, adam = optimizer.chained_optimizers
        assert isinstance(layerwise, LayerWiseDistributedOptimizer)
        assert isinstance(adam, DistributedOptimizer)
        assert ddp.ddp_config.use_distributed_optimizer
        assert ddp.full_param_layout is not None
        assert ddp.module.embedding.is_managed_by_layer_wise_optimizer is sinkhorn
        assert sum(
            type(c.optimizer).__name__ == 'Sinkhorn' for c in layerwise.chained_optimizers
        ) == int(sinkhorn)
        local_muon_params = {
            param
            for child in layerwise.chained_optimizers
            for group in child.float16_groups
            for param in group
        }
        assert all(param.is_managed_by_layer_wise_optimizer for param in local_muon_params)
        scalar_params = {
            param for param in ddp.parameters() if not param.is_managed_by_layer_wise_optimizer
        }
        assert {param for buffer in adam.buffers for param in buffer.params} == scalar_params
        assert adam.data_parallel_group is pg.dp_cp
        # Count actual FP32 master elements, excluding padding. Replicating complete
        # Adam tensors on every rank would multiply this total by the DP size.
        local_numel = sum(
            param.main_param.numel() for param in scalar_params if param.main_param is not None
        )
        total_numel = torch.tensor(local_numel, dtype=torch.int64, device="cuda")
        torch.distributed.all_reduce(total_numel, group=pg.dp_cp)
        assert total_numel.item() == sum(param.numel() for param in scalar_params)
        for param in ddp.parameters():
            if not param.is_managed_by_layer_wise_optimizer:
                assert param.main_param_sharded
                continue
            group = pg.dp_cp if getattr(param, "allreduce", True) else pg.expt_dp
            local_owner = param in local_muon_params
            owner_count = torch.tensor(int(local_owner), device="cuda", dtype=torch.int32)
            torch.distributed.all_reduce(owner_count, group=group)
            assert owner_count.item() == 1
            if local_owner:
                assert param.main_param.shape == param.shape
        for buffer in ddp.buffers + ddp.expert_parallel_buffers:
            muon = buffer.params[0].is_managed_by_layer_wise_optimizer
            assert buffer.ddp_config.use_distributed_optimizer is (layout or not muon)
            if muon and not layout:
                assert buffer.ddp_config is not ddp.ddp_config
                assert buffer.numel == buffer.numel_unpadded
                for bucket in buffer.buckets:
                    bound = [
                        param
                        for rank_params in bucket.layerwise_params_list
                        for param in rank_params
                    ]
                    assert len(bound) == len(set(bound))
                    assert set(bound) == bucket.params
                    assert bucket.param_data is None
            else:
                assert buffer.param_data is not None
    finally:
        Utils.destroy_model_parallel()


@pytest.mark.parametrize('layout', [False, True])
def test_sinkhorn_distributed_update_and_resume(layout):
    """A DP-owned FP32 embedding update must agree with an unsharded optimizer."""
    pytest.importorskip('emerging_optimizers.orthogonalized_optimizers.sinkhorn_utils')
    from megatron.core.optimizer.emerging_optimizers import Sinkhorn

    Utils.initialize_model_parallel()
    try:
        ddp, config, pg = _build_model_and_config(layout, False, False, 1701, sinkhorn=True)
        optimizer = _make_optimizer(ddp, config, pg)
        layerwise = optimizer.chained_optimizers[0]
        reference_param = torch.nn.Parameter(ddp.module.embedding.detach().float().clone())
        reference = Sinkhorn([reference_param], lr=config.lr)
        gradients = {p: torch.randn_like(p) for p in ddp.parameters()}
        gradients[ddp.module.embedding][0].zero_()
        gradients[ddp.module.embedding][1].mul_(1e-6)

        for iteration in range(3):
            ddp.zero_grad_buffer()
            optimizer.zero_grad()
            loss = sum((p.float() * gradients[p].float()).sum() for p in ddp.parameters())
            loss.backward()
            ddp.finish_grad_sync()
            success, _, _ = optimizer.step()
            assert success
            reference_param.grad = gradients[ddp.module.embedding].float()
            reference.step()
            torch.testing.assert_close(
                ddp.module.embedding, reference_param.bfloat16(), atol=0, rtol=0
            )
            if iteration == 0:
                saved = copy.deepcopy(layerwise.state_dict())
                saved_embedding = ddp.module.embedding.detach().clone()
                saved_reference = reference_param.detach().clone()
                saved_reference_state = copy.deepcopy(reference.state_dict())
            elif iteration == 1:
                expected = ddp.module.embedding.detach().clone()
                # Restore the matrix optimizers' state and FP32 masters. Gradients in
                # this test are parameter-independent, so Adam's other weights do not
                # affect the embedding's continuation.
                layerwise.load_state_dict(saved)
                with torch.no_grad():
                    ddp.module.embedding.copy_(saved_embedding)
                    reference_param.copy_(saved_reference)
                reference.load_state_dict(saved_reference_state)
            else:
                torch.testing.assert_close(ddp.module.embedding, expected, atol=0, rtol=0)
    finally:
        Utils.destroy_model_parallel()
