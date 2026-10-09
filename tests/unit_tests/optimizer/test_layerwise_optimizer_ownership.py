# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Preserve main's compact Muon/Adam ownership and existing checkpoint interface."""

from copy import deepcopy
from itertools import cycle

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
from megatron.core.optimizer.optimizer import ChainedOptimizer, Float16OptimizerWithFloat16Params
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


def _build_model_and_config(layout, expert_bias, overlap, seed, direct_core=False):
    torch.manual_seed(seed)
    pg = ProcessGroupCollection.use_mpu_process_groups()
    model = torch.nn.Module()
    # Enough whole parameters for every rank to own Muon and Adam state. The
    # scalar groups deliberately include parameters outside Muon's routing tag.
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
        tag_params_for_buffer_routing([model])
        ddp = DistributedDataParallel(config, ddp_config, model, pg_collection=pg)
        assert ddp_config.use_distributed_optimizer, "DDP must preserve the caller's configuration"
        assert ddp.ddp_config is not ddp_config
    else:
        ddp = wrap_model_chunks_with_ddp(
            [model],
            config,
            ddp_config,
            use_layer_wise_distributed_optimizer=True,
            use_layer_wise_param_layout=layout,
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
    )
    return ddp, optimizer_config, pg


def _make_optimizer(ddp, config, pg):
    return get_megatron_optimizer(config, [ddp], pg_collection=pg, use_gloo_process_groups=False)


def _assert_compact_ownership(ddp, optimizer, pg):
    assert isinstance(optimizer, LayerWiseDistributedOptimizer)
    assert len(optimizer.chained_optimizers) == 2
    assert all(
        isinstance(child, Float16OptimizerWithFloat16Params)
        for child in optimizer.chained_optimizers
    )
    assert (
        sum(
            isinstance(child.optimizer, optimizer_module.Adam)
            for child in optimizer.chained_optimizers
        )
        == 1
    )
    assert not ddp.ddp_config.use_distributed_optimizer
    assert ddp.full_param_layout is None
    local_params = [
        param
        for child in optimizer.chained_optimizers
        for group in child.float16_groups
        for param in group
    ]
    assert len(local_params) == len(set(local_params)), "Each local parameter has one optimizer"
    for param in ddp.parameters():
        group = pg.dp_cp if getattr(param, "allreduce", True) else pg.expt_dp
        local_owner = param in set(local_params)
        owner_count = torch.tensor(int(local_owner), device="cuda", dtype=torch.int32)
        torch.distributed.all_reduce(owner_count, group=group)
        assert owner_count.item() == 1, "Whole-parameter ownership must cover every parameter once"
        if local_owner:
            assert param.main_param.shape == param.shape
    for group in ddp.bucket_groups + ddp.expert_parallel_bucket_groups:
        assert group.param_sync_via_bucket_group
        for bucket in group.buckets:
            bound = [param for rank_params in bucket.layerwise_params_list for param in rank_params]
            assert len(bound) == len(set(bound))
            assert set(bound) == bucket.params, "Compact gather must include the Adam fallback"
            assert bucket.param_data is None


@pytest.mark.parametrize("layout", [False, True])
@pytest.mark.parametrize("expert_bias", [False, True])
def test_layerwise_optimizer_preserves_main_ownership(layout, expert_bias):
    """Compact owns all parameters; padded retains main's scalar DistOpt restrictions."""
    Utils.initialize_model_parallel(expert_model_parallel_size=2, expert_tensor_parallel_size=1)
    try:
        ddp, config, pg = _build_model_and_config(layout, expert_bias, True, 1234)
        if layout and expert_bias:
            with pytest.raises(AssertionError, match="Non-emerging expert-parallel param groups"):
                _make_optimizer(ddp, config, pg)
            return
        optimizer = _make_optimizer(ddp, config, pg)
        if not layout:
            _assert_compact_ownership(ddp, optimizer, pg)
            return
        assert type(optimizer) is ChainedOptimizer
        layerwise, adam = optimizer.chained_optimizers
        assert isinstance(layerwise, LayerWiseDistributedOptimizer)
        assert isinstance(adam, DistributedOptimizer)
        assert ddp.ddp_config.use_distributed_optimizer
        assert ddp.full_param_layout is not None
        assert all(
            param.is_managed_by_layer_wise_optimizer
            for child in layerwise.chained_optimizers
            for group in child.float16_groups
            for param in group
        )
        scalar_params = {
            param for param in ddp.parameters() if not param.is_managed_by_layer_wise_optimizer
        }
        assert {param for buffer in adam.buffers for param in buffer.params} == scalar_params
        assert adam.data_parallel_group is pg.dp_cp
    finally:
        Utils.destroy_model_parallel()


@pytest.mark.parametrize("expert_bias", [False, True])
def test_compact_core_api_preserves_layerwise_ownership(expert_bias):
    """Direct DDP construction uses the same compact ownership as the training entry point."""
    Utils.initialize_model_parallel(expert_model_parallel_size=2, expert_tensor_parallel_size=1)
    try:
        ddp, config, pg = _build_model_and_config(False, expert_bias, True, 1234, direct_core=True)
        optimizer = _make_optimizer(ddp, config, pg)
        _assert_compact_ownership(ddp, optimizer, pg)
        for buffer in ddp.buffers + ddp.expert_parallel_buffers:
            assert not buffer.ddp_config.use_distributed_optimizer
    finally:
        Utils.destroy_model_parallel()


def _shard_with_main_owners(optimizer, base_optimizers, dp_size, expert_dp_size):
    """Reference main's stable-numel ownership independently of DDP buffer offsets."""
    groups = [group for base in base_optimizers for group in base.param_groups]
    entries = [(param, index) for index, group in enumerate(groups) for param in group["params"]]
    entries.sort(key=lambda item: item[0].numel())
    rank_cycles = {
        False: cycle([*range(dp_size), *reversed(range(dp_size))]),
        True: cycle([*range(expert_dp_size), *reversed(range(expert_dp_size))]),
    }
    owners = {False: [[] for _ in range(dp_size)], True: [[] for _ in range(expert_dp_size)]}
    local_rank = {False: optimizer.dp_cp.rank(), True: optimizer.expt_dp.rank()}
    local_groups = [[] for _ in groups]
    for param, group_index in entries:
        expert = groups[group_index].get("is_expert_parallel", False)
        owner = next(rank_cycles[expert])
        owners[expert][owner].append(param)
        if owner == local_rank[expert]:
            local_groups[group_index].append(param)
    for group, params in zip(groups, local_groups):
        group["params"] = params
    optimizer.dp_cp_params_list = owners[False]
    optimizer.expt_dp_params_list = (
        owners[True] if expert_dp_size > 1 and any(owners[True]) else None
    )


def _local_parameter_names(ddp, optimizer):
    names = {param: name for name, param in ddp.named_parameters()}
    return [
        [[names[param] for param in group] for group in child.float16_groups]
        for child in optimizer.chained_optimizers
    ]


def _assert_equal(actual, expected):
    if isinstance(expected, torch.Tensor):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    elif isinstance(expected, dict):
        assert actual.keys() == expected.keys()
        for key in expected:
            _assert_equal(actual[key], expected[key])
    elif isinstance(expected, (tuple, list)):
        assert len(actual) == len(expected)
        for actual_item, expected_item in zip(actual, expected):
            _assert_equal(actual_item, expected_item)
    else:
        assert actual == expected


def _step(ddp, optimizer, step):
    ddp.zero_grad_buffer()
    optimizer.zero_grad()
    for index, param in enumerate(ddp.parameters()):
        grad = torch.arange(param.numel(), device=param.device, dtype=torch.float32)
        grad = ((grad % 17) + index + step + torch.distributed.get_rank()) / 1024
        param.main_grad.copy_(grad.reshape_as(param))
    ddp.finish_grad_sync()
    assert optimizer.step()[0]
    ddp.start_param_sync(force_sync=True)


@pytest.mark.parametrize("expert_parallel_size", [1, 2])
@pytest.mark.parametrize("overlap", [False, True])
def test_compact_main_torch_checkpoint_restores_next_update(
    tmp_path, monkeypatch, expert_parallel_size, overlap
):
    """Main-owner checkpoints preserve masters, moments and the next update after loading."""
    Utils.initialize_model_parallel(
        expert_model_parallel_size=expert_parallel_size, expert_tensor_parallel_size=1
    )
    try:
        ddp, config, pg = _build_model_and_config(False, True, overlap, 1234)
        # Build the saved optimizer using main's ownership rule. The repeated equal-size
        # parameters deliberately expose any new buffer-offset tie-breaker in the loader.
        with monkeypatch.context() as patch:
            patch.setattr(
                LayerWiseDistributedOptimizer, "_shard_params_ping_pong", _shard_with_main_owners
            )
            optimizer = _make_optimizer(ddp, config, pg)
        _assert_compact_ownership(ddp, optimizer, pg)
        saved_owner_names = _local_parameter_names(ddp, optimizer)
        _step(ddp, optimizer, 1)
        _step(ddp, optimizer, 2)
        saved_model = deepcopy(ddp.module.state_dict())
        saved_optimizer = deepcopy(optimizer.state_dict())
        # Float16Optimizer's ordinary state_dict already contains unrounded whole
        # FP32 masters and all inner Adam/Muon state; no DistOpt sidecar is needed.
        adam = next(
            child
            for child in optimizer.chained_optimizers
            if isinstance(child.optimizer, optimizer_module.Adam)
        )
        masters = [master for group in adam.fp32_from_float16_groups for master in group]
        assert masters, "Every rank must exercise actual Adam state"
        assert any(not torch.equal(master, master.bfloat16().float()) for master in masters)
        for master in masters:
            assert torch.count_nonzero(adam.optimizer.state[master]["exp_avg"]) > 0
            assert torch.count_nonzero(adam.optimizer.state[master]["exp_avg_sq"]) > 0
        path = tmp_path / f"layer_wise_optimizer_{torch.distributed.get_rank()}.pt"
        optimizer.save_state_dict_to_file(path)
        _assert_equal(torch.load(path, weights_only=False), saved_optimizer)
        _step(ddp, optimizer, 3)
        expected_optimizer = deepcopy(optimizer.state_dict())
        expected_model = deepcopy(ddp.module.state_dict())

        resumed, resumed_config, resumed_pg = _build_model_and_config(False, True, overlap, 9876)
        resumed_optimizer = _make_optimizer(resumed, resumed_config, resumed_pg)
        assert _local_parameter_names(resumed, resumed_optimizer) == saved_owner_names
        resumed.module.load_state_dict(saved_model)
        resumed_optimizer.load_state_dict_from_file(path)
        _assert_equal(resumed_optimizer.state_dict(), saved_optimizer)
        _step(resumed, resumed_optimizer, 3)
        _assert_equal(resumed_optimizer.state_dict(), expected_optimizer)
        _assert_equal(resumed.module.state_dict(), expected_model)
    finally:
        Utils.destroy_model_parallel()
