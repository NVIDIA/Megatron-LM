# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Legacy rank-local torch checkpoints for compact LayerWise + DistOpt chains."""

from copy import deepcopy

import pytest
import torch

from megatron.core.distributed import DistributedDataParallelConfig
from megatron.core.optimizer import OptimizerConfig, get_megatron_optimizer
from megatron.core.optimizer.distrib_optimizer import DistributedOptimizer
from megatron.core.optimizer.layer_wise_optimizer import LayerWiseDistributedOptimizer
from megatron.core.optimizer.optimizer import ChainedOptimizer, FP32Optimizer
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.transformer import TransformerConfig
from megatron.training.training import wrap_model_chunks_with_ddp
from tests.unit_tests.test_utilities import Utils

pytestmark = pytest.mark.launch_on_gb200


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


def _dist_optimizers(optimizer):
    return [
        child
        for child in optimizer._iter_leaf_optimizers()
        if isinstance(child, DistributedOptimizer)
    ]


def _snapshot(optimizer):
    # DistOpt.state_dict intentionally omits FP32 masters and moments.
    return deepcopy(
        (
            optimizer.state_dict(),
            [child.get_parameter_state_dp_reshardable() for child in _dist_optimizers(optimizer)],
        )
    )


def _build_compact_optimizer(seed, with_experts):
    torch.manual_seed(seed)
    pg_collection = ProcessGroupCollection.use_mpu_process_groups()
    model = torch.nn.Module()
    model.matrices = torch.nn.ParameterList(
        [
            torch.nn.Parameter(torch.randn(64, 64, device="cuda", dtype=torch.bfloat16))
            for _ in range(16)
        ]
    )
    # A sufficiently large Adam tensor gives every rank a nonempty shard. Empty-padding
    # bucket checkpoint behavior is a separate issue, not part of this regression.
    model.embedding = torch.nn.Parameter(torch.randn(512, 128, device="cuda", dtype=torch.bfloat16))
    model.embedding.is_embedding_or_output_parameter = True
    if with_experts:
        model.experts = torch.nn.ParameterList(
            [
                torch.nn.Parameter(torch.randn(64, 64, device="cuda", dtype=torch.bfloat16))
                for _ in range(8)
            ]
        )
        model.expert_scalar = torch.nn.Parameter(
            torch.randn(512, 128, device="cuda", dtype=torch.bfloat16)
        )
        model.expert_scalar.use_muon = False
        for param in [*model.experts, model.expert_scalar]:
            param.allreduce = False

    config = TransformerConfig(
        num_layers=1,
        hidden_size=64,
        num_attention_heads=1,
        bf16=True,
        num_moe_experts=2 if with_experts else None,
        expert_model_parallel_size=2 if with_experts else 1,
        expert_tensor_parallel_size=1,
    )
    ddp = wrap_model_chunks_with_ddp(
        [model],
        config,
        DistributedDataParallelConfig(grad_reduce_in_fp32=True),
        use_layer_wise_distributed_optimizer=True,
        use_layer_wise_param_layout=False,
        pg_collection=pg_collection,
    )[0]
    optimizer_config = OptimizerConfig(
        optimizer="muon",
        lr=0.001,
        bf16=True,
        clip_grad=0.0,
        use_distributed_optimizer=False,
        use_layer_wise_distributed_optimizer=True,
        use_layer_wise_param_layout=False,
        muon_split_qkv=False,
        muon_tp_mode="duplicated",
    )
    optimizer = get_megatron_optimizer(
        optimizer_config, [ddp], pg_collection=pg_collection, use_gloo_process_groups=False
    )
    assert type(optimizer) is ChainedOptimizer
    assert isinstance(optimizer.chained_optimizers[0], LayerWiseDistributedOptimizer)
    assert not optimizer.config.use_distributed_optimizer
    assert len(_dist_optimizers(optimizer)) == (2 if with_experts else 1)
    assert all(child.data_parallel_group_gloo is None for child in _dist_optimizers(optimizer))
    return ddp, optimizer


def _step(ddp, optimizer, step):
    ddp.zero_grad_buffer()
    optimizer.zero_grad()
    rank = torch.distributed.get_rank()
    for index, param in enumerate(ddp.parameters()):
        grad = torch.arange(param.numel(), device=param.device, dtype=torch.float32)
        grad = ((grad % 17) + index + step + rank) / 1024
        param.main_grad.copy_(grad.reshape_as(param))
    ddp.finish_grad_sync()
    assert optimizer.step()[0]


@pytest.mark.parametrize("expert_parallel_size", [1, 2])
def test_compact_torch_checkpoint_restores_shards_and_next_step(tmp_path, expert_parallel_size):
    """Every DP/expert-DP owner resumes exact masters, moments, step and next update."""
    Utils.initialize_model_parallel(expert_model_parallel_size=expert_parallel_size)
    try:
        model, optimizer = _build_compact_optimizer(1234, expert_parallel_size > 1)
        _step(model, optimizer, 1)
        _step(model, optimizer, 2)
        saved_model = deepcopy(model.module.state_dict())
        saved_optimizer = _snapshot(optimizer)
        path = tmp_path / f"layer_wise_optimizer_{torch.distributed.get_rank()}.pt"
        optimizer.save_state_dict_to_file(path)

        saved_file = torch.load(path, map_location="cpu", weights_only=False)
        for state, child in zip(saved_file[1:], _dist_optimizers(optimizer)):
            assert state["param_state_sharding_type"] == "dp_reshardable"
            _assert_equal(
                state["param_state"]["per_bucket_numel_unpadded"], child.per_bucket_numel_unpadded
            )
            saw_real_shard = False
            for buffer_idx in range(len(child.gbuf_ranges)):
                for buckets in state["param_state"][buffer_idx].values():
                    for bucket in buckets:
                        for shard in bucket:
                            saw_real_shard = True
                            assert shard["padding"] is False
                            assert shard["param"].dtype == torch.float32
                            assert torch.count_nonzero(shard["exp_avg"]) > 0
                            assert torch.count_nonzero(shard["exp_avg_sq"]) > 0
            assert saw_real_shard
            # Detect a loader that reconstructs Adam masters from rounded BF16 model values.
            assert any(
                not torch.equal(tensors["param"], tensors["param"].bfloat16().float())
                for tensors in (
                    child._get_main_param_and_optimizer_states(param)
                    for param in child.model_param_group_index_map
                )
            )

        # Old full-parameter Adam checkpoints have no DistOpt shard payload. Reject
        # them before loading even the LayerWise child, rather than silently losing
        # Adam's FP32 masters and moments. Check both dense and expert-DP siblings.
        incomplete_path = tmp_path / f"incomplete_optimizer_{torch.distributed.get_rank()}.pt"
        for optimizer_index in range(1, len(saved_file)):
            incomplete_state = deepcopy(saved_file)
            del incomplete_state[optimizer_index]["param_state"]
            torch.save(incomplete_state, incomplete_path)
            with pytest.raises(RuntimeError, match="missing DistributedOptimizer parameter shards"):
                optimizer.load_state_dict_from_file(incomplete_path)
            _assert_equal(_snapshot(optimizer), saved_optimizer)

        _step(model, optimizer, 3)
        expected_next = _snapshot(optimizer)
        expected_model = deepcopy(model.module.state_dict())

        resumed_model, resumed_optimizer = _build_compact_optimizer(9876, expert_parallel_size > 1)
        resumed_model.module.load_state_dict(saved_model)
        resumed_optimizer.load_state_dict_from_file(path)
        _assert_equal(_snapshot(resumed_optimizer), saved_optimizer)
        _step(resumed_model, resumed_optimizer, 3)
        _assert_equal(_snapshot(resumed_optimizer), expected_next)
        _assert_equal(resumed_model.module.state_dict(), expected_model)
    finally:
        Utils.destroy_model_parallel()


def _layerwise_with_two_inner_optimizers():
    config = OptimizerConfig(optimizer="sgd", lr=0.1)
    children = []
    for value in (1.0, 2.0):
        param = torch.nn.Parameter(torch.tensor([value]))
        inner = torch.optim.SGD([param], lr=0.1, momentum=0.9)
        param.grad = torch.tensor([value + 1.0])
        inner.step()
        children.append(FP32Optimizer(inner, config, init_state_fn=None))
    optimizer = object.__new__(LayerWiseDistributedOptimizer)
    ChainedOptimizer.__init__(optimizer, children)
    return optimizer


@pytest.mark.parametrize("nested", [False, True])
def test_legacy_layerwise_file_and_nested_list_remain_supported(tmp_path, nested):
    """Keep bare LayerWise's old list format and accept it inside the new outer chain."""
    layerwise = _layerwise_with_two_inner_optimizers()
    optimizer = ChainedOptimizer([layerwise]) if nested else layerwise
    expected = deepcopy(optimizer.state_dict())
    path = tmp_path / "layer_wise_optimizer.pt"
    optimizer.save_state_dict_to_file(path)
    _assert_equal(torch.load(path, weights_only=False), expected)
    for child in layerwise.chained_optimizers:
        for state in child.optimizer.state.values():
            state["momentum_buffer"].zero_()
    optimizer.load_state_dict_from_file(path)
    _assert_equal(optimizer.state_dict(), expected)
