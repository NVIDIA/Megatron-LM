# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import os
from contextlib import nullcontext
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

import megatron.training.tensor_metrics.observer as observer_mod
from megatron.core.optimizer import (
    ChainedOptimizer,
    DistributedOptimizer,
    Float16OptimizerWithFloat16Params,
    FP32Optimizer,
    LayerWiseDistributedOptimizer,
)
from megatron.core.parameter_names import CanonicalParameterNameMap
from megatron.training.tensor_metrics import (
    FlatShard,
    MetricSite,
    MetricTensor,
    Owned,
    RankRelation,
    Replica,
    Shard,
    TensorMetricExecutor,
)
from megatron.training.tensor_metrics.definitions import MeanRowL2NormMetric
from megatron.training.tensor_metrics.observer import build_tensor_metric_observer
from megatron.training.tensor_metrics.optimizer_sources import (
    _build_optimizer_parameter_manifest,
    _local_optimizer_tensor_views,
    _optimizer_metric_tensors,
    _optimizer_tensor_views,
)


class _ProcessGroup:
    def __init__(self, size=1, rank=0):
        self._size = size
        self._rank = rank

    def size(self):
        return self._size

    def rank(self):
        return self._rank


@pytest.fixture(autouse=True)
def _disable_tensor_metric_profiling(monkeypatch):
    monkeypatch.setattr(
        observer_mod, "get_nvtx_range", lambda: lambda *args, **kwargs: nullcontext()
    )


def _pg_collection(
    dp_size=1,
    dp_rank=0,
    expert_dp_size=1,
    expert_dp_rank=0,
    ep_size=1,
    ep_rank=0,
):
    return SimpleNamespace(
        tp=_ProcessGroup(),
        expt_tp=_ProcessGroup(),
        ep=_ProcessGroup(ep_size, ep_rank),
        dp_cp=_ProcessGroup(dp_size, dp_rank),
        expt_dp=_ProcessGroup(expert_dp_size, expert_dp_rank),
    )


def _float16_optimizer(model_parameter, main_parameter):
    optimizer = object.__new__(Float16OptimizerWithFloat16Params)
    optimizer.is_stub_optimizer = False
    optimizer.float16_groups = [[model_parameter]]
    optimizer.fp32_from_float16_groups = [[main_parameter]]
    optimizer.fp32_from_fp32_groups = [[]]
    return optimizer


def _fp32_optimizer(model_parameter):
    optimizer = object.__new__(FP32Optimizer)
    optimizer.is_stub_optimizer = False
    optimizer.optimizer = SimpleNamespace(param_groups=[{"params": [model_parameter]}])
    return optimizer


def _distributed_optimizer(model_parameter, main_parameter, start, end):
    optimizer = object.__new__(DistributedOptimizer)
    optimizer.is_stub_optimizer = False
    optimizer.model_float16_groups = [[model_parameter]]
    optimizer.shard_fp32_from_float16_groups = [[main_parameter]]
    optimizer.shard_float16_groups = [[model_parameter.detach().view(-1)[start:end]]]
    optimizer.model_fp32_groups = [[]]
    optimizer.shard_fp32_groups = [[]]
    optimizer._get_model_param_range_map = lambda parameter: {
        "param": SimpleNamespace(start=start, end=end)
    }
    return optimizer


def _chained_optimizer(*children):
    optimizer = object.__new__(ChainedOptimizer)
    optimizer.chained_optimizers = list(children)
    return optimizer


def _stub_optimizer(optimizer_type):
    optimizer = object.__new__(optimizer_type)
    optimizer.is_stub_optimizer = True
    return optimizer


def _named_model(**parameters):
    model = torch.nn.Module()
    for name, parameter in parameters.items():
        model.register_parameter(name, parameter)
    return model


def test_float16_optimizer_source_uses_main_parameter_and_finalized_wgrad():
    model_parameter = torch.nn.Parameter(torch.tensor([1.0, 2.0], dtype=torch.bfloat16))
    main_parameter = torch.nn.Parameter(torch.tensor([1.0, 2.0]))
    model_parameter.main_grad = torch.tensor([3.0, 4.0])

    views = _local_optimizer_tensor_views(
        _float16_optimizer(model_parameter, main_parameter), _pg_collection(dp_size=2)
    )

    assert len(views) == 1
    assert views[0].model_parameter is model_parameter
    assert views[0].parameter.data_ptr() == main_parameter.data_ptr()
    assert views[0].wgrad.data_ptr() == model_parameter.main_grad.data_ptr()
    assert views[0].storage_relations[0].axis == "dp"
    assert views[0].storage_relations[0].placement == Replica()


def test_fp32_optimizer_source_prefers_main_grad_over_grad():
    model_parameter = torch.nn.Parameter(torch.tensor([1.0, 2.0]))
    model_parameter.main_grad = torch.tensor([3.0, 4.0])
    model_parameter.grad = torch.tensor([5.0, 6.0])

    views = _local_optimizer_tensor_views(
        _fp32_optimizer(model_parameter), _pg_collection(dp_size=2)
    )

    assert views[0].parameter.data_ptr() == model_parameter.data_ptr()
    assert views[0].wgrad.data_ptr() == model_parameter.main_grad.data_ptr()
    assert views[0].storage_relations[0].placement == Replica()


def test_distributed_optimizer_source_uses_only_valid_local_ranges():
    model_parameter = torch.nn.Parameter(torch.tensor([[1.0, 2.0], [3.0, 4.0]]))
    main_parameter = torch.nn.Parameter(torch.tensor([30.0, 40.0]))
    model_parameter.main_grad = torch.tensor([[10.0, 20.0], [30.0, 40.0]])
    optimizer = _distributed_optimizer(model_parameter, main_parameter, 2, 4)

    views = _local_optimizer_tensor_views(optimizer, _pg_collection(dp_size=2, dp_rank=1))

    assert len(views) == 1
    assert views[0].model_parameter is model_parameter
    assert views[0].parameter.data_ptr() == main_parameter.data_ptr()
    torch.testing.assert_close(views[0].wgrad, torch.tensor([30.0, 40.0]))
    assert views[0].storage_relations[0].axis == "dp"
    assert views[0].storage_relations[0].placement == FlatShard((2, 2), 2, 4)


def test_distributed_optimizer_source_uses_expert_dp_axis():
    model_parameter = torch.nn.Parameter(torch.tensor([1.0, 2.0]))
    model_parameter.allreduce = False
    main_parameter = torch.nn.Parameter(torch.tensor([1.0]))
    model_parameter.main_grad = torch.tensor([3.0, 4.0])
    optimizer = _distributed_optimizer(model_parameter, main_parameter, 0, 1)

    views = _local_optimizer_tensor_views(optimizer, _pg_collection(expert_dp_size=2))

    assert views[0].storage_relations[0].axis == "expert_dp"
    assert views[0].storage_relations[0].placement == FlatShard((2,), 0, 1)


def test_layerwise_optimizer_source_marks_whole_parameter_owner():
    remote_parameter = torch.nn.Parameter(torch.tensor([0.0]))
    model_parameter = torch.nn.Parameter(torch.tensor([1.0, 2.0], dtype=torch.bfloat16))
    main_parameter = torch.nn.Parameter(torch.tensor([1.0, 2.0]))
    model_parameter.main_grad = torch.tensor([3.0, 4.0])
    optimizer = object.__new__(LayerWiseDistributedOptimizer)
    optimizer.chained_optimizers = [_float16_optimizer(model_parameter, main_parameter)]
    optimizer.dp_cp_params_list = [[remote_parameter], [model_parameter]]
    optimizer.expt_dp_params_list = None

    views = _local_optimizer_tensor_views(
        optimizer, _pg_collection(dp_size=2, dp_rank=1, expert_dp_size=2)
    )

    assert len(views) == 1
    assert views[0].parameter.data_ptr() == main_parameter.data_ptr()
    assert views[0].wgrad.data_ptr() == model_parameter.main_grad.data_ptr()
    assert views[0].storage_relations[0].axis == "dp"
    assert views[0].storage_relations[0].placement == Owned(1)


def test_layerwise_optimizer_source_marks_expert_owner():
    model_parameter = torch.nn.Parameter(torch.tensor([1.0]))
    model_parameter.allreduce = False
    model_parameter.main_grad = torch.tensor([2.0])
    optimizer = object.__new__(LayerWiseDistributedOptimizer)
    optimizer.chained_optimizers = [_fp32_optimizer(model_parameter)]
    optimizer.dp_cp_params_list = None
    optimizer.expt_dp_params_list = [[], [model_parameter]]

    views = _local_optimizer_tensor_views(
        optimizer, _pg_collection(dp_size=2, expert_dp_size=2, expert_dp_rank=1)
    )

    assert views[0].storage_relations[0].axis == "expert_dp"
    assert views[0].storage_relations[0].placement == Owned(1)


def test_chained_optimizer_source_combines_children_and_deduplicates_model_parameter():
    first = torch.nn.Parameter(torch.tensor([1.0]))
    second = torch.nn.Parameter(torch.tensor([2.0]))
    first_optimizer = _fp32_optimizer(first)
    second_optimizer = _fp32_optimizer(second)
    duplicate_optimizer = _fp32_optimizer(first)

    views = _local_optimizer_tensor_views(
        _chained_optimizer(first_optimizer, second_optimizer, duplicate_optimizer),
        _pg_collection(),
    )

    assert [id(view.model_parameter) for view in views] == [id(first), id(second)]


def test_stub_optimizer_source_is_empty():
    optimizer = object.__new__(FP32Optimizer)
    optimizer.is_stub_optimizer = True

    assert _local_optimizer_tensor_views(optimizer, _pg_collection()) == []


def test_layerwise_optimizer_rejects_parameter_without_owner():
    model_parameter = torch.nn.Parameter(torch.tensor([1.0]))
    optimizer = object.__new__(LayerWiseDistributedOptimizer)
    optimizer.chained_optimizers = [_fp32_optimizer(model_parameter)]
    optimizer.dp_cp_params_list = [[], []]
    optimizer.expt_dp_params_list = None

    with pytest.raises(ValueError, match="has no owner"):
        _local_optimizer_tensor_views(optimizer, _pg_collection(dp_size=2))


def test_layerwise_optimizer_rejects_parameter_on_non_owner_rank():
    model_parameter = torch.nn.Parameter(torch.tensor([1.0]))
    optimizer = object.__new__(LayerWiseDistributedOptimizer)
    optimizer.chained_optimizers = [_fp32_optimizer(model_parameter)]
    optimizer.dp_cp_params_list = [[], [model_parameter]]
    optimizer.expt_dp_params_list = None

    with pytest.raises(ValueError, match="other than its owner"):
        _local_optimizer_tensor_views(optimizer, _pg_collection(dp_size=2, dp_rank=0))


def test_rank_symmetric_distributed_optimizer_source_adds_empty_flat_shard():
    local_parameter = torch.nn.Parameter(torch.tensor([1.0, 2.0]))
    missing_parameter = torch.nn.Parameter(torch.tensor([[3.0, 4.0], [5.0, 6.0]]))
    local_parameter.main_grad = torch.tensor([7.0, 8.0])
    optimizer = _distributed_optimizer(
        local_parameter, torch.nn.Parameter(torch.tensor([1.0, 2.0])), 0, 2
    )
    parameter_names = CanonicalParameterNameMap(
        _named_model(z_local=local_parameter, a_missing=missing_parameter)
    )

    views = _optimizer_tensor_views(
        parameter_names, optimizer, _pg_collection(dp_size=2)
    )

    assert [parameter_names[view.model_parameter] for view in views] == [
        "a_missing",
        "z_local",
    ]
    assert views[0].parameter.numel() == 0
    assert views[0].wgrad.numel() == 0
    assert not views[0].is_placeholder
    assert len(views[0].storage_relations) == 1
    assert views[0].storage_relations[0].axis == "dp"
    assert views[0].storage_relations[0].placement == FlatShard((2, 2), 0, 0)
    assert views[1].storage_relations[0].placement == FlatShard((2,), 0, 2)


def test_rank_symmetric_layerwise_source_preserves_remote_owner():
    remote_parameter = torch.nn.Parameter(torch.tensor([1.0]))
    local_parameter = torch.nn.Parameter(torch.tensor([2.0]))
    local_parameter.main_grad = torch.tensor([3.0])
    optimizer = object.__new__(LayerWiseDistributedOptimizer)
    optimizer.chained_optimizers = [_fp32_optimizer(local_parameter)]
    optimizer.dp_cp_params_list = [[remote_parameter], [local_parameter]]
    optimizer.expt_dp_params_list = None
    parameter_names = CanonicalParameterNameMap(
        _named_model(remote=remote_parameter, local=local_parameter)
    )

    views = _optimizer_tensor_views(
        parameter_names, optimizer, _pg_collection(dp_size=2, dp_rank=1)
    )
    views_by_parameter = {view.model_parameter: view for view in views}

    assert views_by_parameter[remote_parameter].parameter.numel() == 0
    assert views_by_parameter[remote_parameter].is_placeholder
    assert views_by_parameter[remote_parameter].storage_relations[0].placement == Owned(0)
    assert not views_by_parameter[local_parameter].is_placeholder
    assert views_by_parameter[local_parameter].storage_relations[0].placement == Owned(1)


def test_rank_symmetric_chained_source_routes_missing_parameter_to_layerwise_owner():
    matrix = torch.nn.Parameter(torch.ones(2, 2))
    scalar = torch.nn.Parameter(torch.ones(1))
    scalar.main_grad = torch.ones(1)
    layerwise = object.__new__(LayerWiseDistributedOptimizer)
    layerwise.chained_optimizers = [_stub_optimizer(FP32Optimizer)]
    layerwise.dp_cp_params_list = [[matrix], []]
    layerwise.expt_dp_params_list = None
    distributed = _distributed_optimizer(scalar, torch.nn.Parameter(torch.ones(1)), 0, 1)
    optimizer = _chained_optimizer(layerwise, distributed)
    parameter_names = CanonicalParameterNameMap(_named_model(matrix=matrix, scalar=scalar))

    views = _optimizer_tensor_views(
        parameter_names, optimizer, _pg_collection(dp_size=2, dp_rank=1)
    )
    views_by_parameter = {view.model_parameter: view for view in views}

    assert views_by_parameter[matrix].parameter.numel() == 0
    assert views_by_parameter[matrix].storage_relations[0].placement == Owned(0)
    assert views_by_parameter[scalar].storage_relations[0].placement == FlatShard((1,), 0, 1)


def test_rank_symmetric_source_rejects_missing_parameter_for_replicated_optimizer():
    local_parameter = torch.nn.Parameter(torch.tensor([1.0]))
    missing_parameter = torch.nn.Parameter(torch.tensor([2.0]))
    parameter_names = CanonicalParameterNameMap(
        _named_model(local=local_parameter, missing=missing_parameter)
    )

    with pytest.raises(ValueError, match="does not describe storage"):
        _optimizer_tensor_views(
            parameter_names, _fp32_optimizer(local_parameter), _pg_collection(dp_size=2)
        )


def test_optimizer_metric_tensors_compose_model_and_storage_placements():
    dense = torch.nn.Parameter(torch.tensor([1.0, 2.0]))
    dense.main_grad = torch.tensor([3.0, 4.0])
    dense.tensor_model_parallel = True
    dense.partition_dim = 0
    parameter_names = CanonicalParameterNameMap(_named_model(weight=dense))
    pg_collection = _pg_collection(dp_size=2)

    values = _optimizer_metric_tensors(
        parameter_names,
        _distributed_optimizer(dense, torch.nn.Parameter(torch.tensor([1.0])), 0, 1),
        pg_collection,
        [MeanRowL2NormMetric()],
    )

    assert [value.sites[0].kind for value in values] == ["parameter", "wgrad"]
    assert [relation.axis for relation in values[0].rank_relations] == ["tp", "ep", "dp"]
    assert values[0].relation("tp").placement == Shard(0)
    assert values[0].relation("ep").placement == Replica()
    assert values[0].relation("dp").placement == FlatShard((2,), 0, 1)
    assert all(not value.is_placeholder for value in values)
    torch.testing.assert_close(values[1].tensor, torch.tensor([3.0]))


def test_ep_manifest_materializes_remote_owned_parameter_slots(monkeypatch):
    parameter = torch.nn.Parameter(torch.tensor([1.0, 2.0]))
    parameter.allreduce = False
    parameter_names = CanonicalParameterNameMap(_named_model(local_expert=parameter))
    pg_collection = _pg_collection(ep_size=2, ep_rank=0)
    optimizer = _fp32_optimizer(parameter)

    monkeypatch.setattr(torch.distributed, "is_available", lambda: True)
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)

    def fake_all_gather_object(output, local_entries, group):
        assert group is pg_collection.ep
        assert len(local_entries) == 1
        remote_model_relations = tuple(
            RankRelation(relation.axis, Owned(1))
            if relation.axis == "ep"
            else relation
            for relation in local_entries[0].model_relations
        )
        remote_entry = replace(
            local_entries[0],
            name="remote_expert",
            logical_shape=(3, 4),
            ep_owner=1,
            model_relations=remote_model_relations,
        )
        output[:] = [local_entries, (remote_entry,)]

    monkeypatch.setattr(torch.distributed, "all_gather_object", fake_all_gather_object)

    manifest = _build_optimizer_parameter_manifest(
        parameter_names, optimizer, pg_collection
    )
    values = _optimizer_metric_tensors(
        parameter_names,
        optimizer,
        pg_collection,
        [MeanRowL2NormMetric()],
        manifest,
    )

    assert [entry.name for entry in manifest] == ["local_expert", "remote_expert"]
    assert manifest[1].logical_shape == (3, 4)
    assert manifest[1].ep_owner == 1
    remote_values = [
        value for value in values if value.sites[0].name == "remote_expert"
    ]
    assert len(remote_values) == 2
    assert all(not value.tensor.numel() for value in remote_values)
    assert all(value.is_placeholder for value in remote_values)
    assert remote_values[0].relation("ep").placement == Owned(1)


def test_ep_manifest_preserves_rank_local_dense_optimizer_shards(monkeypatch):
    parameter = torch.nn.Parameter(torch.tensor([1.0, 2.0]))
    parameter_names = CanonicalParameterNameMap(_named_model(weight=parameter))
    pg_collection = _pg_collection(dp_size=2, ep_size=2, ep_rank=0)
    optimizer = _distributed_optimizer(
        parameter, torch.nn.Parameter(torch.tensor([1.0])), 0, 1
    )

    monkeypatch.setattr(torch.distributed, "is_available", lambda: True)
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)

    def fake_all_gather_object(output, local_entries, group):
        assert group is pg_collection.ep
        remote_entry = replace(
            local_entries[0],
            storage_relations=(RankRelation("dp", FlatShard((2,), 1, 2)),),
        )
        output[:] = [local_entries, (remote_entry,)]

    monkeypatch.setattr(torch.distributed, "all_gather_object", fake_all_gather_object)

    manifest = _build_optimizer_parameter_manifest(
        parameter_names, optimizer, pg_collection
    )

    assert len(manifest) == 1
    assert manifest[0].storage_relations == (
        RankRelation("dp", FlatShard((2,), 0, 1)),
    )


def test_rank_symmetric_flat_shards_execute_same_collective_with_empty_ranks():
    launched_world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if launched_world_size < 2:
        pytest.skip("This test requires at least two distributed ranks.")
    if not torch.distributed.is_initialized():
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
        torch.distributed.init_process_group(backend="nccl")

    device = torch.device("cuda", torch.cuda.current_device())
    full_wgrad = torch.tensor([[3.0, 4.0, 0.0, 0.0], [0.0, 5.0, 12.0, 0.0]], device=device)
    model_parameter = torch.nn.Parameter(torch.zeros_like(full_wgrad))
    model_parameter.main_grad = full_wgrad
    rank = torch.distributed.get_rank()
    if rank == 0:
        optimizer = _distributed_optimizer(
            model_parameter, torch.nn.Parameter(torch.zeros(3, device=device)), 0, 3
        )
    elif rank == 1:
        optimizer = _distributed_optimizer(
            model_parameter, torch.nn.Parameter(torch.zeros(5, device=device)), 3, 8
        )
    else:
        optimizer = _stub_optimizer(DistributedOptimizer)
    parameter_names = CanonicalParameterNameMap(_named_model(weight=model_parameter))
    world = torch.distributed.group.WORLD

    views = _optimizer_tensor_views(
        parameter_names,
        optimizer,
        SimpleNamespace(dp_cp=world, expt_dp=world),
    )
    values = [
        MetricTensor(
            tensor=views[0].wgrad,
            sites=(MetricSite("weight", "wgrad"),),
            rank_relations=views[0].storage_relations,
        )
    ]
    results = TensorMetricExecutor({"dp": world}).run(MeanRowL2NormMetric(), values)

    torch.testing.assert_close(results[0].value.tensor, torch.tensor(9.0, device=device))


def test_observer_reduces_parameter_and_wgrad_flat_shards_end_to_end():
    launched_world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if launched_world_size < 2:
        pytest.skip("This test requires at least two distributed ranks.")
    if not torch.distributed.is_initialized():
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
        torch.distributed.init_process_group(backend="nccl")

    device = torch.device("cuda", torch.cuda.current_device())
    full_parameter = torch.tensor([[3.0, 4.0], [0.0, 0.0]], device=device)
    full_wgrad = torch.tensor([[0.0, 0.0], [5.0, 12.0]], device=device)
    model_parameter = torch.nn.Parameter(full_parameter)
    model_parameter.main_grad = full_wgrad
    model = torch.nn.Module()
    model.decoder = torch.nn.Module()
    model.decoder.layers = torch.nn.ModuleList([torch.nn.Module()])
    model.decoder.layers[0].register_parameter("weight", model_parameter)

    rank = torch.distributed.get_rank()
    world_size = torch.distributed.get_world_size()
    start = rank * model_parameter.numel() // world_size
    end = (rank + 1) * model_parameter.numel() // world_size
    if start < end:
        local_parameter = torch.nn.Parameter(full_parameter.view(-1)[start:end].clone())
        optimizer = _distributed_optimizer(model_parameter, local_parameter, start, end)
    else:
        optimizer = _stub_optimizer(DistributedOptimizer)
    world = torch.distributed.group.WORLD
    pg_collection = SimpleNamespace(
        tp=world,
        expt_tp=world,
        ep=world,
        dp_cp=world,
        expt_dp=world,
    )
    captured = {}
    observer = build_tensor_metric_observer(
        [
            "global-param-l2:1",
            "layer-param-l2:1",
            "global-wgrad-l2:1",
            "layer-wgrad-l2:1",
        ],
        result_sink=lambda metric, results, iteration: captured.update(
            {metric.name: (tuple(results), iteration)}
        ),
    )
    assert observer is not None

    observer(model=[model], optimizer=optimizer, iteration=0, pg_collection=pg_collection)

    for metric_name, expected in (
        ("global-param-l2", 5.0),
        ("layer-param-l2", 5.0),
        ("global-wgrad-l2", 13.0),
        ("layer-wgrad-l2", 13.0),
    ):
        results, iteration = captured[metric_name]
        assert iteration == 1
        expected_labels = {
            "global" if metric_name.startswith("global") else "decoder.layers.0"
        }
        if metric_name == "layer-wgrad-l2":
            expected_labels.add("global")
        results_by_label = {str(result.label): result.value.tensor for result in results}
        assert set(results_by_label) == expected_labels
        for result in results_by_label.values():
            torch.testing.assert_close(result, torch.tensor(expected, device=device))


def test_observer_reduces_ep_owned_expert_values_end_to_end():
    launched_world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if launched_world_size < 2:
        pytest.skip("This test requires at least two distributed ranks.")
    if not torch.distributed.is_initialized():
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
        torch.distributed.init_process_group(backend="nccl")

    device = torch.device("cuda", torch.cuda.current_device())
    rank = torch.distributed.get_rank()
    world_size = torch.distributed.get_world_size()
    model_parameter = torch.nn.Parameter(torch.tensor([float(rank + 1)], device=device))
    model_parameter.main_grad = torch.tensor([float(rank + 5)], device=device)
    model_parameter.allreduce = False
    model = torch.nn.Module()
    model.config = SimpleNamespace(num_moe_experts=world_size)
    model.decoder = torch.nn.Module()
    model.decoder.layers = torch.nn.ModuleList([torch.nn.Module()])
    model.decoder.layers[0].mlp = torch.nn.Module()
    model.decoder.layers[0].mlp.experts = torch.nn.Module()
    model.decoder.layers[0].mlp.experts.local_experts = torch.nn.ModuleList(
        [torch.nn.Module()]
    )
    model.decoder.layers[0].mlp.experts.local_experts[0].register_parameter(
        "weight", model_parameter
    )

    world = torch.distributed.group.WORLD
    pg_collection = SimpleNamespace(
        tp=world,
        expt_tp=world,
        ep=world,
        dp_cp=world,
        expt_dp=world,
    )
    captured = {}
    observer = build_tensor_metric_observer(
        [
            "global-param-l2:1",
            "layer-param-l2:1",
            "global-param-mean-row-l2:1",
            "global-param-mean-column-l2:1",
            "global-wgrad-l2:1",
            "layer-wgrad-l2:1",
        ],
        result_sink=lambda metric, results, iteration: captured.update(
            {metric.name: tuple(results)}
        ),
    )
    assert observer is not None

    observer(
        model=[model],
        optimizer=_fp32_optimizer(model_parameter),
        iteration=0,
        pg_collection=pg_collection,
    )

    expected_parameter = sum(float(index + 1) ** 2 for index in range(world_size)) ** 0.5
    expected_mean_parameter = sum(float(index + 1) for index in range(world_size)) / world_size
    expected_wgrad = sum(float(index + 5) ** 2 for index in range(world_size)) ** 0.5
    for metric_name, expected in (
        ("global-param-l2", expected_parameter),
        ("layer-param-l2", expected_parameter),
        ("global-param-mean-row-l2", expected_mean_parameter),
        ("global-param-mean-column-l2", expected_mean_parameter),
        ("global-wgrad-l2", expected_wgrad),
        ("layer-wgrad-l2", expected_wgrad),
    ):
        expected_labels = {
            "global" if metric_name.startswith("global") else "decoder.layers.0"
        }
        if metric_name == "layer-wgrad-l2":
            expected_labels.add("global")
        results_by_label = {
            str(result.label): result.value.tensor for result in captured[metric_name]
        }
        assert set(results_by_label) == expected_labels
        for result in results_by_label.values():
            torch.testing.assert_close(result, torch.tensor(expected, device=device))


def test_observer_reduces_layerwise_owned_values_end_to_end():
    launched_world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if launched_world_size < 2:
        pytest.skip("This test requires at least two distributed ranks.")
    if not torch.distributed.is_initialized():
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
        torch.distributed.init_process_group(backend="nccl")

    device = torch.device("cuda", torch.cuda.current_device())
    rank = torch.distributed.get_rank()
    world_size = torch.distributed.get_world_size()
    model = torch.nn.Module()
    model.decoder = torch.nn.Module()
    model.decoder.layers = torch.nn.ModuleList()
    parameters = []
    for index in range(world_size):
        layer = torch.nn.Module()
        parameter = torch.nn.Parameter(torch.tensor([float(index + 1)], device=device))
        parameter.main_grad = torch.tensor([float(2 * index + 1)], device=device)
        layer.register_parameter("weight", parameter)
        model.decoder.layers.append(layer)
        parameters.append(parameter)

    optimizer = object.__new__(LayerWiseDistributedOptimizer)
    optimizer.chained_optimizers = [_fp32_optimizer(parameters[rank])]
    optimizer.dp_cp_params_list = [[parameter] for parameter in parameters]
    optimizer.expt_dp_params_list = None
    world = torch.distributed.group.WORLD
    pg_collection = SimpleNamespace(
        tp=world,
        expt_tp=world,
        ep=world,
        dp_cp=world,
        expt_dp=world,
    )
    captured = {}
    observer = build_tensor_metric_observer(
        [
            "global-param-l2:1",
            "layer-param-l2:1",
            "global-wgrad-l2:1",
            "layer-wgrad-l2:1",
        ],
        result_sink=lambda metric, results, iteration: captured.update(
            {metric.name: tuple(results)}
        ),
    )
    assert observer is not None

    observer(model=[model], optimizer=optimizer, iteration=0, pg_collection=pg_collection)

    expected_parameter = sum(float(index + 1) ** 2 for index in range(world_size)) ** 0.5
    expected_wgrad = sum(float(2 * index + 1) ** 2 for index in range(world_size)) ** 0.5
    torch.testing.assert_close(
        captured["global-param-l2"][0].value.tensor,
        torch.tensor(expected_parameter, device=device),
    )
    torch.testing.assert_close(
        captured["global-wgrad-l2"][0].value.tensor,
        torch.tensor(expected_wgrad, device=device),
    )
    parameter_results = {
        result.label: result.value.tensor for result in captured["layer-param-l2"]
    }
    wgrad_results = {
        result.label: result.value.tensor for result in captured["layer-wgrad-l2"]
    }
    layer_labels = {f"decoder.layers.{index}" for index in range(world_size)}
    assert set(parameter_results) == layer_labels
    assert set(wgrad_results) == layer_labels | {"global"}
    for index in range(world_size):
        label = f"decoder.layers.{index}"
        torch.testing.assert_close(
            parameter_results[label], torch.tensor(float(index + 1), device=device)
        )
        torch.testing.assert_close(
            wgrad_results[label], torch.tensor(float(2 * index + 1), device=device)
        )
    torch.testing.assert_close(
        wgrad_results["global"], torch.tensor(expected_wgrad, device=device)
    )
