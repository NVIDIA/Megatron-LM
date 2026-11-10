# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from collections.abc import Sequence

import pytest
import torch

from megatron.training.tensor_metrics import (
    AllGather,
    AllReduce,
    CollectiveRequest,
    CollectiveStage,
    FlatShard,
    LogicalReductionMetric,
    MetricResult,
    MetricSite,
    MetricTensor,
    Owned,
    RankRelation,
    Replica,
    Shard,
    TensorMetric,
    TensorMetricExecutor,
)

SITE = MetricSite("decoder.layers.0.linear.weight", "parameter")


def _item(
    name: str,
    tensor: torch.Tensor,
    relations: Sequence[RankRelation] = (),
    *,
    is_placeholder: bool = False,
) -> MetricTensor:
    return MetricTensor(
        tensor, (MetricSite(name, "parameter"),), tuple(relations), is_placeholder=is_placeholder
    )


def _result_tensor(results: Sequence[MetricResult]) -> torch.Tensor:
    assert len(results) == 1
    return results[0].tensor


def _fake_distributed(monkeypatch, remote_values: Sequence[torch.Tensor]):
    values = iter(remote_values)
    calls = []
    monkeypatch.setattr(torch.distributed, "is_available", lambda: True)
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda group: 2)

    def all_reduce(tensor, op, group):
        calls.append((tensor.shape, op, group))
        remote_value = next(values).to(tensor)
        if op is torch.distributed.ReduceOp.MAX:
            torch.maximum(tensor, remote_value, out=tensor)
        else:
            tensor.add_(remote_value)

    monkeypatch.setattr(torch.distributed, "all_reduce", all_reduce)
    return calls


class _SumMetric(LogicalReductionMetric):
    def contribution(self, tensor):
        return tensor.float().sum()


class _IdentityMetric(LogicalReductionMetric):
    def __init__(self, reduction_op=torch.distributed.ReduceOp.SUM):
        self.reduction_op = reduction_op

    def contribution(self, tensor):
        return tensor


def test_metric_tensor_validates_and_updates_relations():
    value = MetricTensor(
        torch.ones(2), (SITE,), (RankRelation("tp", Shard(0)), RankRelation("dp", Replica()))
    )

    assert value.relation("tp").placement == Shard(0)
    assert value.with_placement("tp", Owned(0)).relation("tp").placement == Owned(0)
    assert value.relation("tp").placement == Shard(0)

    placeholder = MetricTensor(
        torch.empty(0),
        (SITE,),
        (RankRelation("ep", Owned(1)), RankRelation("dp", FlatShard((2,), 0, 1))),
        is_placeholder=True,
    )
    after_shard = placeholder.with_placement("dp", Replica())
    assert after_shard.is_placeholder
    assert not after_shard.with_placement("ep", Replica()).is_placeholder

    with pytest.raises(ValueError, match="each axis at most once"):
        MetricTensor(
            torch.ones(2), (SITE,), (RankRelation("tp", Shard(0)), RankRelation("tp", Replica()))
        )
    with pytest.raises(KeyError, match="no rank relation"):
        value.relation("missing")
    with pytest.raises(ValueError, match="must not be empty"):
        CollectiveStage((), None)


def test_flat_shard_and_metric_result_validate_their_contracts():
    assert FlatShard(torch.Size((2, 3)), 1, 5) == FlatShard((2, 3), 1, 5)
    with pytest.raises(ValueError, match="dimensions must be non-negative"):
        FlatShard((2, -1), 0, 1)
    with pytest.raises(ValueError, match="0 <= start <= end"):
        FlatShard((2, 3), 4, 3)

    tensor = torch.ones(1)
    assert MetricResult(tensor).label == "global"
    with pytest.raises(TypeError, match="must be a string"):
        MetricResult(tensor, 1)
    with pytest.raises(ValueError, match="must not be empty"):
        MetricResult(tensor, "")


def test_logical_reduction_aggregates_globally():
    values = [_item("a", torch.tensor([1.0, 2.0])), _item("b", torch.tensor([3.0]))]

    results = TensorMetricExecutor({}).run(_SumMetric(), values)

    assert results[0].label == "global"
    torch.testing.assert_close(_result_tensor(results), torch.tensor(6.0))


def test_logical_reduction_resolves_shards_before_owned_axes(monkeypatch):
    calls = _fake_distributed(monkeypatch, [torch.tensor([6.0]), torch.tensor([2.0])])
    tp_group = object()
    ep_group = object()
    value = _item(
        "weight",
        torch.tensor([1.0, 3.0]),
        (RankRelation("ep", Owned(0)), RankRelation("tp", Shard(0))),
    )

    result = _result_tensor(
        TensorMetricExecutor({"tp": tp_group, "ep": ep_group}).run(_SumMetric(), [value])
    )

    torch.testing.assert_close(result, torch.tensor(12.0))
    assert [call[2] for call in calls] == [tp_group, ep_group]


def test_executor_packs_compatible_collectives(monkeypatch):
    calls = _fake_distributed(monkeypatch, [torch.tensor([7.0, 11.0])])
    values = [
        _item("a", torch.tensor([1.0, 2.0]), (RankRelation("tp", Shard(0)),)),
        _item("b", torch.tensor([5.0]), (RankRelation("tp", Shard(0)),)),
    ]

    result = _result_tensor(TensorMetricExecutor({"tp": object()}).run(_SumMetric(), values))

    torch.testing.assert_close(result, torch.tensor(26.0))
    assert [call[0] for call in calls] == [torch.Size([2])]


@pytest.mark.parametrize(
    ("reduction_op", "left", "right", "expected"),
    (
        (torch.distributed.ReduceOp.SUM, 2, 3, 5),
        (torch.distributed.ReduceOp.PRODUCT, 2, 3, 6),
        (torch.distributed.ReduceOp.MIN, 2, 3, 2),
        (torch.distributed.ReduceOp.MAX, 2, 3, 3),
        (torch.distributed.ReduceOp.BAND, 6, 3, 2),
        (torch.distributed.ReduceOp.BOR, 6, 3, 7),
        (torch.distributed.ReduceOp.BXOR, 6, 3, 5),
    ),
)
def test_logical_reduction_combines_values_with_its_reduce_op(reduction_op, left, right, expected):
    values = [_item("left", torch.tensor(left)), _item("right", torch.tensor(right))]

    result = _result_tensor(TensorMetricExecutor({}).run(_IdentityMetric(reduction_op), values))

    torch.testing.assert_close(result, torch.tensor(expected))


def test_executor_detaches_prepare_inputs_and_outputs():
    input_leaf = torch.ones(2, requires_grad=True)
    input_with_history = input_leaf.square()
    output_with_history = input_leaf.sum()

    class GraphReturningMetric(TensorMetric):
        def prepare(self, values):
            assert not values[0].tensor.requires_grad
            assert values[0].tensor.grad_fn is None
            return [values[0].with_tensor(output_with_history)]

        def start(self, values):
            raise AssertionError("This test only exercises preparation.")

        def resume(self, values, continuation):
            raise AssertionError("This test only exercises preparation.")

    prepared = TensorMetricExecutor({}).prepare(
        GraphReturningMetric(), [_item("activation", input_with_history)]
    )

    assert not prepared[0].tensor.requires_grad
    assert prepared[0].tensor.grad_fn is None


class _SingularResumeMetric(TensorMetric):
    def __init__(self):
        self.resume_calls = 0

    def start(self, values):
        return [
            CollectiveStage((CollectiveRequest(value, "tp", AllReduce()),), f"result-{index}")
            for index, value in enumerate(values)
        ]

    def resume(self, values, continuation):
        self.resume_calls += 1
        return MetricResult(values[0].tensor, continuation)


def test_executor_resumes_each_computation_after_batching_collectives(monkeypatch):
    calls = _fake_distributed(monkeypatch, [torch.tensor([[2.0], [3.0]])])
    metric = _SingularResumeMetric()
    relations = (RankRelation("tp", Replica()),)
    values = [_item("a", torch.ones(1), relations), _item("b", torch.ones(1), relations)]

    results = TensorMetricExecutor({"tp": object()}).run(metric, values)

    torch.testing.assert_close(results[0].tensor, torch.tensor([3.0]))
    torch.testing.assert_close(results[1].tensor, torch.tensor([4.0]))
    assert metric.resume_calls == 2
    assert calls[0][0] == torch.Size([2, 1])


def test_executor_rejects_incomplete_collective_results(monkeypatch):
    _fake_distributed(monkeypatch, [])
    executor = TensorMetricExecutor({"tp": object()})
    value = _item("a", torch.ones(1), (RankRelation("tp", Replica()),))
    request = CollectiveRequest(value, "tp", AllReduce())
    monkeypatch.setattr(executor, "_execute_compatible", lambda requests: [])

    with pytest.raises(RuntimeError, match="one value for every metric request"):
        executor._execute_requests([request])


def test_executor_rejects_nonstandard_all_reduce_operations(monkeypatch):
    _fake_distributed(monkeypatch, [])
    executor = TensorMetricExecutor({"tp": object()})
    value = _item("a", torch.ones(1), (RankRelation("tp", Replica()),))
    request = CollectiveRequest(value, "tp", AllReduce(object()))

    with pytest.raises(ValueError, match="standard PyTorch all-reduce operations"):
        executor._execute_requests([request])


def test_executor_collective_compatibility_keys_are_uniform_numeric_tuples():
    executor = TensorMetricExecutor({"tp": object()})
    value = _item("a", torch.ones(1), (RankRelation("tp", Replica()),))
    requests = (
        CollectiveRequest(value, "tp", AllReduce(torch.distributed.ReduceOp.MAX)),
        CollectiveRequest(value, "tp", AllGather()),
        CollectiveRequest(value, "tp", AllGather(0)),
    )

    collective_keys = [executor._compatibility_key(request)[1] for request in requests]

    assert all(len(key) == 4 for key in collective_keys)
    assert all(type(field) is int for key in collective_keys for field in key)
    assert collective_keys == sorted(collective_keys)


class _GatherMetric(TensorMetric):
    def start(self, values):
        return [
            CollectiveStage((CollectiveRequest(value, "tp", AllGather(0)),), "gathered")
            for value in values
        ]

    def resume(self, values, continuation):
        return MetricResult(values[0].tensor)


def test_executor_all_gathers_an_explicit_shard(monkeypatch):
    monkeypatch.setattr(torch.distributed, "is_available", lambda: True)
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda group: 2)

    def all_gather(outputs, tensor, group):
        outputs[0].copy_(tensor)
        outputs[1].copy_(torch.tensor([[3.0, 4.0]]))

    monkeypatch.setattr(torch.distributed, "all_gather", all_gather)
    value = _item("vector", torch.tensor([1.0, 2.0]), (RankRelation("tp", Shard(0)),))

    result = _result_tensor(TensorMetricExecutor({"tp": object()}).run(_GatherMetric(), [value]))

    torch.testing.assert_close(result, torch.tensor([1.0, 2.0, 3.0, 4.0]))


def test_collective_requires_an_explicit_process_group_mapping(monkeypatch):
    _fake_distributed(monkeypatch, [])
    value = _item("shard", torch.ones(1), (RankRelation("tp", Shard(0)),))

    with pytest.raises(ValueError, match="No process group"):
        TensorMetricExecutor({}).run(_SumMetric(), [value])
