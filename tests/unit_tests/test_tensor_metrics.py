# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from collections.abc import Sequence

import pytest
import torch

from megatron.core.transformer.moe.router_diagnostics import (
    ROUTER_DIAGNOSTIC_CHANNEL_COUNT,
    RouterDiagnosticChannel,
)
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
from megatron.training.tensor_metrics.definitions import (
    L2NormMetric,
    LayerL2NormMetric,
    LayerMaxMetric,
    LayerNormalizedEntropyMetric,
    LayerSampledMedianMetric,
    MeanColumnL2NormMetric,
    MeanRowL2NormMetric,
    _accumulation_dtype,
)
from megatron.training.tensor_metrics.router_metrics import (
    LayerRouterExpertBiasMetric,
    LayerRouterHealthMetric,
    LayerRouterRoutingBalanceMetric,
    LayerRouterSeqAuxDecompositionMetric,
)

SITE = MetricSite("decoder.layers.0.linear.weight", "parameter")


def test_accumulation_dtype_upcasts_float8():
    float8_dtypes = (
        torch.float8_e4m3fn,
        torch.float8_e4m3fnuz,
        torch.float8_e5m2,
        torch.float8_e5m2fnuz,
    )

    assert all(_accumulation_dtype(dtype) is torch.float32 for dtype in float8_dtypes)


def _item(
    name: str,
    tensor: torch.Tensor,
    relations: Sequence[RankRelation] = (),
    *,
    kind: str = "parameter",
    is_placeholder: bool = False,
) -> MetricTensor:
    return MetricTensor(
        tensor, (MetricSite(name, kind),), tuple(relations), is_placeholder=is_placeholder
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
        calls.append((tensor.shape, op, group, tensor.detach().clone()))
        remote_value = next(values).to(tensor)
        if op is torch.distributed.ReduceOp.MAX:
            torch.maximum(tensor, remote_value, out=tensor)
        else:
            tensor.add_(remote_value)

    monkeypatch.setattr(torch.distributed, "all_reduce", all_reduce)
    return calls


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

    with pytest.raises(ValueError, match="must not be empty"):
        CollectiveStage((), None)


def test_flat_shard_validates_logical_interval():
    assert FlatShard(torch.Size((2, 3)), 1, 5) == FlatShard((2, 3), 1, 5)

    with pytest.raises(ValueError, match="dimensions must be non-negative"):
        FlatShard((2, -1), 0, 1)
    with pytest.raises(ValueError, match="0 <= start <= end"):
        FlatShard((2, 3), 4, 3)
    with pytest.raises(ValueError, match="0 <= start <= end"):
        FlatShard((2, 3), 0, 7)


def test_metric_result_uses_a_nonempty_string_label():
    tensor = torch.ones(1)

    assert MetricResult(tensor).label == "global"
    with pytest.raises(TypeError, match="must be a string"):
        MetricResult(tensor, 1)
    with pytest.raises(ValueError, match="must not be empty"):
        MetricResult(tensor, "")


def test_default_logical_reduction_aggregates_globally():
    metric = L2NormMetric()
    values = [_item("a", torch.ones(1)), _item("b", torch.ones(1))]

    results = TensorMetricExecutor({}).run(metric, values)

    assert len(results) == 1
    assert results[0].label == "global"
    torch.testing.assert_close(results[0].tensor, torch.sqrt(torch.tensor(2.0)))
    assert metric.start([]) == []


@pytest.mark.parametrize("dtype", (torch.float16, torch.bfloat16, torch.float32, torch.float64))
def test_l2_norm_local(dtype):
    item = _item("weight", torch.tensor([3.0, 4.0], dtype=dtype))

    result = _result_tensor(TensorMetricExecutor({}).run(L2NormMetric(), [item]))

    assert result.dtype == (torch.float32 if dtype in (torch.float16, torch.bfloat16) else dtype)
    torch.testing.assert_close(result, torch.tensor(5.0, dtype=result.dtype))


def test_layer_max_is_exact_across_tensors_in_one_layer():
    items = [
        _item("decoder.layers.2.first", torch.tensor([-4.0, 3.0])),
        _item("decoder.layers.2.second", torch.tensor([7.0, 1.0])),
    ]

    results = TensorMetricExecutor({}).run(LayerMaxMetric(), items)

    assert results[0].label == "decoder.layers.2"
    torch.testing.assert_close(_result_tensor(results), torch.tensor(7.0))


def test_layer_normalized_entropy_averages_per_item_decision_entropy():
    item = _item("decoder.layers.2.router_scores", torch.tensor([[0.5, 0.5], [1.0, 0.0]]))

    results = TensorMetricExecutor({}).run(LayerNormalizedEntropyMetric(), [item])

    assert results[0].label == "decoder.layers.2"
    torch.testing.assert_close(_result_tensor(results), torch.tensor(0.5))


def test_layer_sampled_median_can_retain_the_full_local_population():
    items = [
        _item("decoder.layers.2.first", torch.tensor([9.0, 1.0])),
        _item("decoder.layers.2.second", torch.tensor([4.0])),
    ]

    results = TensorMetricExecutor({}).run(LayerSampledMedianMetric(sample_factor=1), items)

    assert results[0].label == "decoder.layers.2"
    torch.testing.assert_close(_result_tensor(results), torch.tensor(4.0))


def test_layer_sampled_median_defaults_to_about_one_percent():
    metric = LayerSampledMedianMetric()
    item = _item("decoder.layers.2.router_logits", torch.arange(201.0))

    first = metric.prepare([item])
    second = metric.prepare([item])

    assert first[0].tensor.numel() == 3
    torch.testing.assert_close(first[0].tensor, second[0].tensor)


def test_prepare_can_apply_tensor_aware_filtering_after_site_selection():
    class MatrixL2NormMetric(L2NormMetric):
        def prepare(self, values):
            return super().prepare([value for value in values if value.tensor.ndim == 2])

    items = [_item("vector", torch.tensor([100.0])), _item("matrix", torch.tensor([[3.0, 4.0]]))]

    results = TensorMetricExecutor({}).run(MatrixL2NormMetric(), items)

    torch.testing.assert_close(_result_tensor(results), torch.tensor(5.0))


def test_executor_packs_compatible_metric_collectives(monkeypatch):
    calls = _fake_distributed(monkeypatch, [torch.tensor([7.0, 11.0])])
    items = [
        _item("a", torch.tensor([3.0, 4.0]), (RankRelation("tp", Shard(0)),)),
        _item("b", torch.tensor([0.0, 5.0]), (RankRelation("tp", Shard(0)),)),
    ]

    results = TensorMetricExecutor({"tp": object()}).run(L2NormMetric(), items)

    assert [result.label for result in results] == ["global"]
    torch.testing.assert_close(results[0].tensor, torch.sqrt(torch.tensor(68.0)))
    assert len(calls) == 1
    assert calls[0][0] == torch.Size([2])


def test_l2_norm_reduces_each_contributing_axis(monkeypatch):
    calls = _fake_distributed(monkeypatch, [torch.tensor([11.0]), torch.tensor([13.0])])
    item = _item(
        "weight",
        torch.tensor([3.0, 4.0]),
        (RankRelation("tp", Shard(0)), RankRelation("ep", Owned(0))),
    )

    result = _result_tensor(
        TensorMetricExecutor({"tp": object(), "ep": object()}).run(L2NormMetric(), [item])
    )

    torch.testing.assert_close(result, torch.tensor(7.0))
    assert len(calls) == 2


def test_l2_norm_reduces_flat_shard(monkeypatch):
    calls = _fake_distributed(monkeypatch, [torch.tensor([25.0])])
    item = _item("weight", torch.tensor([12.0]), (RankRelation("dp", FlatShard((1, 2), 1, 2)),))

    result = _result_tensor(TensorMetricExecutor({"dp": object()}).run(L2NormMetric(), [item]))

    torch.testing.assert_close(result, torch.tensor(13.0))
    assert len(calls) == 1


def _named_item(name, kind, tensor, relations=()):
    return MetricTensor(tensor, (MetricSite(name, kind),), tuple(relations))


def test_layer_l2_norm_metric_reduces_selected_parameters_by_layer():
    items = [
        _named_item(
            "decoder.layers.0.self_attention.linear_qkv.weight",
            "parameter",
            torch.tensor([3.0, 4.0]),
        ),
        _named_item(
            "decoder.layers.0.self_attention.linear_proj.weight",
            "parameter",
            torch.tensor([0.0, 5.0]),
        ),
        _named_item(
            "decoder.layers.1.self_attention.linear_qkv.weight",
            "parameter",
            torch.tensor([6.0, 8.0]),
        ),
        _named_item("embedding.word_embeddings.weight", "parameter", torch.tensor([100.0])),
        _named_item("decoder.layers.0.self_attention", "activation", torch.tensor([100.0])),
    ]

    results = TensorMetricExecutor({}).run(LayerL2NormMetric(), items)

    assert [result.label for result in results] == ["decoder.layers.0", "decoder.layers.1"]
    torch.testing.assert_close(results[0].tensor, torch.sqrt(torch.tensor(50.0)))
    torch.testing.assert_close(results[1].tensor, torch.tensor(10.0))


def test_layer_l2_norm_metric_can_add_an_exact_global_result():
    class LayerAndGlobalL2NormMetric(LayerL2NormMetric):
        include_global = True

    items = [
        _item("decoder.layers.0.weight", torch.tensor([3.0, 4.0])),
        _item("decoder.layers.1.weight", torch.tensor([0.0, 12.0])),
    ]

    results = TensorMetricExecutor({}).run(LayerAndGlobalL2NormMetric(), items)

    assert [result.label for result in results] == [
        "decoder.layers.0",
        "decoder.layers.1",
        "global",
    ]
    torch.testing.assert_close(results[0].tensor, torch.tensor(5.0))
    torch.testing.assert_close(results[1].tensor, torch.tensor(12.0))
    torch.testing.assert_close(results[2].tensor, torch.tensor(13.0))


def test_layer_l2_norm_metric_prepare_rejects_tensor_spanning_layers():
    item = MetricTensor(
        torch.tensor([3.0, 4.0]),
        (
            MetricSite("decoder.layers.0.weight", "parameter"),
            MetricSite("decoder.layers.1.weight", "parameter"),
        ),
    )

    results = TensorMetricExecutor({}).run(LayerL2NormMetric(), [item])

    assert results == []


def test_layer_l2_packs_tensor_states_for_distributed_reduction(monkeypatch):
    calls = _fake_distributed(monkeypatch, [torch.tensor([3.0, 11.0])])
    relations = (RankRelation("tp", Shard(0)),)
    items = [
        _named_item(
            "decoder.layers.0.self_attention.linear_qkv.weight",
            "parameter",
            torch.tensor([3.0, 4.0]),
            relations,
        ),
        _named_item(
            "decoder.layers.0.self_attention.linear_proj.weight",
            "parameter",
            torch.tensor([0.0, 5.0]),
            relations,
        ),
    ]

    results = TensorMetricExecutor({"tp": object()}).run(LayerL2NormMetric(), items)

    torch.testing.assert_close(_result_tensor(results), torch.tensor(8.0))
    assert results[0].label == "decoder.layers.0"
    assert len(calls) == 1
    assert calls[0][0] == torch.Size([2])


def test_layer_l2_metric_handles_heterogeneous_rank_relations(monkeypatch):
    remote_values = [torch.tensor([11.0, 7.0, 3.0]), torch.tensor([5.0])]
    calls = _fake_distributed(monkeypatch, remote_values)
    tp_group = object()
    ep_group = object()
    items = [
        _named_item(
            "decoder.layers.0.self_attention.linear_qkv.weight",
            "parameter",
            torch.tensor([3.0, 4.0]),
            (RankRelation("tp", Shard(0)),),
        ),
        _named_item(
            "decoder.layers.0.self_attention.linear_proj.weight",
            "parameter",
            torch.tensor([[0.0, 5.0]]),
            (RankRelation("tp", Shard(1)),),
        ),
        _named_item(
            "decoder.layers.0.self_attention.linear_proj.bias",
            "parameter",
            torch.tensor([2.0]),
            (RankRelation("tp", Replica()),),
        ),
        _named_item(
            "decoder.layers.0.mlp.experts.linear_fc1.weight",
            "parameter",
            torch.tensor([1.0]),
            (RankRelation("ep", Owned(0)), RankRelation("tp", Shard(0))),
        ),
    ]

    executor = TensorMetricExecutor({"tp": tp_group, "ep": ep_group})
    metric = LayerL2NormMetric()
    prepared = executor.prepare(metric, items)
    assert all(value.tensor.numel() == 1 for value in prepared)
    initial_steps = executor.start(metric, prepared)
    assert len(initial_steps) == 1
    assert isinstance(initial_steps[0], CollectiveStage)
    assert len(initial_steps[0].requests) == 3
    results = executor.complete(metric, initial_steps)

    torch.testing.assert_close(_result_tensor(results), torch.tensor(9.0))
    assert results[0].label == "decoder.layers.0"
    assert [call[0] for call in calls] == [torch.Size([3]), torch.Size([1])]
    assert [call[2] for call in calls] == [tp_group, ep_group]
    torch.testing.assert_close(calls[-1][3], torch.tensor([4.0]))


class _MeanMetric(LogicalReductionMetric):
    def __init__(self):
        self.batch_sizes = []

    def contribution(self, tensor):
        tensor = tensor.float()
        return torch.stack((tensor.sum(), tensor.new_tensor(tensor.numel())))

    def contribution_batch(self, tensors):
        self.batch_sizes.append(len(tensors))
        return super().contribution_batch(tensors)

    def finalize(self, contribution):
        return contribution[0] / contribution[1]


class _MaxMetric(LogicalReductionMetric):
    reduction_op = torch.distributed.ReduceOp.MAX

    def contribution(self, tensor):
        return tensor.amax()


class _IdentityMetric(LogicalReductionMetric):
    def __init__(self, reduction_op):
        self.reduction_op = reduction_op

    def contribution(self, tensor):
        return tensor


class _OrderedReductionMetric(_IdentityMetric):
    def combine_contributions(self, values, contributions):
        assert len(contributions) == len(values)
        return torch.stack(tuple(contributions))


def test_logical_reduction_metric_reduces_shards_then_replicates_owned_axes(monkeypatch):
    calls = _fake_distributed(monkeypatch, [torch.tensor([[6.0, 2.0]]), torch.tensor([[0.0, 0.0]])])
    tp_group = object()
    ep_group = object()
    item = _item(
        "mean",
        torch.tensor([1.0, 3.0]),
        (RankRelation("ep", Owned(0)), RankRelation("tp", Shard(0))),
    )

    results = TensorMetricExecutor({"tp": tp_group, "ep": ep_group}).run(_MeanMetric(), [item])

    torch.testing.assert_close(_result_tensor(results), torch.tensor(2.5))
    assert [call[2] for call in calls] == [tp_group, ep_group]


def test_logical_reduction_metric_restores_heterogeneous_branches_to_result_order(monkeypatch):
    calls = _fake_distributed(monkeypatch, [torch.tensor([[20.0]]), torch.tensor([[10.0]])])
    tp_group = object()
    ep_group = object()
    metric = _OrderedReductionMetric(torch.distributed.ReduceOp.SUM)
    items = [
        _item("tp", torch.tensor([1.0]), (RankRelation("tp", Shard(0)),)),
        _item("ep", torch.tensor([2.0]), (RankRelation("ep", Shard(0)),)),
    ]

    result = _result_tensor(
        TensorMetricExecutor({"tp": tp_group, "ep": ep_group}).run(metric, items)
    )

    torch.testing.assert_close(result, torch.tensor([[11.0], [22.0]]))
    assert [call[2] for call in calls] == [ep_group, tp_group]


def test_logical_reduction_metric_supports_other_reduce_ops(monkeypatch):
    calls = _fake_distributed(monkeypatch, [torch.tensor([5.0])])
    item = _item("maximum", torch.tensor([1.0, 4.0]), (RankRelation("tp", Shard(0)),))

    result = _result_tensor(TensorMetricExecutor({"tp": object()}).run(_MaxMetric(), [item]))

    torch.testing.assert_close(result, torch.tensor(5.0))
    assert calls[0][1] is torch.distributed.ReduceOp.MAX


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
def test_logical_reduction_metric_combines_values_with_reduction_op(
    reduction_op, left, right, expected
):
    metric = _IdentityMetric(reduction_op)
    items = [_item("left", torch.tensor(left)), _item("right", torch.tensor(right))]

    result = _result_tensor(TensorMetricExecutor({}).run(metric, items))

    torch.testing.assert_close(result, torch.tensor(expected))


def test_logical_reduction_metric_stacks_numeric_result_states(monkeypatch):
    original_stack = torch.stack
    calls = []

    def recorded_stack(tensors, dim=0):
        tensors = tuple(tensors)
        calls.append((len(tensors), dim))
        return original_stack(tensors, dim=dim)

    monkeypatch.setattr(torch, "stack", recorded_stack)
    metric = _IdentityMetric(torch.distributed.ReduceOp.SUM)
    items = [
        _item("first", torch.tensor(1.0)),
        _item("second", torch.tensor(2.0)),
        _item("third", torch.tensor(3.0)),
    ]

    result = _result_tensor(TensorMetricExecutor({}).run(metric, items))

    torch.testing.assert_close(result, torch.tensor(6.0))
    assert calls == [(3, 0)]


@pytest.mark.parametrize(
    "reduction_op", (torch.distributed.ReduceOp.SUM, torch.distributed.ReduceOp.PRODUCT)
)
def test_logical_reduction_metric_preserves_integer_state_dtype(reduction_op):
    metric = _IdentityMetric(reduction_op)
    items = [
        _item("left", torch.tensor(2, dtype=torch.int32)),
        _item("right", torch.tensor(3, dtype=torch.int32)),
    ]

    result = _result_tensor(TensorMetricExecutor({}).run(metric, items))

    assert result.dtype == torch.int32


def test_logical_reduction_metric_requires_override_for_unsupported_local_op():
    metric = _IdentityMetric(torch.distributed.ReduceOp.AVG)
    items = [_item("left", torch.tensor(2.0)), _item("right", torch.tensor(3.0))]

    with pytest.raises(ValueError, match="no default local multi-tensor implementation"):
        TensorMetricExecutor({}).run(metric, items)


def test_logical_reduction_metric_exposes_contribution_batch_hook():
    metric = _MeanMetric()
    items = [_item("a", torch.tensor([1.0, 3.0])), _item("b", torch.tensor([2.0, 6.0]))]

    results = TensorMetricExecutor({}).run(metric, items)

    torch.testing.assert_close(_result_tensor(results), torch.tensor(3.0))
    assert results[0].label == "global"
    assert metric.batch_sizes == [2]


def test_executor_can_prepare_locally_and_complete_steps_later(monkeypatch):
    calls = _fake_distributed(monkeypatch, [torch.tensor([11.0])])
    metric = L2NormMetric()
    item = _item("activation", torch.tensor([3.0, 4.0]), (RankRelation("tp", Shard(0)),))
    executor = TensorMetricExecutor({"tp": object()})

    prepared = executor.prepare(metric, [item])
    assert prepared[0].tensor.numel() == 1
    deferred_steps = executor.start(metric, prepared)
    assert len(deferred_steps) == 1
    assert isinstance(deferred_steps[0], CollectiveStage)
    assert deferred_steps[0].requests[0].value.tensor.numel() == 1

    result = _result_tensor(executor.complete(metric, deferred_steps))

    torch.testing.assert_close(result, torch.tensor(6.0))
    assert len(calls) == 1


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


class _QKVL2NormMetric(L2NormMetric):
    source_kinds = frozenset({"activation"})

    def accepts(self, site):
        return super().accepts(site) and ".linear_qkv" in site.name

    def start(self, values):
        return [self._start_logical_reduction(values, "all-qkv")] if values else []


def test_prepared_activations_from_separate_calls_can_be_aggregated_later():
    metric = _QKVL2NormMetric()
    executor = TensorMetricExecutor({})
    items = [
        _named_item(
            "decoder.layers.0.self_attention.linear_qkv", "activation", torch.tensor([3.0, 4.0])
        ),
        _named_item(
            "decoder.layers.0.self_attention.linear_proj", "activation", torch.tensor([100.0])
        ),
        _named_item(
            "decoder.layers.1.self_attention.linear_qkv", "activation", torch.tensor([6.0, 8.0])
        ),
    ]

    prepared = []
    for item in items:
        prepared.extend(executor.prepare(metric, [item]))

    assert len(prepared) == 2
    assert all(value.tensor.numel() == 1 for value in prepared)
    items[0].tensor.fill_(100.0)

    results = executor.complete(metric, executor.start(metric, prepared))

    torch.testing.assert_close(_result_tensor(results), torch.sqrt(torch.tensor(125.0)))
    assert results[0].label == "all-qkv"


def test_mean_row_l2_norm_local():
    item = _item("matrix", torch.tensor([[3.0, 4.0], [0.0, 5.0]]))

    result = _result_tensor(TensorMetricExecutor({}).run(MeanRowL2NormMetric(), [item]))

    torch.testing.assert_close(result, torch.tensor(5.0))


def test_mean_column_l2_norm_local():
    tensor = torch.tensor([[3.0, 4.0], [0.0, 5.0]])
    item = _item("matrix", tensor)

    result = _result_tensor(TensorMetricExecutor({}).run(MeanColumnL2NormMetric(), [item]))

    expected = torch.linalg.vector_norm(tensor, dim=0).mean()
    torch.testing.assert_close(result, expected)


def test_l2_norm_uses_one_fused_kernel_call_for_compatible_tensors(monkeypatch):
    te_optimizers = pytest.importorskip("transformer_engine.pytorch.optimizers")
    original_applier = te_optimizers.multi_tensor_applier
    calls = []

    def recorded_applier(operation, overflow_buffer, tensor_lists, *args):
        calls.append((len(tensor_lists[0]), args))
        return original_applier(operation, overflow_buffer, tensor_lists, *args)

    monkeypatch.setattr(te_optimizers, "multi_tensor_applier", recorded_applier)
    device = torch.device("cuda", torch.cuda.current_device())
    values = [
        _item("first", torch.tensor([3.0, 4.0], device=device)),
        _item("second", torch.tensor([0.0, 12.0], device=device)),
        _item("remote", torch.empty(0, device=device), (RankRelation("ep", Owned(1)),)),
    ]
    metric = L2NormMetric()
    executor = TensorMetricExecutor({})

    prepared = executor.prepare(metric, values)

    # The empty remote slot cannot be fused and does not join the kernel call.
    assert calls == [(2, (True,))]
    torch.testing.assert_close(
        torch.stack(tuple(value.tensor for value in prepared)),
        torch.tensor([25.0, 144.0, 0.0], device=device),
    )


def test_l2_norm_falls_back_per_tensor_for_inputs_the_fused_kernel_rejects():
    metric = L2NormMetric()
    tensors = (
        torch.tensor([3.0, 4.0]),
        torch.empty(0),
        torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float64),
        torch.tensor([5, 12], dtype=torch.int32),
    )

    contributions = metric.contribution_batch(tensors)

    assert [contribution.dtype for contribution in contributions] == [
        torch.float32,
        torch.float32,
        torch.float64,
        torch.float32,
    ]
    torch.testing.assert_close(
        torch.stack([contribution.to(torch.float64) for contribution in contributions]),
        torch.tensor([25.0, 0.0, 30.0, 169.0], dtype=torch.float64),
    )


def test_mean_row_l2_norm_combines_tensor_populations():
    items = [
        _item("expert-0", torch.tensor([[3.0, 4.0]])),
        _item("expert-1", torch.tensor([[0.0, 5.0], [0.0, 12.0]])),
    ]

    results = TensorMetricExecutor({}).run(MeanRowL2NormMetric(), items)

    torch.testing.assert_close(_result_tensor(results), torch.tensor(22.0 / 3.0))


def test_mean_column_l2_norm_combines_tensor_populations():
    items = [
        _item("expert-0", torch.tensor([[3.0, 0.0], [4.0, 0.0]])),
        _item("expert-1", torch.tensor([[12.0], [5.0]])),
    ]

    result = _result_tensor(TensorMetricExecutor({}).run(MeanColumnL2NormMetric(), items))

    torch.testing.assert_close(result, torch.tensor(6.0))


def test_dimwise_l2_exposes_all_ready_branches(monkeypatch):
    calls = _fake_distributed(
        monkeypatch,
        [torch.tensor([[13.0, 1.0]]), torch.tensor([[11.0]]), torch.tensor([[4.0, 1.0]])],
    )
    tp_group = object()
    ep_group = object()
    items = [
        _item(
            "expert-0",
            torch.tensor([[3.0, 4.0]]),
            (RankRelation("tp", Shard(1)), RankRelation("ep", Owned(0))),
        ),
        _item("expert-1", torch.tensor([[0.0, 5.0], [0.0, 12.0]]), (RankRelation("ep", Owned(0)),)),
    ]
    metric = MeanRowL2NormMetric()
    executor = TensorMetricExecutor({"tp": tp_group, "ep": ep_group})

    prepared = executor.prepare(metric, items)
    initial_steps = executor.start(metric, prepared)

    assert len(initial_steps) == 1
    assert isinstance(initial_steps[0], CollectiveStage)
    assert len(initial_steps[0].requests) == 2

    result = _result_tensor(executor.complete(metric, initial_steps))

    torch.testing.assert_close(result, torch.tensor(8.0))
    assert [call[0] for call in calls] == [
        torch.Size([1, 2]),
        torch.Size([1, 1]),
        torch.Size([1, 2]),
    ]
    assert [call[2] for call in calls] == [ep_group, tp_group, ep_group]


def test_dimwise_l2_packs_compatible_expert_branches(monkeypatch):
    calls = _fake_distributed(
        monkeypatch, [torch.tensor([[11.0], [24.0]]), torch.tensor([[4.0, 1.0], [11.0, 2.0]])]
    )
    tp_group = object()
    ep_group = object()
    relations = (RankRelation("tp", Shard(1)), RankRelation("ep", Owned(0)))
    items = [
        _item("expert-0", torch.tensor([[3.0, 4.0]]), relations),
        _item("expert-1", torch.tensor([[0.0, 5.0]]), relations),
    ]

    result = _result_tensor(
        TensorMetricExecutor({"tp": tp_group, "ep": ep_group}).run(MeanRowL2NormMetric(), items)
    )

    torch.testing.assert_close(result, torch.tensor(28.0 / 5.0))
    assert [call[0] for call in calls] == [torch.Size([2, 1]), torch.Size([2, 2])]
    assert [call[2] for call in calls] == [tp_group, ep_group]


def test_mean_row_l2_norm_uses_separate_norm_and_population_collectives(monkeypatch):
    remote_norm_squares = torch.tensor([[11.0]])
    remote_row_population = torch.tensor([[10.0, 1.0]])
    calls = _fake_distributed(monkeypatch, [remote_norm_squares, remote_row_population])
    item = _item(
        "matrix",
        torch.tensor([[3.0, 4.0]]),
        (RankRelation("tp", Shard(1)), RankRelation("cp", Shard(0))),
    )

    result = _result_tensor(
        TensorMetricExecutor({"tp": object(), "cp": object()}).run(MeanRowL2NormMetric(), [item])
    )

    expected = (torch.tensor(36.0).sqrt() + 10.0) / 2.0
    torch.testing.assert_close(result, expected)
    assert [call[0] for call in calls] == [torch.Size([1, 1]), torch.Size([1, 2])]


def test_mean_column_l2_norm_reduces_sharded_rows_before_sqrt(monkeypatch):
    calls = _fake_distributed(monkeypatch, [torch.tensor([[16.0, 0.0]])])
    item = _item("matrix", torch.tensor([[3.0, 4.0]]), (RankRelation("tp", Shard(0)),))

    result = _result_tensor(
        TensorMetricExecutor({"tp": object()}).run(MeanColumnL2NormMetric(), [item])
    )

    torch.testing.assert_close(result, torch.tensor(4.5))
    assert len(calls) == 1


def test_mean_row_l2_norm_reconstructs_distributed_optimizer_flat_shard(monkeypatch):
    calls = _fake_distributed(monkeypatch, [torch.tensor([[25.0, 0.0, 225.0]])])
    item = _item(
        "matrix",
        torch.tensor([0.0, 0.0, 0.0, 0.0, 5.0, 12.0, 8.0]),
        (RankRelation("dp", FlatShard((3, 4), 2, 9)),),
    )

    result = _result_tensor(
        TensorMetricExecutor({"dp": object()}).run(MeanRowL2NormMetric(), [item])
    )

    torch.testing.assert_close(result, torch.tensor(35.0 / 3.0))
    assert [call[0] for call in calls] == [torch.Size([1, 3])]


def test_mean_column_l2_norm_reconstructs_distributed_optimizer_flat_shard(monkeypatch):
    calls = _fake_distributed(monkeypatch, [torch.tensor([[9.0, 241.0, 0.0, 0.0]])])
    item = _item(
        "matrix",
        torch.tensor([0.0, 0.0, 0.0, 0.0, 5.0, 12.0, 8.0]),
        (RankRelation("dp", FlatShard((3, 4), 2, 9)),),
    )

    result = _result_tensor(
        TensorMetricExecutor({"dp": object()}).run(MeanColumnL2NormMetric(), [item])
    )

    expected = (torch.sqrt(torch.tensor(73.0)) + torch.sqrt(torch.tensor(241.0)) + 17.0) / 4.0
    torch.testing.assert_close(result, expected)
    assert [call[0] for call in calls] == [torch.Size([1, 4])]


def test_mean_row_l2_norm_composes_flat_dp_and_column_tp_shards(monkeypatch):
    tp_group = object()
    dp_group = object()
    calls = _fake_distributed(
        monkeypatch, [torch.tensor([[9.0, 25.0]]), torch.tensor([[0.0, 120.0]])]
    )
    item = _item(
        "matrix",
        torch.tensor([4.0, 12.0]),
        (RankRelation("tp", Shard(1)), RankRelation("dp", FlatShard((2, 2), 1, 3))),
    )

    result = _result_tensor(
        TensorMetricExecutor({"tp": tp_group, "dp": dp_group}).run(MeanRowL2NormMetric(), [item])
    )

    torch.testing.assert_close(result, torch.tensor(11.0))
    assert [call[2] for call in calls] == [tp_group, dp_group]


def test_dimwise_l2_norm_rejects_flat_shard_length_mismatch():
    item = _item("matrix", torch.ones(3), (RankRelation("dp", FlatShard((2, 2), 0, 2)),))

    with pytest.raises(ValueError, match="interval length"):
        TensorMetricExecutor({"dp": object()}).run(MeanRowL2NormMetric(), [item])


@pytest.mark.parametrize("metric", (MeanRowL2NormMetric(), MeanColumnL2NormMetric()))
def test_empty_owned_dimwise_l2_contributes_zero_population(monkeypatch, metric):
    calls = _fake_distributed(monkeypatch, [torch.tensor([10.0, 2.0])])
    item = _item("remote", torch.empty(0), (RankRelation("ep", Owned(1)),), is_placeholder=True)

    result = _result_tensor(TensorMetricExecutor({"ep": object()}).run(metric, [item]))

    torch.testing.assert_close(result, torch.tensor(5.0))
    torch.testing.assert_close(calls[0][3], torch.zeros(1, 2))


@pytest.mark.parametrize("metric", (MeanRowL2NormMetric(), MeanColumnL2NormMetric()))
def test_empty_owned_flat_shard_reduces_storage_before_owner(monkeypatch, metric):
    expert_dp_group = object()
    ep_group = object()
    calls = _fake_distributed(monkeypatch, [torch.zeros(1, 2), torch.tensor([[10.0, 2.0]])])
    item = _item(
        "remote",
        torch.empty(0),
        (RankRelation("ep", Owned(1)), RankRelation("expert_dp", FlatShard((2, 2), 0, 2))),
        is_placeholder=True,
    )

    result = _result_tensor(
        TensorMetricExecutor({"ep": ep_group, "expert_dp": expert_dp_group}).run(metric, [item])
    )

    torch.testing.assert_close(result, torch.tensor(5.0))
    assert [call[2] for call in calls] == [expert_dp_group, ep_group]
    torch.testing.assert_close(calls[0][3], torch.zeros(1, 2))


@pytest.mark.parametrize(
    ("metric", "remote_norm_state", "expected"),
    (
        (MeanRowL2NormMetric(), torch.tensor([[25.0, 25.0, 144.0]]), 22.0 / 3.0),
        (MeanColumnL2NormMetric(), torch.tensor([[169.0, 144.0]]), 12.5),
    ),
)
def test_empty_local_owned_flat_shard_resolves_norm_before_population(
    monkeypatch, metric, remote_norm_state, expected
):
    expert_dp_group = object()
    ep_group = object()
    calls = _fake_distributed(monkeypatch, [remote_norm_state, torch.zeros(1, 2)])
    item = _item(
        "local-owner",
        torch.empty(0),
        (RankRelation("ep", Owned(0)), RankRelation("expert_dp", FlatShard((3, 2), 0, 0))),
    )

    result = _result_tensor(
        TensorMetricExecutor({"ep": ep_group, "expert_dp": expert_dp_group}).run(metric, [item])
    )

    torch.testing.assert_close(result, torch.tensor(expected))
    assert [call[0] for call in calls] == [remote_norm_state.shape, torch.Size([1, 2])]
    assert [call[2] for call in calls] == [expert_dp_group, ep_group]


@pytest.mark.parametrize("metric", (MeanRowL2NormMetric(), MeanColumnL2NormMetric()))
def test_dimwise_l2_norm_rejects_unknown_shard_dimension(metric):
    item = _item("matrix", torch.ones(2, 2), (RankRelation("tp", Shard(None)),))

    with pytest.raises(ValueError, match="requires the dimension"):
        TensorMetricExecutor({"tp": object()}).run(metric, [item])


class _BatchStartMetric(TensorMetric):
    def __init__(self):
        self.batch_sizes = []

    def start(self, values):
        self.batch_sizes.append(len(values))
        return [MetricResult(value.tensor, value.sites[0].name) for value in values]

    def resume(self, values, continuation):
        raise AssertionError("This metric does not request collectives.")


def test_metric_start_receives_all_prepared_values():
    metric = _BatchStartMetric()
    items = [_item("a", torch.ones(1)), _item("b", torch.ones(1))]

    results = TensorMetricExecutor({}).run(metric, items)

    assert [result.label for result in results] == ["a", "b"]
    assert metric.batch_sizes == [2]


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


def test_executor_resumes_each_computation_independently_after_batching_collectives(monkeypatch):
    calls = _fake_distributed(monkeypatch, [torch.tensor([[2.0], [3.0]])])
    metric = _SingularResumeMetric()
    relations = (RankRelation("tp", Replica()),)
    items = [_item("a", torch.ones(1), relations), _item("b", torch.ones(1), relations)]

    results = TensorMetricExecutor({"tp": object()}).run(metric, items)

    torch.testing.assert_close(results[0].tensor, torch.tensor([3.0]))
    torch.testing.assert_close(results[1].tensor, torch.tensor([4.0]))
    assert metric.resume_calls == 2
    assert calls[0][0] == torch.Size([2, 1])


class _GatherMetric(TensorMetric):
    def __init__(self, collective):
        self.collective = collective

    def start(self, values):
        return [
            CollectiveStage((CollectiveRequest(value, "tp", self.collective),), "gathered")
            for value in values
        ]

    def resume(self, values, continuation):
        assert continuation == "gathered"
        assert len(values) == 1
        assert values[0].relation("tp").placement == Replica()
        return MetricResult(values[0].tensor)


def _fake_all_gather(monkeypatch, remote_value):
    monkeypatch.setattr(torch.distributed, "is_available", lambda: True)
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda group: 2)

    def all_gather(outputs, tensor, group):
        outputs[0].copy_(tensor)
        outputs[1].copy_(remote_value.to(tensor))

    monkeypatch.setattr(torch.distributed, "all_gather", all_gather)


def test_layer_sampled_median_gathers_the_sharded_population(monkeypatch):
    _fake_all_gather(monkeypatch, torch.tensor([[3.0, 4.0]]))
    item = _item(
        "decoder.layers.2.router_logits",
        torch.tensor([1.0, 2.0]),
        (RankRelation("dp", Shard(None)),),
    )

    results = TensorMetricExecutor({"dp": object()}).run(
        LayerSampledMedianMetric(sample_factor=1), [item]
    )

    torch.testing.assert_close(_result_tensor(results), torch.tensor(2.5))


def test_executor_all_gathers_an_explicit_shard(monkeypatch):
    _fake_all_gather(monkeypatch, torch.tensor([[3.0, 4.0]]))
    item = _item("vector", torch.tensor([1.0, 2.0]), (RankRelation("tp", Shard(0)),))

    result = _result_tensor(
        TensorMetricExecutor({"tp": object()}).run(_GatherMetric(AllGather(0)), [item])
    )

    torch.testing.assert_close(result, torch.tensor([1.0, 2.0, 3.0, 4.0]))


def test_executor_all_gathers_a_replica_along_a_new_leading_dimension(monkeypatch):
    _fake_all_gather(monkeypatch, torch.tensor([[1.0, 2.25]]))
    item = _item("replica", torch.tensor([1.0, 2.0]), (RankRelation("tp", Replica()),))

    results = TensorMetricExecutor({"tp": object()}).run(_GatherMetric(AllGather()), [item])

    torch.testing.assert_close(_result_tensor(results), torch.tensor([[1.0, 2.0], [1.0, 2.25]]))


def test_global_result_can_contain_the_same_input_more_than_once():
    value = _item("duplicate", torch.ones(1))

    results = TensorMetricExecutor({}).run(L2NormMetric(), [value, value])

    torch.testing.assert_close(_result_tensor(results), torch.sqrt(torch.tensor(2.0)))
    assert results[0].label == "global"


def test_all_reduce_requires_explicit_process_group_mapping(monkeypatch):
    _fake_distributed(monkeypatch, [])
    item = _item("shard", torch.ones(1), (RankRelation("tp", Shard(0)),))

    with pytest.raises(ValueError, match="No process group"):
        TensorMetricExecutor({}).run(L2NormMetric(), [item])


def test_executor_passes_through_torch_reduce_op(monkeypatch):
    calls = _fake_distributed(monkeypatch, [torch.tensor([[2.0]])])
    item = _item("replica", torch.ones(1), (RankRelation("tp", Replica()),))

    class ExplicitMetric(TensorMetric):
        def start(self, values):
            return [
                CollectiveStage(
                    (CollectiveRequest(value, "tp", AllReduce(torch.distributed.ReduceOp.MAX)),),
                    None,
                )
                for value in values
            ]

        def resume(self, values, continuation):
            return MetricResult(values[0].tensor)

    TensorMetricExecutor({"tp": object()}).run(ExplicitMetric(), [item])

    assert calls[0][1] is torch.distributed.ReduceOp.MAX


def _router_diagnostic_item(diagnostics: torch.Tensor, layer: int = 2) -> MetricTensor:
    return _item(
        f"decoder.layers.{layer}.router.router_diagnostics", diagnostics, kind="router_diagnostics"
    )


def _router_results(metric, diagnostics: torch.Tensor) -> dict[str, torch.Tensor]:
    results = TensorMetricExecutor({}).run(metric, [_router_diagnostic_item(diagnostics)])
    layer_prefix = "decoder.layers.2/"
    return {
        result.label.removeprefix(layer_prefix): result.tensor
        for result in results
        if result.label.startswith(layer_prefix)
    }


def _diagnostics(rows: int = 2, experts: int = 2) -> torch.Tensor:
    diagnostics = torch.zeros(rows, ROUTER_DIAGNOSTIC_CHANNEL_COUNT, experts)
    diagnostics[:, RouterDiagnosticChannel.VALID_TOKEN_COUNT, 0] = 1
    return diagnostics


def test_router_seq_aux_metric_decomposes_loss_growth():
    diagnostics = _diagnostics()
    diagnostics[:, RouterDiagnosticChannel.MEAN_SCORE] = torch.tensor([[0.75, 0.25], [0.5, 0.5]])
    diagnostics[:, RouterDiagnosticChannel.AUX_LOAD] = torch.tensor([[1.0, 0.0], [0.5, 0.5]])

    results = _router_results(LayerRouterSeqAuxDecompositionMetric(), diagnostics)

    expected = {
        "loss-mean": 1.25,
        "loss-max": 1.5,
        "assignment-imbalance-mean": 0.5,
        "score-imbalance-mean": 0.125,
        "imbalance-coupling-mean": 0.25,
        "imbalance-alignment-mean": 0.5,
    }
    assert set(results) == set(expected)
    for name, value in expected.items():
        torch.testing.assert_close(results[name], torch.tensor(value))


def test_router_routing_balance_metric_compares_biased_and_unbiased_load():
    diagnostics = _diagnostics()
    diagnostics[:, RouterDiagnosticChannel.AUX_LOAD] = torch.tensor([[1.0, 0.0], [0.5, 0.5]])
    diagnostics[:, RouterDiagnosticChannel.ACTUAL_LOAD] = torch.tensor([[0.5, 0.5], [0.0, 1.0]])
    diagnostics[:, RouterDiagnosticChannel.AUX_ACTUAL_OVERLAP, 0] = 0.5
    diagnostics[:, RouterDiagnosticChannel.VALID_TOKEN_COUNT, 0] = torch.tensor([1.0, 3.0])

    results = _router_results(LayerRouterRoutingBalanceMetric(), diagnostics)

    expected = {
        "actual-imbalance-mean": 0.5,
        "actual-imbalance-max": 1.0,
        "aux-actual-tv-mean": 0.5,
        "topk-overlap-mean": 0.5,
        "topk-overlap-min": 0.5,
        "aux-max-over-mean": 1.5,
        "actual-max-over-mean": 1.5,
        "actual-inactive-fraction": 0.25,
        "global-actual-imbalance": 0.5625,
        "global-aux-actual-tv": 0.5,
        "global-aux-max-over-mean": 1.25,
        "global-actual-max-over-mean": 1.75,
    }
    assert set(results) == set(expected)
    for name, value in expected.items():
        torch.testing.assert_close(results[name], torch.tensor(value))


def test_router_diagnostic_global_result_reduces_layer_metrics_without_cancellation():
    first = _diagnostics(rows=1)
    first[:, RouterDiagnosticChannel.MEAN_SCORE] = torch.tensor([[1.0, 0.0]])
    first[:, RouterDiagnosticChannel.AUX_LOAD] = torch.tensor([[1.0, 0.0]])
    first[:, RouterDiagnosticChannel.ACTUAL_LOAD] = torch.tensor([[1.0, 0.0]])
    second = _diagnostics(rows=1)
    second[:, RouterDiagnosticChannel.MEAN_SCORE] = torch.tensor([[0.0, 1.0]])
    second[:, RouterDiagnosticChannel.AUX_LOAD] = torch.tensor([[0.0, 1.0]])
    second[:, RouterDiagnosticChannel.ACTUAL_LOAD] = torch.tensor([[0.0, 1.0]])
    items = [_router_diagnostic_item(first, layer=2), _router_diagnostic_item(second, layer=3)]

    routing_results = TensorMetricExecutor({}).run(LayerRouterRoutingBalanceMetric(), items)
    seq_aux_results = TensorMetricExecutor({}).run(LayerRouterSeqAuxDecompositionMetric(), items)
    routing_by_label = {result.label: result.tensor for result in routing_results}
    seq_aux_by_label = {result.label: result.tensor for result in seq_aux_results}

    torch.testing.assert_close(
        routing_by_label["global/global-actual-max-over-mean"], torch.tensor(2.0)
    )
    torch.testing.assert_close(
        routing_by_label["global/global-aux-max-over-mean"], torch.tensor(2.0)
    )
    torch.testing.assert_close(routing_by_label["global/actual-max-over-mean"], torch.tensor(2.0))
    torch.testing.assert_close(seq_aux_by_label["global/loss-mean"], torch.tensor(2.0))


def test_router_health_metric_contrasts_sequence_and_global_batch_loss():
    diagnostics = _diagnostics()
    diagnostics[:, RouterDiagnosticChannel.MEAN_SCORE] = torch.tensor([[0.75, 0.25], [0.5, 0.5]])
    diagnostics[:, RouterDiagnosticChannel.AUX_LOAD] = torch.tensor([[1.0, 0.0], [0.5, 0.5]])
    diagnostics[:, RouterDiagnosticChannel.ACTUAL_LOAD] = torch.tensor([[0.5, 0.5], [0.0, 1.0]])
    diagnostics[:, RouterDiagnosticChannel.TOPK_BOUNDARY_RELATIVE_MARGIN, 0] = torch.tensor(
        [0.2, 0.4]
    )

    results = TensorMetricExecutor({}).run(
        LayerRouterHealthMetric(), [_router_diagnostic_item(diagnostics)]
    )
    results_by_label = {result.label: result.tensor for result in results}

    expected = {
        "seq-loss": 1.25,
        "seq-assignment-imbalance": 0.5,
        "seq-score-imbalance": 0.125,
        "global-batch-loss": 1.125,
        "seq-global-loss-gap": 0.125,
        "global-aux-max-over-mean": 1.5,
        "global-actual-imbalance": 0.25,
        "global-actual-max-over-mean": 1.5,
        "topk-boundary-relative-margin": 0.3,
    }
    assert set(results_by_label) == {
        f"{rollup}/{operation}" for rollup in ("global", "worst-layer") for operation in expected
    }
    for rollup in ("global", "worst-layer"):
        for operation, value in expected.items():
            torch.testing.assert_close(
                results_by_label[f"{rollup}/{operation}"], torch.tensor(value)
            )


def test_router_health_metric_reports_the_worst_layer_for_each_signal():
    first = _diagnostics(rows=1)
    first[:, RouterDiagnosticChannel.MEAN_SCORE] = torch.tensor([[0.5, 0.5]])
    first[:, RouterDiagnosticChannel.AUX_LOAD] = torch.tensor([[0.5, 0.5]])
    first[:, RouterDiagnosticChannel.ACTUAL_LOAD] = torch.tensor([[0.5, 0.5]])
    first[:, RouterDiagnosticChannel.TOPK_BOUNDARY_RELATIVE_MARGIN, 0] = 0.4
    second = _diagnostics(rows=1)
    second[:, RouterDiagnosticChannel.MEAN_SCORE] = torch.tensor([[1.0, 0.0]])
    second[:, RouterDiagnosticChannel.AUX_LOAD] = torch.tensor([[1.0, 0.0]])
    second[:, RouterDiagnosticChannel.ACTUAL_LOAD] = torch.tensor([[1.0, 0.0]])
    second[:, RouterDiagnosticChannel.TOPK_BOUNDARY_RELATIVE_MARGIN, 0] = 0.1
    items = [_router_diagnostic_item(first, layer=2), _router_diagnostic_item(second, layer=3)]

    results = TensorMetricExecutor({}).run(LayerRouterHealthMetric(), items)
    results_by_label = {result.label: result.tensor for result in results}

    torch.testing.assert_close(results_by_label["global/seq-loss"], torch.tensor(1.5))
    torch.testing.assert_close(results_by_label["worst-layer/seq-loss"], torch.tensor(2.0))
    torch.testing.assert_close(
        results_by_label["global/topk-boundary-relative-margin"], torch.tensor(0.25)
    )
    torch.testing.assert_close(
        results_by_label["worst-layer/topk-boundary-relative-margin"], torch.tensor(0.1)
    )


def test_router_expert_bias_metric_tracks_scale_and_load_correlations():
    diagnostics = _diagnostics()
    diagnostics[:, RouterDiagnosticChannel.EXPERT_BIAS] = torch.tensor([[1.0, -1.0], [0.5, -0.5]])
    diagnostics[:, RouterDiagnosticChannel.MEAN_SCORE] = torch.tensor([[0.75, 0.25], [0.25, 0.75]])
    diagnostics[:, RouterDiagnosticChannel.AUX_LOAD] = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    diagnostics[:, RouterDiagnosticChannel.ACTUAL_LOAD] = torch.tensor([[0.5, 0.5], [1.0, 0.0]])

    results = _router_results(LayerRouterExpertBiasMetric(), diagnostics)

    expected = {
        "mean": 0.0,
        "std": 0.75,
        "rms": 0.75,
        "range": 1.5,
        "abs-max": 0.75,
        "score-correlation": 0.0,
        "aux-load-correlation": 0.0,
        "actual-load-correlation": 0.5,
    }
    assert set(results) == set(expected)
    for name, value in expected.items():
        torch.testing.assert_close(results[name], torch.tensor(value))
