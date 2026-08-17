# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from collections.abc import Sequence

import pytest
import torch

from megatron.training.tensor_metrics import (
    FlatShard,
    MetricResult,
    MetricSite,
    MetricTensor,
    Owned,
    RankRelation,
    Replica,
    Shard,
    TensorMetricExecutor,
)
from megatron.training.tensor_metrics.examples import (
    FP8UnderflowFractionExample,
    GlobalParameterAmaxExample,
    LayerParameterL2NormExample,
    MaxReplicaDriftExample,
    MeanRowL2NormExample,
    MultiGranularityParameterL2NormExample,
    ParameterL2NormExample,
    SampledMaxReplicaDriftExample,
    TransformerEngineBatchedParameterL2NormExample,
)


def _value(
    name: str,
    kind: str,
    tensor: torch.Tensor,
    relations: Sequence[RankRelation] = (),
    *,
    is_placeholder: bool = False,
) -> MetricTensor:
    return MetricTensor(
        tensor, (MetricSite(name, kind),), tuple(relations), is_placeholder=is_placeholder
    )


def _only_result(results: Sequence[MetricResult]) -> MetricResult:
    assert len(results) == 1
    return results[0]


def _fake_distributed(monkeypatch, remote_values_by_op):
    remote_values = {op: iter(values) for op, values in remote_values_by_op.items()}
    calls = []
    monkeypatch.setattr(torch.distributed, "is_available", lambda: True)
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda group: 2)

    def all_reduce(tensor, op, group):
        calls.append((tensor.shape, op, group))
        remote = next(remote_values[op]).to(tensor)
        if op is torch.distributed.ReduceOp.MAX:
            torch.maximum(tensor, remote, out=tensor)
        elif op is torch.distributed.ReduceOp.MIN:
            torch.minimum(tensor, remote, out=tensor)
        else:
            tensor.add_(remote)

    monkeypatch.setattr(torch.distributed, "all_reduce", all_reduce)
    return calls


def test_parameter_l2_example_selects_parameters_and_reduces_globally():
    values = [
        _value("decoder.layers.0.weight", "parameter", torch.tensor([3.0, 4.0])),
        _value("decoder.layers.0", "activation", torch.tensor([100.0])),
    ]

    result = _only_result(TensorMetricExecutor({}).run(ParameterL2NormExample(), values))

    assert result.label == "global"
    torch.testing.assert_close(result.tensor, torch.tensor(5.0))


def test_global_parameter_amax_example_uses_zero_for_empty_parameters():
    values = [
        _value("decoder.layers.0.weight", "parameter", torch.tensor([-3.0, 4.0])),
        _value("decoder.layers.1.weight", "parameter", torch.empty(0)),
        _value("decoder.layers.0", "activation", torch.tensor([100.0])),
    ]

    result = _only_result(TensorMetricExecutor({}).run(GlobalParameterAmaxExample(), values))

    assert result.label == "global"
    torch.testing.assert_close(result.tensor, torch.tensor(4.0))


def test_te_batched_parameter_l2_example_uses_one_per_tensor_kernel_call(monkeypatch):
    te_optimizers = pytest.importorskip("transformer_engine.pytorch.optimizers")
    original_applier = te_optimizers.multi_tensor_applier
    calls = []

    def recorded_applier(operation, overflow_buffer, tensor_lists, *args):
        calls.append((len(tensor_lists[0]), args))
        return original_applier(operation, overflow_buffer, tensor_lists, *args)

    monkeypatch.setattr(te_optimizers, "multi_tensor_applier", recorded_applier)
    device = torch.device("cuda", torch.cuda.current_device())
    values = [
        _value("decoder.layers.0.weight", "parameter", torch.tensor([3.0, 4.0], device=device)),
        _value("decoder.layers.1.weight", "parameter", torch.tensor([0.0, 12.0], device=device)),
        _value("decoder.layers.2.weight", "parameter", torch.empty(0, device=device)),
    ]
    metric = TransformerEngineBatchedParameterL2NormExample()
    executor = TensorMetricExecutor({})

    prepared = executor.prepare(metric, values)

    assert calls == [(2, (True,))]
    torch.testing.assert_close(
        torch.stack(tuple(value.tensor for value in prepared)),
        torch.tensor([25.0, 144.0, 0.0], device=device),
    )
    result = _only_result(executor.complete(metric, executor.start(metric, prepared)))
    torch.testing.assert_close(result.tensor, torch.tensor(13.0, device=device))


def test_layer_parameter_l2_example_reduces_by_numbered_layer():
    values = [
        _value("decoder.layers.0.attention.weight", "parameter", torch.tensor([3.0, 4.0])),
        _value("decoder.layers.0.mlp.weight", "parameter", torch.tensor([12.0])),
        _value("decoder.layers.1.attention.weight", "parameter", torch.tensor([0.0, 5.0])),
        _value("embedding.weight", "parameter", torch.tensor([100.0])),
    ]

    results = TensorMetricExecutor({}).run(LayerParameterL2NormExample(), values)

    assert [result.label for result in results] == ["decoder.layers.0", "decoder.layers.1"]
    torch.testing.assert_close(results[0].tensor, torch.tensor(13.0))
    torch.testing.assert_close(results[1].tensor, torch.tensor(5.0))


def test_multi_granularity_example_reuses_one_prepared_state_in_several_results():
    value = _value("decoder.layers.0.mlp.linear_fc1.weight", "parameter", torch.tensor([3.0, 4.0]))
    metric = MultiGranularityParameterL2NormExample()
    executor = TensorMetricExecutor({})

    prepared = executor.prepare(metric, [value])
    results = executor.complete(metric, executor.start(metric, prepared))

    assert [result.label for result in results] == [
        "tensor/decoder.layers.0.mlp.linear_fc1.weight",
        "family/decoder.layers.*.mlp.linear_fc1.weight",
        "layer/decoder.layers.0",
        "global",
    ]
    assert len(results) == 4
    assert all(result.tensor == 5 for result in results)


def test_fp8_underflow_fraction_example_uses_explicit_tensor_unit_threshold():
    values = [
        _value("decoder.layers.0.weight", "wgrad", torch.tensor([0.0, 0.0005, 0.01, 1.0])),
        _value("decoder.layers.0.weight", "parameter", torch.tensor([0.0001])),
    ]

    result = _only_result(TensorMetricExecutor({}).run(FP8UnderflowFractionExample(0.001), values))

    assert result.tensor.dtype == torch.float64
    torch.testing.assert_close(result.tensor, torch.tensor(0.25, dtype=torch.float64))

    with pytest.raises(ValueError, match="must be positive"):
        FP8UnderflowFractionExample(0.0)


def test_mean_row_l2_example_reconstructs_flat_shard_before_sqrt(monkeypatch):
    calls = _fake_distributed(
        monkeypatch, {torch.distributed.ReduceOp.SUM: [torch.tensor([25.0, 0.0, 225.0])]}
    )
    value = _value(
        "decoder.layers.0.weight",
        "wgrad",
        torch.tensor([0.0, 0.0, 0.0, 0.0, 5.0, 12.0, 8.0]),
        (RankRelation("dp", FlatShard((3, 4), 2, 9)),),
    )

    result = _only_result(
        TensorMetricExecutor({"dp": object()}).run(MeanRowL2NormExample(), [value])
    )

    torch.testing.assert_close(result.tensor, torch.tensor(35.0 / 3.0))
    assert [call[0] for call in calls] == [torch.Size([1, 3])]


def test_mean_row_l2_example_uses_neutral_population_for_remotely_owned_wgrad(monkeypatch):
    calls = _fake_distributed(
        monkeypatch, {torch.distributed.ReduceOp.SUM: [torch.tensor([5.0, 1.0])]}
    )
    value = _value(
        "decoder.layers.0.weight",
        "wgrad",
        torch.empty(0),
        (RankRelation("dp", Owned(0)),),
        is_placeholder=True,
    )

    result = _only_result(
        TensorMetricExecutor({"dp": object()}).run(MeanRowL2NormExample(), [value])
    )

    torch.testing.assert_close(result.tensor, torch.tensor(5.0))
    assert [call[0] for call in calls] == [torch.Size([1, 2])]


def test_max_replica_drift_example_uses_extrema_then_reduces_shards(monkeypatch):
    calls = _fake_distributed(
        monkeypatch,
        {
            torch.distributed.ReduceOp.MAX: [torch.tensor([3.0, 2.0]), torch.tensor([4.0, 0.5])],
            torch.distributed.ReduceOp.MIN: [torch.tensor([3.0, 2.0])],
        },
    )
    value = _value(
        "decoder.layers.0.weight",
        "parameter",
        torch.tensor([1.0, 5.0]),
        (RankRelation("dp", Replica()), RankRelation("tp", Shard(0))),
    )

    result = _only_result(
        TensorMetricExecutor({"dp": object(), "tp": object()}).run(
            MaxReplicaDriftExample("dp"), [value]
        )
    )

    torch.testing.assert_close(result.tensor, torch.tensor([4.0, 2.0 / 3.0]))
    assert MaxReplicaDriftExample.result_components == ("max_absolute_drift", "max_relative_drift")
    assert [call[0] for call in calls if call[1] is torch.distributed.ReduceOp.MAX] == [
        torch.Size([1, 2]),
        torch.Size([1, 2]),
    ]


def test_max_replica_drift_uses_symmetric_relative_scale_with_absolute_floor(monkeypatch):
    value = _value(
        "decoder.layers.0.weight",
        "parameter",
        torch.tensor([100.0, 1e-10]),
        (RankRelation("dp", Replica()),),
    )
    calls = _fake_distributed(
        monkeypatch,
        {
            torch.distributed.ReduceOp.MAX: [torch.tensor([101.0, -1e-10])],
            torch.distributed.ReduceOp.MIN: [torch.tensor([101.0, -1e-10])],
        },
    )

    result = _only_result(
        TensorMetricExecutor({"dp": object()}).run(
            MaxReplicaDriftExample("dp", relative_scale_floor=1e-6), [value]
        )
    )

    torch.testing.assert_close(result.tensor, torch.tensor([1.0, 1.0 / 101.0]))
    assert len(calls) == 2


def test_max_replica_drift_reports_large_relative_sign_reversal(monkeypatch):
    value = _value(
        "decoder.layers.0.weight",
        "parameter",
        torch.tensor([-1.0]),
        (RankRelation("dp", Replica()),),
    )
    _fake_distributed(
        monkeypatch,
        {
            torch.distributed.ReduceOp.MAX: [torch.tensor([1.0])],
            torch.distributed.ReduceOp.MIN: [torch.tensor([1.0])],
        },
    )

    result = _only_result(
        TensorMetricExecutor({"dp": object()}).run(MaxReplicaDriftExample("dp"), [value])
    )

    torch.testing.assert_close(result.tensor, torch.tensor([2.0, 2.0]))


def test_max_replica_drift_example_requires_replica_placement():
    value = _value(
        "decoder.layers.0.weight", "parameter", torch.tensor([1.0]), (RankRelation("dp", Shard(0)),)
    )

    with pytest.raises(ValueError, match="must have Replica placement"):
        TensorMetricExecutor({"dp": object()}).run(MaxReplicaDriftExample("dp"), [value])

    with pytest.raises(ValueError, match="scale floor must be positive and finite"):
        MaxReplicaDriftExample("dp", relative_scale_floor=0.0)


def test_sampled_replica_drift_defaults_to_stable_one_in_one_hundred_sample():
    value = _value(
        "decoder.layers.0.weight",
        "parameter",
        torch.arange(1000, dtype=torch.float32),
        (RankRelation("dp", Replica()),),
    )
    executor = TensorMetricExecutor({"dp": object()})

    first = executor.prepare(SampledMaxReplicaDriftExample("dp"), [value])[0]
    repeated = executor.prepare(SampledMaxReplicaDriftExample("dp"), [value])[0]
    different_seed = executor.prepare(SampledMaxReplicaDriftExample("dp", sample_seed=1), [value])[
        0
    ]

    assert first.tensor.numel() == 10
    assert first.tensor.unique().numel() == 10
    torch.testing.assert_close(first.tensor, repeated.tensor)
    assert not torch.equal(first.tensor, different_seed.tensor)
    assert first.sites == value.sites
    assert first.rank_relations == value.rank_relations

    with pytest.raises(ValueError, match="must be positive"):
        SampledMaxReplicaDriftExample("dp", sample_factor=0)


def test_sampled_replica_drift_reuses_exact_extrema_algorithm(monkeypatch):
    value = _value(
        "decoder.layers.0.weight",
        "parameter",
        torch.arange(200, dtype=torch.float32),
        (RankRelation("dp", Replica()),),
    )
    metric = SampledMaxReplicaDriftExample("dp")
    executor = TensorMetricExecutor({"dp": object()})
    prepared = executor.prepare(metric, [value])
    remote = prepared[0].tensor.clone()
    remote[0].add_(7.0)
    calls = _fake_distributed(
        monkeypatch,
        {torch.distributed.ReduceOp.MAX: [remote], torch.distributed.ReduceOp.MIN: [remote]},
    )

    result = _only_result(executor.complete(metric, executor.start(metric, prepared)))

    original = prepared[0].tensor[0]
    expected_relative = 7.0 / max(abs(float(original)), abs(float(original + 7.0)), 1e-8)
    torch.testing.assert_close(result.tensor, torch.tensor([7.0, expected_relative]))
    assert [call[0] for call in calls] == [torch.Size([1, 2]), torch.Size([1, 2])]
