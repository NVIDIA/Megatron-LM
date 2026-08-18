# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import os
from argparse import ArgumentParser
from contextlib import contextmanager, nullcontext
from types import SimpleNamespace
from unittest import mock

import pytest
import torch

import megatron.training.tensor_metrics.observer as observer_mod
import megatron.training.tensor_metrics.optimizer_sources as optimizer_sources_mod
from megatron.core.optimizer import FP32Optimizer
from megatron.core.parameter_names import CanonicalParameterNameMap
from megatron.core.tensor_observation import observe_layer_residuals, observe_tensor
from megatron.core.transformer.enums import CudaGraphModule
from megatron.core.transformer.moe.router_diagnostics import (
    ROUTER_DIAGNOSTIC_CHANNEL_COUNT,
    RouterDiagnosticChannel,
)
from megatron.training.arguments import _add_logging_args
from megatron.training.tensor_metrics import Replica, Shard
from megatron.training.tensor_metrics.definitions import LayerL2NormMetric
from megatron.training.tensor_metrics.observer import (
    ScheduledMetric,
    TrainingTensorMetricObserver,
    build_tensor_metric_observer,
    parse_tensor_metric_specs,
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


def _pg_collection(**sizes):
    return SimpleNamespace(
        tp=_ProcessGroup(sizes.get("tp", 1)),
        cp=_ProcessGroup(sizes.get("cp", 1)),
        expt_tp=_ProcessGroup(sizes.get("expert_tp", 1)),
        pp=_ProcessGroup(sizes.get("pp", 1)),
        ep=_ProcessGroup(sizes.get("ep", 1)),
        dp_cp=_ProcessGroup(sizes.get("dp", 1)),
        expt_dp=_ProcessGroup(sizes.get("expert_dp", 1)),
        gtp_remat=_ProcessGroup(sizes.get("gtp_remat", 1)),
        expt_gtp_remat=_ProcessGroup(sizes.get("expt_gtp_remat", 1)),
    )


def _layer_model():
    model = torch.nn.Module()
    model.decoder = torch.nn.Module()
    model.decoder.layers = torch.nn.ModuleList([torch.nn.Linear(2, 2)])
    with torch.no_grad():
        model.decoder.layers[0].weight.copy_(torch.tensor([[3.0, 4.0], [0.0, 0.0]]))
        model.decoder.layers[0].bias.zero_()
    return model


def _fp32_optimizer(model):
    optimizer = object.__new__(FP32Optimizer)
    optimizer.is_stub_optimizer = False
    optimizer.optimizer = SimpleNamespace(param_groups=[{"params": list(model.parameters())}])
    return optimizer


def _forward_model(cuda_graph_impl="none", cuda_graph_modules=()):
    model = _layer_model()
    model.config = SimpleNamespace(
        cuda_graph_impl=cuda_graph_impl, cuda_graph_modules=list(cuda_graph_modules)
    )
    model.decoder.layers[0].config = SimpleNamespace(sequence_parallel=False)
    model.decoder.layers[0].router = torch.nn.Module()
    model.output_layer = torch.nn.Module()
    return model


def test_logger_config_parses_per_metric_intervals():
    parser = ArgumentParser()
    _add_logging_args(parser)

    args = parser.parse_args(
        ["--tensor-metrics", "layer-param-l2:10", "future-activation-metric:100"]
    )

    assert args.tensor_metrics == ["layer-param-l2:10", "future-activation-metric:100"]


def test_tensor_metric_specs_build_independent_schedules():
    scheduled_metrics = parse_tensor_metric_specs(
        [
            "global-param-l2:5",
            "layer-param-l2:10",
            "global-param-mean-row-l2:20",
            "global-param-mean-column-l2:25",
            "global-wgrad-l2:30",
            "layer-wgrad-l2:40",
        ]
    )

    assert [scheduled.metric.name for scheduled in scheduled_metrics] == [
        "global-param-l2",
        "layer-param-l2",
        "global-param-mean-row-l2",
        "global-param-mean-column-l2",
        "global-wgrad-l2",
        "layer-wgrad-l2",
    ]
    assert [scheduled.interval for scheduled in scheduled_metrics] == [5, 10, 20, 25, 30, 40]
    assert build_tensor_metric_observer([]) is None


def test_tensor_metric_specs_include_forward_sources():
    scheduled_metrics = parse_tensor_metric_specs(
        [
            "layer-residual-accumulator-l2:1",
            "layer-residual-contribution-l2:2",
            "global-output-logits-l2:3",
            "global-mtp-logits-l2:4",
            "layer-router-logits-l2:5",
            "layer-router-logits-max:6",
            "layer-router-logits-sampled-median:7",
            "layer-router-decision-entropy:8",
            "layer-router-seq-aux-decomposition:9",
            "layer-router-routing-balance:10",
            "layer-router-expert-bias:11",
            "layer-router-health:12",
        ]
    )

    assert [scheduled.metric.source_kinds for scheduled in scheduled_metrics] == [
        frozenset({"residual_accumulator"}),
        frozenset({"residual_contribution"}),
        frozenset({"output_logits"}),
        frozenset({"mtp_logits"}),
        frozenset({"router_logits"}),
        frozenset({"router_logits"}),
        frozenset({"router_logits"}),
        frozenset({"router_scores"}),
        frozenset({"router_diagnostics"}),
        frozenset({"router_diagnostics"}),
        frozenset({"router_diagnostics"}),
        frozenset({"router_diagnostics"}),
    ]


def test_scheduled_metric_keeps_interval_outside_metric_definition():
    scheduled = ScheduledMetric(LayerL2NormMetric(), interval=3)

    assert not scheduled.is_due(0)
    assert scheduled.is_due(2)
    with pytest.raises(ValueError, match="must be positive"):
        ScheduledMetric(LayerL2NormMetric(), interval=0)


def test_training_observer_requires_explicit_metric_sources():
    class UndeclaredSourceMetric(LayerL2NormMetric):
        source_kinds = frozenset()

    with pytest.raises(ValueError, match="must explicitly declare source_kinds"):
        TrainingTensorMetricObserver([ScheduledMetric(UndeclaredSourceMetric(), interval=1)])


def test_training_observer_validates_metric_sources_at_construction():
    class UnsupportedSourceMetric(LayerL2NormMetric):
        source_kinds = frozenset({"future-source"})

    with pytest.raises(NotImplementedError, match="future-source"):
        TrainingTensorMetricObserver([ScheduledMetric(UnsupportedSourceMetric(), interval=1)])


@pytest.mark.parametrize(
    ("specifications", "message"),
    (
        (["layer-param-l2"], "expected NAME:INTERVAL"),
        (["missing:10"], "Unknown tensor metric"),
        (["layer-param-l2:zero"], "integer interval"),
        (["layer-param-l2:0"], "positive interval"),
        (["layer-param-l2:1", "layer-param-l2:2"], "configured more than once"),
    ),
)
def test_tensor_metric_specs_are_validated(specifications, message):
    with pytest.raises(ValueError, match=message):
        parse_tensor_metric_specs(specifications)


def test_parameter_observation_names_deduplicates_and_describes_tp_placement():
    model = _layer_model()
    model.decoder.layers[0].weight.tensor_model_parallel = True
    model.decoder.layers[0].weight.partition_dim = 0

    values = optimizer_sources_mod._optimizer_metric_tensors(
        CanonicalParameterNameMap([model, model]),
        _fp32_optimizer(model),
        _pg_collection(),
        [LayerL2NormMetric()],
    )

    assert [value.sites[0].name for value in values] == [
        "decoder.layers.0.bias",
        "decoder.layers.0.weight",
    ]
    values_by_name = {value.sites[0].name: value for value in values}
    assert values_by_name["decoder.layers.0.weight"].relation("tp").placement == Shard(0)
    assert values_by_name["decoder.layers.0.bias"].relation("tp").placement == Replica()


def test_optimizer_observer_filters_sites_before_building_metric_tensors():
    class WeightOnlyMetric(LayerL2NormMetric):
        def accepts(self, site):
            return super().accepts(site) and site.name.endswith(".weight")

    model = _layer_model()

    values = optimizer_sources_mod._optimizer_metric_tensors(
        CanonicalParameterNameMap(model),
        _fp32_optimizer(model),
        _pg_collection(),
        [WeightOnlyMetric()],
    )

    assert [value.sites[0].name for value in values] == ["decoder.layers.0.weight"]


def test_observer_runs_only_due_metrics_and_combines_layer_parameters():
    captured = []
    observer = build_tensor_metric_observer(
        ["layer-param-l2:2"],
        result_sink=lambda metric, results, iteration: captured.append(
            (metric, tuple(results), iteration)
        ),
    )
    assert observer is not None
    model = [_layer_model()]
    kwargs = {
        "model": model,
        "optimizer": _fp32_optimizer(model[0]),
        "pg_collection": _pg_collection(),
    }

    observer(iteration=0, **kwargs)
    assert captured == []

    observer(iteration=1, **kwargs)

    assert len(captured) == 1
    metric, results, iteration = captured[0]
    assert metric.name == "layer-param-l2"
    assert iteration == 2
    assert len(results) == 1
    assert results[0].label == "decoder.layers.0"
    torch.testing.assert_close(results[0].tensor, torch.tensor(5.0))


def test_observer_profiles_only_due_metric_work(monkeypatch):
    events = []

    @contextmanager
    def metric_range(name, *, time):
        events.append(("enter", name, time))
        yield
        events.append(("exit", name, time))

    range_factory = mock.Mock(return_value=metric_range)
    monkeypatch.setattr(observer_mod, "get_nvtx_range", range_factory)
    observer = build_tensor_metric_observer(
        ["layer-param-l2:2"], result_sink=lambda *args: events.append(("sink",))
    )
    assert observer is not None
    model = _layer_model()
    kwargs = {
        "model": [model],
        "optimizer": _fp32_optimizer(model),
        "pg_collection": _pg_collection(),
    }

    observer(iteration=0, **kwargs)
    assert events == []
    range_factory.assert_not_called()

    observer(iteration=1, **kwargs)

    range_factory.assert_called_once_with()
    assert events == [
        ("enter", "tensor-metrics", True),
        ("sink",),
        ("exit", "tensor-metrics", True),
    ]


def test_observer_caches_canonical_parameter_names():
    observer = build_tensor_metric_observer(["layer-param-l2:1"], result_sink=lambda *args: None)
    assert observer is not None
    model = [_layer_model()]
    kwargs = {
        "model": model,
        "optimizer": _fp32_optimizer(model[0]),
        "pg_collection": _pg_collection(),
    }

    observer(iteration=0, **kwargs)
    parameter_names = observer._parameter_names
    manifest = observer._optimizer_parameter_manifest
    observer(iteration=1, **kwargs)

    assert parameter_names is not None
    assert manifest is not None
    assert observer._parameter_names is parameter_names
    assert observer._optimizer_parameter_manifest is manifest


def test_observer_rejects_unsupported_parallel_topology():
    observer = build_tensor_metric_observer(["layer-param-l2:1"], result_sink=lambda *args: None)
    assert observer is not None
    model = _layer_model()

    with pytest.raises(NotImplementedError, match="pipeline parallelism"):
        observer(
            model=[model],
            optimizer=_fp32_optimizer(model),
            iteration=0,
            pg_collection=_pg_collection(pp=2),
        )


def test_parameter_only_observation_scope_allows_gtp_topology():
    observer = build_tensor_metric_observer(
        ["global-param-l2:1"], result_sink=lambda *args: None
    )
    assert observer is not None

    with observer.observe_forward_backward(
        model=[_layer_model()], iteration=0, pg_collection=_pg_collection(gtp_remat=2)
    ):
        pass


def test_forward_metrics_treat_gtp_as_an_activation_population_shard():
    observer = build_tensor_metric_observer(
        ["global-output-logits-l2:1"], result_sink=lambda *args: None
    )
    assert observer is not None
    model = _forward_model()

    with observer.observe_forward_backward(
        model=[model], iteration=0, pg_collection=_pg_collection(gtp_remat=2)
    ):
        observe_tensor(
            model.output_layer, "output_logits", "output_logits", torch.tensor([3.0, 4.0])
        )

    assert observer._prepared_forward_values is not None
    (prepared_value,) = next(iter(observer._prepared_forward_values.values()))
    assert prepared_value.relation("gtp").placement == Shard(None)
    assert prepared_value.relation("dp").placement == Replica()


def test_forward_metrics_reduce_gtp_and_dp_activation_populations_end_to_end():
    launched_world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if launched_world_size < 4 or launched_world_size % 2:
        pytest.skip("This test requires an even number of at least four distributed ranks.")
    if not torch.distributed.is_initialized():
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
        torch.distributed.init_process_group(backend="nccl")

    rank = torch.distributed.get_rank()
    world_size = torch.distributed.get_world_size()
    gtp_size = 2
    dp_size = world_size // gtp_size
    gtp_group = None
    for dp_rank in range(dp_size):
        ranks = list(range(dp_rank * gtp_size, (dp_rank + 1) * gtp_size))
        group = torch.distributed.new_group(ranks)
        if rank in ranks:
            gtp_group = group
    dp_group = None
    for gtp_rank in range(gtp_size):
        ranks = list(range(gtp_rank, world_size, gtp_size))
        group = torch.distributed.new_group(ranks)
        if rank in ranks:
            dp_group = group
    assert gtp_group is not None
    assert dp_group is not None

    captured = []
    observer = build_tensor_metric_observer(
        ["global-output-logits-l2:1"],
        result_sink=lambda metric, results, iteration: captured.extend(results),
    )
    assert observer is not None
    model = _forward_model()
    pg_collection = _pg_collection()
    pg_collection.gtp_remat = gtp_group
    pg_collection.dp_cp = dp_group
    device = torch.device("cuda", torch.cuda.current_device())

    with observer.observe_forward_backward(
        model=[model], iteration=0, pg_collection=pg_collection
    ):
        observe_tensor(
            model.output_layer,
            "output_logits",
            "output_logits",
            torch.tensor([rank + 1.0], device=device),
        )
    observer(
        model=[model],
        optimizer=_fp32_optimizer(model),
        iteration=0,
        pg_collection=pg_collection,
    )

    assert len(captured) == 1
    expected = (
        torch.arange(1, world_size + 1, dtype=torch.float32, device=device)
        .square()
        .sum()
        .sqrt()
    )
    torch.testing.assert_close(captured[0].tensor, expected)


def test_gtp_topology_does_not_filter_optimizer_metrics_by_type():
    observer = build_tensor_metric_observer(
        ["global-param-mean-row-l2:1"], result_sink=lambda *args: None
    )
    assert observer is not None
    model = _layer_model()

    observer(
        model=[model],
        optimizer=_fp32_optimizer(model),
        iteration=0,
        pg_collection=_pg_collection(gtp_remat=2),
    )


def test_default_sink_writes_tensorboard_and_wandb():
    tensorboard_writer = mock.Mock()
    wandb_writer = mock.Mock()
    observer = TrainingTensorMetricObserver(parse_tensor_metric_specs(["layer-param-l2:1"]))

    with (
        mock.patch.object(observer_mod, "get_tensorboard_writer", return_value=tensorboard_writer),
        mock.patch.object(observer_mod, "get_wandb_writer", return_value=wandb_writer),
    ):
        model = _layer_model()
        observer(
            model=[model],
            optimizer=_fp32_optimizer(model),
            iteration=0,
            pg_collection=_pg_collection(),
        )

    tag = "tensor-metrics/layer-param-l2/decoder.layers.0"
    tensorboard_writer.add_scalar.assert_called_once()
    assert tensorboard_writer.add_scalar.call_args.args[0] == tag
    torch.testing.assert_close(tensorboard_writer.add_scalar.call_args.args[1], torch.tensor(5.0))
    assert tensorboard_writer.add_scalar.call_args.args[2] == 1
    wandb_values, wandb_iteration = wandb_writer.log.call_args.args
    assert tuple(wandb_values) == (tag,)
    torch.testing.assert_close(wandb_values[tag], torch.tensor(5.0))
    assert wandb_iteration == 1


def test_observer_runs_global_and_layer_parameter_and_wgrad_metrics():
    captured = {}
    model = _layer_model()
    model.decoder.layers[0].weight.main_grad = torch.tensor([[0.0, 0.0], [5.0, 12.0]])
    model.decoder.layers[0].bias.main_grad = torch.zeros(2)
    observer = build_tensor_metric_observer(
        ["global-param-l2:1", "layer-param-l2:1", "global-wgrad-l2:1", "layer-wgrad-l2:1"],
        result_sink=lambda metric, results, iteration: captured.update(
            {metric.name: (tuple(results), iteration)}
        ),
    )
    assert observer is not None

    observer(
        model=[model], optimizer=_fp32_optimizer(model), iteration=0, pg_collection=_pg_collection()
    )

    assert set(captured) == {
        "global-param-l2",
        "layer-param-l2",
        "global-wgrad-l2",
        "layer-wgrad-l2",
    }
    for metric_name, expected in (
        ("global-param-l2", 5.0),
        ("layer-param-l2", 5.0),
        ("global-wgrad-l2", 13.0),
        ("layer-wgrad-l2", 13.0),
    ):
        results, iteration = captured[metric_name]
        assert iteration == 1
        expected_label = "global" if metric_name.startswith("global") else "decoder.layers.0"
        expected_labels = {expected_label}
        if metric_name == "layer-wgrad-l2":
            expected_labels.add("global")
        results_by_label = {result.label: result.tensor for result in results}
        assert set(results_by_label) == expected_labels
        for result in results_by_label.values():
            torch.testing.assert_close(result, torch.tensor(expected))


def test_observer_prepares_and_accumulates_forward_sources_until_commit():
    captured = {}
    model = _forward_model()
    layer = model.decoder.layers[0]
    observer = build_tensor_metric_observer(
        [
            "layer-residual-accumulator-l2:1",
            "layer-residual-contribution-l2:1",
            "global-output-logits-l2:1",
            "global-mtp-logits-l2:1",
            "layer-router-logits-l2:1",
            "layer-router-logits-max:1",
            "layer-router-logits-sampled-median:1",
            "layer-router-decision-entropy:1",
        ],
        result_sink=lambda metric, results, iteration: captured.update(
            {metric.name: (tuple(results), iteration)}
        ),
    )
    assert observer is not None
    pg_collection = _pg_collection()

    with observer.observe_forward_backward(model=[model], iteration=0, pg_collection=pg_collection):
        accumulator = torch.tensor([3.0, 4.0], requires_grad=True)
        observe_layer_residuals(layer, accumulator, accumulator + torch.tensor([3.0, 4.0]))
        observe_tensor(
            model.output_layer, "output_logits", "output_logits", torch.tensor([5.0, 12.0])
        )
        observe_tensor(model.output_layer, "mtp_logits.0", "mtp_logits", torch.tensor([3.0, 4.0]))
        observe_tensor(model.output_layer, "mtp_logits.1", "mtp_logits", torch.tensor([0.0, 12.0]))
        observe_tensor(layer.router, "router_logits", "router_logits", torch.tensor([8.0, 8.0]))
        observe_tensor(
            layer.router, "router_scores", "router_scores", torch.tensor([[0.5, 0.5], [1.0, 0.0]])
        )

        assert observer._prepared_forward_values is not None
        assert all(
            value.tensor.numel() <= 2
            for values in observer._prepared_forward_values.values()
            for value in values
        )

    observer(
        model=[model], optimizer=_fp32_optimizer(model), iteration=0, pg_collection=pg_collection
    )

    expected = {
        "layer-residual-accumulator-l2": ("decoder.layers.0", 5.0),
        "layer-residual-contribution-l2": ("decoder.layers.0", 5.0),
        "global-output-logits-l2": ("global", 13.0),
        "global-mtp-logits-l2": ("global", 13.0),
        "layer-router-logits-l2": ("decoder.layers.0", 128.0**0.5),
        "layer-router-logits-max": ("decoder.layers.0", 8.0),
        "layer-router-logits-sampled-median": ("decoder.layers.0", 8.0),
        "layer-router-decision-entropy": ("decoder.layers.0", 0.5),
    }
    assert set(captured) == set(expected)
    for metric_name, (label, value) in expected.items():
        results, iteration = captured[metric_name]
        assert iteration == 1
        expected_labels = {label}
        if metric_name.startswith("layer-"):
            expected_labels.add("global")
        results_by_label = {result.label: result.tensor for result in results}
        assert set(results_by_label) == expected_labels
        for result in results_by_label.values():
            torch.testing.assert_close(result, torch.tensor(value))
    assert observer._prepared_forward_values is None


def test_forward_observer_delegates_site_filtering_to_executor():
    class ResidualMetric(LayerL2NormMetric):
        source_kinds = frozenset({"residual_accumulator"})

    metric = ResidualMetric()
    metric.accepts = mock.Mock(wraps=metric.accepts)
    observer = TrainingTensorMetricObserver(
        [ScheduledMetric(metric, interval=1)], result_sink=lambda *args: None
    )
    model = _forward_model()
    layer = model.decoder.layers[0]
    pg_collection = _pg_collection()

    with observer.observe_forward_backward(model=[model], iteration=0, pg_collection=pg_collection):
        observe_tensor(
            layer,
            "residual_accumulator",
            "residual_accumulator",
            torch.tensor([3.0, 4.0]),
        )

    assert metric.accepts.call_count == 1


def test_repeated_forward_attempt_replaces_prepared_values_before_commit():
    captured = []
    observer = build_tensor_metric_observer(
        ["global-output-logits-l2:1"],
        result_sink=lambda metric, results, iteration: captured.extend(results),
    )
    assert observer is not None
    model = _forward_model()
    pg_collection = _pg_collection()

    with observer.observe_forward_backward(model=[model], iteration=0, pg_collection=pg_collection):
        observe_tensor(
            model.output_layer, "output_logits", "output_logits", torch.tensor([3.0, 4.0])
        )
    with observer.observe_forward_backward(model=[model], iteration=0, pg_collection=pg_collection):
        observe_tensor(
            model.output_layer, "output_logits", "output_logits", torch.tensor([5.0, 12.0])
        )

    observer(
        model=[model],
        optimizer=_fp32_optimizer(model),
        iteration=0,
        pg_collection=pg_collection,
    )

    assert len(captured) == 1
    torch.testing.assert_close(captured[0].tensor, torch.tensor(13.0))


def test_observer_runs_router_diagnostic_metric_families():
    captured = {}
    model = _forward_model()
    observer = build_tensor_metric_observer(
        [
            "layer-router-seq-aux-decomposition:1",
            "layer-router-routing-balance:1",
            "layer-router-expert-bias:1",
            "layer-router-health:1",
        ],
        result_sink=lambda metric, results, iteration: captured.update(
            {metric.name: (tuple(results), iteration)}
        ),
    )
    assert observer is not None
    diagnostics = torch.zeros(1, ROUTER_DIAGNOSTIC_CHANNEL_COUNT, 2)
    diagnostics[:, RouterDiagnosticChannel.MEAN_SCORE] = torch.tensor([[0.75, 0.25]])
    diagnostics[:, RouterDiagnosticChannel.AUX_LOAD] = torch.tensor([[1.0, 0.0]])
    diagnostics[:, RouterDiagnosticChannel.ACTUAL_LOAD] = torch.tensor([[0.5, 0.5]])
    diagnostics[:, RouterDiagnosticChannel.EXPERT_BIAS] = torch.tensor([[0.1, -0.1]])
    diagnostics[:, RouterDiagnosticChannel.AUX_ACTUAL_OVERLAP, 0] = 0.5
    diagnostics[:, RouterDiagnosticChannel.VALID_TOKEN_COUNT, 0] = 8
    diagnostics[:, RouterDiagnosticChannel.TOPK_BOUNDARY_RELATIVE_MARGIN, 0] = 0.25
    pg_collection = _pg_collection()

    with observer.observe_forward_backward(model=[model], iteration=0, pg_collection=pg_collection):
        observe_tensor(
            model.decoder.layers[0].router, "router_diagnostics", "router_diagnostics", diagnostics
        )
        assert observer._prepared_forward_values is not None
        assert all(
            value.tensor.data_ptr() != diagnostics.data_ptr()
            for values in observer._prepared_forward_values.values()
            for value in values
        )

    observer(
        model=[model], optimizer=_fp32_optimizer(model), iteration=0, pg_collection=pg_collection
    )

    assert set(captured) == {
        "layer-router-seq-aux-decomposition",
        "layer-router-routing-balance",
        "layer-router-expert-bias",
        "layer-router-health",
    }
    results_by_metric = {
        metric_name: {result.label: result.tensor for result in results}
        for metric_name, (results, iteration) in captured.items()
        if iteration == 1
    }
    torch.testing.assert_close(
        results_by_metric["layer-router-seq-aux-decomposition"]["decoder.layers.0/loss-mean"],
        torch.tensor(1.5),
    )
    torch.testing.assert_close(
        results_by_metric["layer-router-routing-balance"]["decoder.layers.0/aux-actual-tv-mean"],
        torch.tensor(0.5),
    )
    torch.testing.assert_close(
        results_by_metric["layer-router-expert-bias"]["decoder.layers.0/range"], torch.tensor(0.2)
    )
    torch.testing.assert_close(
        results_by_metric["layer-router-health"]["global/global-batch-loss"], torch.tensor(1.5)
    )
    torch.testing.assert_close(
        results_by_metric["layer-router-health"]["worst-layer/topk-boundary-relative-margin"],
        torch.tensor(0.25),
    )


def test_forward_metric_requires_observation_scope_before_commit():
    observer = build_tensor_metric_observer(
        ["global-output-logits-l2:1"], result_sink=lambda *args: None
    )
    assert observer is not None
    model = _forward_model()

    with pytest.raises(RuntimeError, match="observe_forward_backward"):
        observer(
            model=[model],
            optimizer=_fp32_optimizer(model),
            iteration=0,
            pg_collection=_pg_collection(),
        )


def test_forward_observation_is_not_timed_by_a_synchronizing_timer(monkeypatch):
    """Megatron timers synchronize the device when they start and stop.

    Timing each observation would serialize the forward-backward pass once per observed tensor,
    so only the pre-step commit may request timing.
    """
    ranges = []

    @contextmanager
    def recording_range(message, time=False, log_level=1):
        ranges.append((message, time))
        yield

    monkeypatch.setattr(observer_mod, "get_nvtx_range", lambda: recording_range)
    model = _forward_model()
    observer = build_tensor_metric_observer(
        ["global-output-logits-l2:1"], result_sink=lambda *args: None
    )
    assert observer is not None
    pg_collection = _pg_collection()

    with observer.observe_forward_backward(model=[model], iteration=0, pg_collection=pg_collection):
        observe_tensor(
            model.output_layer, "output_logits", "output_logits", torch.tensor([5.0, 12.0])
        )
    observer(
        model=[model], optimizer=_fp32_optimizer(model), iteration=0, pg_collection=pg_collection
    )

    forward_range, commit_range = ranges
    assert forward_range == ("tensor-metrics", False)
    assert commit_range == ("tensor-metrics", True)


def test_forward_metrics_allow_partial_cuda_graphs_outside_observation_sites():
    observer = build_tensor_metric_observer(
        ["layer-residual-contribution-l2:1", "global-output-logits-l2:1"],
        result_sink=lambda *args: None,
    )
    assert observer is not None
    model = _forward_model(
        cuda_graph_impl="local", cuda_graph_modules=(CudaGraphModule.attn, CudaGraphModule.mamba)
    )

    with observer.observe_forward_backward(
        model=[model], iteration=0, pg_collection=_pg_collection()
    ):
        pass


@pytest.mark.parametrize(
    "specification",
    (
        "layer-router-logits-l2:1",
        "layer-router-decision-entropy:1",
        "layer-router-seq-aux-decomposition:1",
        "layer-router-health:1",
    ),
)
def test_router_metrics_reject_cuda_graphs_that_capture_router(specification):
    observer = build_tensor_metric_observer([specification], result_sink=lambda *args: None)
    assert observer is not None
    model = _forward_model(
        cuda_graph_impl="local", cuda_graph_modules=(CudaGraphModule.moe_router,)
    )

    with pytest.raises(NotImplementedError, match="eager MoE router"):
        with observer.observe_forward_backward(
            model=[model], iteration=0, pg_collection=_pg_collection()
        ):
            pass


@pytest.mark.parametrize(
    ("sizes", "tp_shard_dim", "expected_axis"),
    (
        ({"gtp_remat": 2}, None, "gtp"),
        ({"tp": 2}, 0, "tp"),
        ({"cp": 2, "dp": 2}, None, "dp"),
    ),
)
def test_router_diagnostic_metrics_do_not_encode_parallel_axis_policy(
    sizes, tp_shard_dim, expected_axis
):
    observer = build_tensor_metric_observer(
        ["layer-router-health:1"], result_sink=lambda *args: None
    )
    assert observer is not None
    model = _forward_model()
    diagnostics = torch.zeros(1, ROUTER_DIAGNOSTIC_CHANNEL_COUNT, 2)

    with observer.observe_forward_backward(
        model=[model], iteration=0, pg_collection=_pg_collection(**sizes)
    ):
        observe_tensor(
            model.decoder.layers[0].router,
            "router_diagnostics",
            "router_diagnostics",
            diagnostics,
            tp_shard_dim=tp_shard_dim,
        )

    assert observer._prepared_forward_values is not None
    scheduled = observer.scheduled_metrics[0]
    steps = scheduled.metric.start(observer._prepared_forward_values[id(scheduled.metric)])
    assert steps
    assert {request.axis for step in steps for request in step.requests} == {expected_axis}


def test_forward_metrics_reject_full_iteration_cuda_graphs():
    observer = build_tensor_metric_observer(
        ["global-output-logits-l2:1"], result_sink=lambda *args: None
    )
    assert observer is not None
    model = _forward_model(cuda_graph_impl="full_iteration")

    with pytest.raises(NotImplementedError, match="full-iteration CUDA graphs"):
        with observer.observe_forward_backward(
            model=[model], iteration=0, pg_collection=_pg_collection()
        ):
            pass
