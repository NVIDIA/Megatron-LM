# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Structured GPU-sniff measurement tests."""

import importlib.util
import math
import sys
from pathlib import Path

import pytest

from .test_telemetry import _import_fresh, _install_fake_lens
from .test_training_events import _EventSpan, _install_current_span

REPO_ROOT = Path(__file__).resolve().parents[3]
GPU_SNIFF_PATH = REPO_ROOT / "megatron/training/gpu_sniff_test.py"
_SPEC = importlib.util.spec_from_file_location("_test_gpu_sniff_module", GPU_SNIFF_PATH)
gpu_sniff = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = gpu_sniff
_SPEC.loader.exec_module(gpu_sniff)


class FakeGroup:
    def __init__(self, size, rank=0):
        self._size = size
        self._rank = rank

    def size(self):
        return self._size

    def rank(self):
        return self._rank


@pytest.fixture
def fake_cuda_benchmarks(monkeypatch):
    monkeypatch.setattr(gpu_sniff, "MSG_SIZES", [1_000_000])
    monkeypatch.setattr(gpu_sniff, "_time_cuda_op", lambda fn, warmup, iters: 1.0)
    monkeypatch.setattr(gpu_sniff.torch, "randn", lambda *args, **kwargs: object())
    monkeypatch.setattr(gpu_sniff.torch, "empty", lambda *args, **kwargs: object())
    monkeypatch.setattr(gpu_sniff.torch, "empty_like", lambda *args, **kwargs: object())


def test_multiple_gemm_shapes_are_distinct_and_keep_log_names(monkeypatch, fake_cuda_benchmarks):
    monkeypatch.setattr(gpu_sniff, "STANDARD_GEMM_SHAPES", [(2, 3, 4), (5, 6, 7)])
    monkeypatch.setattr(gpu_sniff.torch, "mm", lambda left, right: None)

    results = gpu_sniff.bench_gemms(extra_shapes=[(8, 9, 10, "up-proj/fc1")])

    assert [result.log_name for result in results] == [
        "GEMM throughput (2x3x4, bf16) [TFLOP/s/GPU]",
        "GEMM throughput (5x6x7, bf16) [TFLOP/s/GPU]",
        "GEMM throughput (8x9x10, bf16, up-proj/fc1) [TFLOP/s/GPU]",
    ]
    assert {(result.gemm_m, result.gemm_n, result.gemm_k) for result in results} == {
        (2, 3, 4),
        (5, 6, 7),
        (8, 9, 10),
    }
    assert all(result.benchmark == "gemm" for result in results)
    assert all(result.unit == "TFLOP/s" for result in results)
    assert all(result.gemm_dtype == "bf16" for result in results)
    assert results[-1].gemm_label == "up-proj/fc1"


@pytest.mark.parametrize(
    "benchmark,runner,prefix",
    [
        ("all_reduce", "bench_all_reduce", "All-reduce busbw (global PG"),
        ("reduce_scatter", "bench_reduce_scatter", "Reduce-scatter busbw (TP PG"),
        ("all_to_all", "bench_all_to_all", "All-to-all busbw (EP PG"),
    ],
)
def test_communication_results_carry_workload_parameters(
    monkeypatch, fake_cuda_benchmarks, benchmark, runner, prefix
):
    monkeypatch.setattr(gpu_sniff.dist, "all_reduce", lambda *args, **kwargs: None)
    monkeypatch.setattr(gpu_sniff.dist, "reduce_scatter_tensor", lambda *args, **kwargs: None)
    monkeypatch.setattr(gpu_sniff.dist, "all_to_all_single", lambda *args, **kwargs: None)

    result = getattr(gpu_sniff, runner)(FakeGroup(4))[0]

    assert result.log_name.startswith(prefix)
    assert result.benchmark == benchmark
    assert result.unit == "GB/s"
    assert result.message_size == 1_000_000
    assert result.group_size == 4
    assert math.isfinite(result.value)


def test_nonparticipating_sendrecv_rank_has_no_measurement(monkeypatch, fake_cuda_benchmarks):
    group = FakeGroup(3, rank=2)
    monkeypatch.setattr(gpu_sniff.dist, "get_process_group_ranks", lambda group: [4, 5, 6])

    result = gpu_sniff.bench_sendrecv(group)[0]

    assert result.log_name == ("Send/recv busbw (DP PG, 1 MB, e.g., rank 4 <-> rank 5) [GB/s]")
    assert isinstance(result, gpu_sniff.GpuSniffResult)
    assert result.value is None


@pytest.mark.parametrize("with_sink", [False, True], ids=["no-sink", "sink"])
@pytest.mark.parametrize("failing_family", [None, "all_reduce", "sendrecv"])
def test_families_report_before_next_benchmark(monkeypatch, with_sink, failing_family):
    first = gpu_sniff.GpuSniffResult("gemm-a", "gemm", 10.0, "TFLOP/s", 1, 2, 3, "bf16")
    second = gpu_sniff.GpuSniffResult("gemm-b", "gemm", 20.0, "TFLOP/s", 4, 5, 6, "bf16")
    missing = gpu_sniff.GpuSniffResult("sendrecv-missing", "send_recv", None, "GB/s")
    all_reduce = gpu_sniff.GpuSniffResult("all-reduce", "all_reduce", 30.0, "GB/s")
    nonfinite = gpu_sniff.GpuSniffResult("all-to-all", "all_to_all", float("inf"), "GB/s")
    sendrecv = gpu_sniff.GpuSniffResult("sendrecv", "sendrecv", 40.0, "GB/s")
    families = [
        ("gemms", [first, second]),
        ("all_reduce", [all_reduce]),
        ("reduce_scatter", []),
        ("all_to_all", [nonfinite]),
        ("sendrecv", [missing, sendrecv]),
    ]
    calls = []
    failure = RuntimeError("benchmark failed")
    groups = {name: object() for name, _ in families}
    extra_shapes = [(1, 2, 3, "extra")]

    def benchmark(name, results):
        def run(*args, **kwargs):
            if name == "gemms":
                assert not args
                assert kwargs == {"extra_shapes": extra_shapes}
            else:
                assert args == (groups[name],)
                assert not kwargs
            calls.append(("bench", name))
            if name == failing_family:
                raise failure
            return results

        return run

    def gather(name, value, hostnames):
        assert hostnames == ["host-a"]
        calls.append(("gather", name, "nan" if math.isnan(value) else value))
        return name == "gemm-b"  # An outlier must not short-circuit later gathers.

    monkeypatch.setattr(gpu_sniff.dist, "get_rank", lambda: 1)
    monkeypatch.setattr(gpu_sniff.dist, "barrier", lambda: calls.append(("barrier",)))
    monkeypatch.setattr(gpu_sniff, "_gather_hostnames", lambda: ["host-a"])
    for name, results in families:
        monkeypatch.setattr(gpu_sniff, "bench_" + name, benchmark(name, results))
    monkeypatch.setattr(gpu_sniff, "_gather_and_check", gather)

    def run():
        gpu_sniff.run_sniff_tests(
            groups["all_to_all"],
            groups["sendrecv"],
            ar_group=groups["all_reduce"],
            tp_group=groups["reduce_scatter"],
            extra_gemm_shapes=extra_shapes,
            measurement_sink=(
                (lambda result: calls.append(("emit", result.log_name, result.value)))
                if with_sink
                else None
            ),
        )

    expected = [
        ("bench", "gemms"),
        ("emit", "gemm-a", 10.0),
        ("gather", "gemm-a", 10.0),
        ("emit", "gemm-b", 20.0),
        ("gather", "gemm-b", 20.0),
        ("bench", "all_reduce"),
        ("emit", "all-reduce", 30.0),
        ("gather", "all-reduce", 30.0),
        ("bench", "reduce_scatter"),
        ("bench", "all_to_all"),
        ("gather", "all-to-all", float("inf")),
        ("bench", "sendrecv"),
        ("gather", "sendrecv-missing", "nan"),
        ("emit", "sendrecv", 40.0),
        ("gather", "sendrecv", 40.0),
        ("barrier",),
    ]
    if failing_family is not None:
        with pytest.raises(RuntimeError) as caught:
            run()
        assert caught.value is failure
        expected = expected[: expected.index(("bench", failing_family)) + 1]
    else:
        run()
    if not with_sink:
        expected = [call for call in expected if call[0] != "emit"]
    assert calls == expected


def test_trainer_measurement_adapter_preserves_result_fields_and_invocation_parent(
    monkeypatch, telemetry_module
):
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

    telemetry = telemetry_module
    monkeypatch.setattr(telemetry, "is_enabled", lambda group: True)
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    tracer = provider.get_tracer("gpu-adapter")
    result = gpu_sniff.GpuSniffResult(
        "measurement",
        "gemm",
        10.5,
        "TFLOP/s",
        2,
        3,
        4,
        "bf16",
        "up",
        message_size=32,
        group_size=8,
        peer_stride=2,
        peer_rank=6,
    )
    try:
        with tracer.start_as_current_span("creation"):
            sink = telemetry.gpu_sniff_measurement_sink(17)
        with tracer.start_as_current_span("invocation"):
            sink(result)
        creation, invocation = exporter.get_finished_spans()
        assert not creation.events
        (event,) = invocation.events
        assert event.name == telemetry.EVENT_GPU_SNIFF_MEASUREMENT
        assert dict(event.attributes) == {
            telemetry.MEASUREMENT_DOMAIN: telemetry.MEASUREMENT_DOMAIN_LOCAL,
            telemetry.GPU_SNIFF_BENCHMARK: "gemm",
            telemetry.GPU_SNIFF_VALUE: 10.5,
            telemetry.GPU_SNIFF_UNIT: "TFLOP/s",
            telemetry.TRAINING_STEP: 17,
            telemetry.GPU_SNIFF_GEMM_M: 2,
            telemetry.GPU_SNIFF_GEMM_N: 3,
            telemetry.GPU_SNIFF_GEMM_K: 4,
            telemetry.GPU_SNIFF_GEMM_DTYPE: "bf16",
            telemetry.GPU_SNIFF_GEMM_LABEL: "up",
            telemetry.GPU_SNIFF_MESSAGE_SIZE: 32,
            telemetry.GPU_SNIFF_GROUP_SIZE: 8,
            telemetry.GPU_SNIFF_PEER_STRIDE: 2,
            telemetry.GPU_SNIFF_PEER_RANK: 6,
        }
    finally:
        provider.shutdown()


def test_gpu_sniff_events_have_structured_workload_identity(monkeypatch):
    _install_fake_lens(monkeypatch)
    telemetry = _import_fresh(monkeypatch)
    span = _EventSpan()
    _install_current_span(monkeypatch, span)

    startup = gpu_sniff.GpuSniffResult(
        "startup gemm", "gemm", 900.0, "TFLOP/s", 1024, 2048, 4096, "bf16", "up-proj/fc1"
    )
    periodic = gpu_sniff.GpuSniffResult(
        "periodic send/recv",
        "send_recv",
        150.0,
        "GB/s",
        message_size=1048576,
        group_size=8,
        peer_stride=2,
        peer_rank=0,
    )
    telemetry.gpu_sniff_measurement_sink(None)(startup)
    telemetry.gpu_sniff_measurement_sink(23)(periodic)

    _, startup_attributes = span.events[0]
    assert telemetry.TRAINING_STEP not in startup_attributes
    assert startup_attributes[telemetry.MEASUREMENT_DOMAIN] == telemetry.MEASUREMENT_DOMAIN_LOCAL
    assert startup_attributes[telemetry.GPU_SNIFF_GEMM_LABEL] == "up-proj/fc1"

    _, periodic_attributes = span.events[1]
    assert periodic_attributes[telemetry.TRAINING_STEP] == 23
    assert periodic_attributes[telemetry.GPU_SNIFF_MESSAGE_SIZE] == 1048576
    assert periodic_attributes[telemetry.GPU_SNIFF_PEER_RANK] == 0


EXPECTED_MEASUREMENT_NAMES = {
    'GPU_SNIFF_BENCHMARK': 'nv.dl.resiliency.gpu_sniff.benchmark',
    'GPU_SNIFF_VALUE': 'nv.dl.resiliency.gpu_sniff.value',
    'GPU_SNIFF_UNIT': 'nv.dl.resiliency.gpu_sniff.unit',
    'GPU_SNIFF_GEMM_M': 'nv.dl.resiliency.gpu_sniff.gemm.m',
    'GPU_SNIFF_GEMM_N': 'nv.dl.resiliency.gpu_sniff.gemm.n',
    'GPU_SNIFF_GEMM_K': 'nv.dl.resiliency.gpu_sniff.gemm.k',
    'GPU_SNIFF_GEMM_DTYPE': 'nv.dl.resiliency.gpu_sniff.gemm.dtype',
    'GPU_SNIFF_GEMM_LABEL': 'nv.dl.resiliency.gpu_sniff.gemm.label',
    'GPU_SNIFF_MESSAGE_SIZE': 'nv.dl.resiliency.gpu_sniff.message_size',
    'GPU_SNIFF_GROUP_SIZE': 'nv.dl.resiliency.gpu_sniff.group_size',
    'GPU_SNIFF_PEER_STRIDE': 'nv.dl.resiliency.gpu_sniff.peer_stride',
    'GPU_SNIFF_PEER_RANK': 'nv.dl.resiliency.gpu_sniff.peer_rank',
    'EVENT_GPU_SNIFF_MEASUREMENT': 'nv.dl.resiliency.gpu_sniff.measurement',
}


def test_gpu_measurement_names_are_canonical(telemetry_module):
    assert {
        name: getattr(telemetry_module, name) for name in EXPECTED_MEASUREMENT_NAMES
    } == EXPECTED_MEASUREMENT_NAMES


def test_gpu_measurement_docs_name_the_event():
    text = (REPO_ROOT / "docs/user-guide/observability/metrics.md").read_text()
    assert "nv.dl.resiliency.gpu_sniff.measurement" in text
