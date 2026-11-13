# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Tests for Megatron-owned trainer Resource mapping."""

import builtins
import importlib.util
import subprocess
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
RESOURCE_ATTRS_PATH = REPO_ROOT / "megatron/core/telemetry/resource_attrs.py"


def _args(**overrides):
    values = {
        "rank": 3,
        "world_size": 8,
        "local_rank": 1,
        "tensor_model_parallel_size": 2,
        "pipeline_model_parallel_size": 4,
        "data_parallel_size": 8,
        "global_batch_size": 128,
        "micro_batch_size": 4,
        "seq_length": 2048,
        "optimizer": "adam",
        "recompute_granularity": "selective",
        "train_iters": 100,
        "train_samples": None,
        "train_tokens": None,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _load_resource_attrs(monkeypatch):
    torch = types.ModuleType("torch")
    torch.__version__ = "2.9.0"
    torch.version = SimpleNamespace(cuda="12.8")
    torch.cuda = SimpleNamespace(nccl=SimpleNamespace(version=lambda: (2, 27, 3)))
    monkeypatch.setitem(sys.modules, "torch", torch)

    module_name = "_megatron_test_resource_attrs"
    spec = importlib.util.spec_from_file_location(module_name, RESOURCE_ATTRS_PATH)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, module_name, module)
    spec.loader.exec_module(module)
    return module


def test_module_imports_without_optional_lens(monkeypatch):
    real_import = builtins.__import__

    def blocked_import(name, *args, **kwargs):
        if name == "nemo" or name.startswith("nemo."):
            raise ModuleNotFoundError("nemo-lens intentionally hidden", name="nemo")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", blocked_import)
    resource_attrs = _load_resource_attrs(monkeypatch)
    with pytest.raises(ModuleNotFoundError, match="nemo-lens is required"):
        resource_attrs.build_telemetry_resource_attrs(_args())


def test_maps_megatron_arguments_and_application_versions(monkeypatch):
    resource_attrs = _load_resource_attrs(monkeypatch)
    package_info = types.ModuleType("megatron.core.package_info")
    package_info.__version__ = "0.20.0+abc123"
    monkeypatch.setitem(sys.modules, "megatron.core.package_info", package_info)
    monkeypatch.setattr(
        resource_attrs,
        "_distribution_version",
        lambda *names: "2.3.0" if "transformer-engine" in names else None,
    )

    attrs = resource_attrs.build_telemetry_resource_attrs(_args())

    assert attrs == {
        "nv.dl.rank": 3,
        "nv.dl.world_size": 8,
        "nv.dl.local_rank": 1,
        "nv.dl.provider.name": "mcore",
        "nv.dl.topology.size.tp": 2,
        "nv.dl.topology.size.pp": 4,
        "nv.dl.topology.size.dp": 8,
        "nv.dl.training.config.global_batch_size": 128,
        "nv.dl.training.config.micro_batch_size": 4,
        "nv.dl.training.config.sequence_length": 2048,
        "nv.dl.training.config.optimizer": "adam",
        "nv.dl.training.config.recompute_granularity": "selective",
        "nv.dl.training.target.train_iters": 100,
        "nv.dl.training.target.train_samples": 12_800,
        "nv.dl.training.target.train_tokens": 26_214_400,
        "nv.dl.software.torch": "2.9.0",
        "nv.dl.software.cuda": "12.8",
        "nv.dl.software.nccl": "2.27.3",
        "nv.dl.software.transformer_engine": "2.3.0",
        "nv.mcore.version": "0.20.0+abc123",
    }


def test_non_training_resource_omits_training_attributes(monkeypatch):
    resource_attrs = _load_resource_attrs(monkeypatch)
    monkeypatch.setattr(resource_attrs, "_software_attributes", lambda: {})

    attrs = resource_attrs.build_telemetry_resource_attrs(_args(), include_training=False)

    assert attrs["nv.dl.rank"] == 3
    assert not any(name.startswith("nv.dl.training.") for name in attrs)


@pytest.mark.parametrize(
    ("overrides", "samples", "tokens"),
    [
        ({"train_iters": "invalid"}, None, None),
        ({"global_batch_size": None}, None, None),
        ({"seq_length": "invalid"}, 12_800, None),
        ({"train_samples": 17, "train_tokens": 23}, 17, 23),
    ],
)
def test_target_derivations_handle_missing_invalid_and_explicit_values(
    monkeypatch, overrides, samples, tokens
):
    resource_attrs = _load_resource_attrs(monkeypatch)
    monkeypatch.setattr(resource_attrs, "_software_attributes", lambda: {})

    attrs = resource_attrs.build_telemetry_resource_attrs(_args(**overrides))

    assert attrs.get("nv.dl.training.target.train_samples") == samples
    assert attrs.get("nv.dl.training.target.train_tokens") == tokens


@pytest.mark.parametrize("array", [False, True])
@pytest.mark.parametrize("restart", [0, 1])
def test_plain_trainer_derives_worker_attempt_identity(monkeypatch, array, restart):
    resource_attrs = _load_resource_attrs(monkeypatch)
    monkeypatch.setattr(resource_attrs, "_software_attributes", lambda: {})
    monkeypatch.setattr(resource_attrs, "detect_gpu", lambda **_kwargs: {})
    monkeypatch.delenv("OTEL_RESOURCE_ATTRIBUTES", raising=False)
    monkeypatch.setenv("SLURM_CLUSTER_NAME", "test-cluster")
    monkeypatch.setenv("SLURM_JOB_ID", "123")
    monkeypatch.setenv("TORCHELASTIC_RESTART_COUNT", str(restart))
    for key in ("SLURM_ARRAY_JOB_ID", "SLURM_ARRAY_TASK_ID", "SLURM_RESTART_COUNT"):
        monkeypatch.delenv(key, raising=False)
    if array:
        monkeypatch.setenv("SLURM_ARRAY_JOB_ID", "120")
        monkeypatch.setenv("SLURM_ARRAY_TASK_ID", "3")
    first = resource_attrs.compose_telemetry_resource_attrs(_args(rank=0))
    second = resource_attrs.compose_telemetry_resource_attrs(_args(rank=1))
    expected = resource_attrs.derive_nv_dl_run_uuid()
    assert first["nv.dl.run.uuid"] == second["nv.dl.run.uuid"] == expected
    monkeypatch.setenv("TORCHELASTIC_RESTART_COUNT", str(restart + 1))
    restarted = resource_attrs.compose_telemetry_resource_attrs(_args())
    assert restarted["nv.dl.run.uuid"] != expected
    monkeypatch.setenv("OTEL_RESOURCE_ATTRIBUTES", "nv.dl.run.uuid=nvrx-cycle-owned")
    inherited = resource_attrs.compose_telemetry_resource_attrs(_args())
    assert inherited["nv.dl.run.uuid"] == "nvrx-cycle-owned"


def test_local_trainer_without_attempt_identity_does_not_invent_uuid(monkeypatch):
    resource_attrs = _load_resource_attrs(monkeypatch)
    monkeypatch.setattr(resource_attrs, "_software_attributes", lambda: {})
    monkeypatch.setattr(resource_attrs, "detect_gpu", lambda **_kwargs: {})
    for key in (
        "OTEL_RESOURCE_ATTRIBUTES",
        "SLURM_JOB_ID",
        "SLURM_ARRAY_JOB_ID",
        "TORCHELASTIC_RUN_ID",
    ):
        monkeypatch.delenv(key, raising=False)
    assert "nv.dl.run.uuid" not in resource_attrs.compose_telemetry_resource_attrs(_args())


def test_composes_inherited_detected_gpu_and_trainer_identity_once(monkeypatch):
    resource_attrs = _load_resource_attrs(monkeypatch)
    monkeypatch.setattr(resource_attrs, "_software_attributes", lambda: {})
    monkeypatch.setenv(
        "OTEL_RESOURCE_ATTRIBUTES",
        (
            "nv.dl.rank=9,nv.dl.world_size=,nv.dl.topology.size.tp=6,"
            "nv.dl.role=checkpoint_worker,nv.dl.run.uuid=attempt-7,"
            "nv.gpu.index=4,nv.gpu.uuid=,k8s.pod.name=old-pod,"
            "slurm.job.id=old-job,host.name=parent-host,process.pid=12,host.gpu.count=8"
        ),
    )
    monkeypatch.setenv("OTEL_SERVICE_NAME", "   ")
    monkeypatch.setattr(
        resource_attrs, "detect_slurm", lambda: {"slurm.job.id": "detected-job", "slurm.nnodes": 2}
    )
    monkeypatch.setattr(resource_attrs, "detect_kubernetes", lambda: {"k8s.pod.name": "new-pod"})
    gpu_calls = []

    def detect_gpu(*, local_rank):
        gpu_calls.append(local_rank)
        return {"nv.gpu.index": 7, "nv.gpu.uuid": "GPU-detected"}

    monkeypatch.setattr(resource_attrs, "detect_gpu", detect_gpu)

    attrs = resource_attrs.compose_telemetry_resource_attrs(
        _args(otel_service_name=None, local_rank=1)
    )

    assert attrs["nv.dl.rank"] == 9
    assert type(attrs["nv.dl.rank"]) is int
    assert attrs["nv.dl.world_size"] == ""
    assert attrs["nv.dl.topology.size.tp"] == 6
    assert type(attrs["nv.dl.topology.size.tp"]) is int
    assert attrs["nv.dl.role"] == "trainer"
    assert attrs["nv.dl.run.uuid"] == "attempt-7"
    assert attrs["nv.gpu.index"] == 4
    assert type(attrs["nv.gpu.index"]) is int
    assert attrs["nv.gpu.uuid"] == ""
    assert attrs["slurm.job.id"] == "detected-job"
    assert attrs["slurm.nnodes"] == 2
    assert attrs["k8s.pod.name"] == "new-pod"
    assert attrs["service.name"] == "megatron-lm"
    assert not ({"host.name", "process.pid", "host.gpu.count"} & attrs.keys())
    assert gpu_calls == [1]


@pytest.mark.parametrize(
    ("override", "environment", "inherited", "expected"),
    [
        ("cli-name", "env-name", "carrier-name", "cli-name"),
        (None, "env-name", "carrier-name", "env-name"),
        (None, "   ", "carrier-name", "carrier-name"),
        (None, None, None, "megatron-lm"),
    ],
)
def test_service_name_precedence(monkeypatch, override, environment, inherited, expected):
    resource_attrs = _load_resource_attrs(monkeypatch)
    monkeypatch.setattr(resource_attrs, "_software_attributes", lambda: {})
    monkeypatch.setattr(resource_attrs, "detect_slurm", lambda: {})
    monkeypatch.setattr(resource_attrs, "detect_kubernetes", lambda: {})
    monkeypatch.setattr(resource_attrs, "detect_gpu", lambda *, local_rank: {})
    if inherited is None:
        monkeypatch.delenv("OTEL_RESOURCE_ATTRIBUTES", raising=False)
    else:
        monkeypatch.setenv("OTEL_RESOURCE_ATTRIBUTES", f"service.name={inherited}")
    if environment is None:
        monkeypatch.delenv("OTEL_SERVICE_NAME", raising=False)
    else:
        monkeypatch.setenv("OTEL_SERVICE_NAME", environment)

    attrs = resource_attrs.compose_telemetry_resource_attrs(_args(otel_service_name=override))

    assert attrs["service.name"] == expected


def _run_real_lens_resource_probe():
    import os

    monkeypatch = pytest.MonkeyPatch()
    resource_attrs = _load_resource_attrs(monkeypatch)
    monkeypatch.setattr(resource_attrs, "_software_attributes", lambda: {})
    monkeypatch.setattr(resource_attrs, "detect_gpu", lambda *, local_rank: {})
    monkeypatch.setenv(
        "OTEL_RESOURCE_ATTRIBUTES",
        (
            "service.name=carrier-service,nv.dl.rank=7,k8s.pod.name=old-pod,"
            "host.name=parent-host,process.pid=12,host.gpu.count=8"
        ),
    )
    monkeypatch.setenv("KUBERNETES_SERVICE_HOST", "kubernetes.default.svc")
    monkeypatch.setenv("K8S_POD_NAME", "new-pod")
    monkeypatch.delenv("OTEL_SERVICE_NAME", raising=False)

    from nemo.lens import NemoLensConfig, setup_telemetry
    from nemo.lens.resources.attributes import (
        format_otel_resource_attributes,
        parse_otel_resource_attributes,
    )
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

    attrs = resource_attrs.compose_telemetry_resource_attrs(_args(otel_service_name=None))
    assert attrs["nv.dl.rank"] == 7
    assert type(attrs["nv.dl.rank"]) is int
    assert attrs["k8s.pod.name"] == "new-pod"
    assert not ({"host.name", "process.pid", "host.gpu.count"} & attrs.keys())
    published = parse_otel_resource_attributes(format_otel_resource_attributes(attrs))
    assert published["k8s.pod.name"] == "new-pod"
    exporter = InMemorySpanExporter()
    config = NemoLensConfig(
        enabled=True, service_name=attrs["service.name"], metrics_enabled=False, exporter="console"
    )
    handle = setup_telemetry(config, resource_attributes=attrs, span_exporter=exporter)
    assert handle.is_exporting
    with handle.tracer.start_as_current_span("megatron.resource.probe"):
        pass
    handle.shutdown()
    spans = exporter.get_finished_spans()
    assert [span.name for span in spans] == ["megatron.resource.probe"]
    resource = spans[0].resource.attributes
    assert resource["nv.dl.rank"] == 7
    assert type(resource["nv.dl.rank"]) is int
    assert resource["k8s.pod.name"] == "new-pod"
    assert resource["host.name"] != "parent-host"
    assert resource["process.pid"] == os.getpid()
    monkeypatch.undo()


def test_real_lens_setup_accepts_composed_resource():
    result = subprocess.run(
        [sys.executable, str(Path(__file__).resolve()), "--real-lens-resource-probe"],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


if __name__ == "__main__":
    assert sys.argv[1:] == ["--real-lens-resource-probe"]
    _run_real_lens_resource_probe()
