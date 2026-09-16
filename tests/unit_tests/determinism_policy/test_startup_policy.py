# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""CPU policy contracts; use --confcutdir here to avoid the GPU suite's conftest.

Load the policy module directly because importing the full MCore package needs
Triton. The separate library integration test exercises the public import on GPU.
"""

import importlib.util
import json
import logging
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

ROOT = Path(__file__).resolve().parents[3]


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, ROOT / path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def policy(monkeypatch):
    module = load_module("_test_core_determinism", "megatron/core/determinism.py")
    enabled = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    benchmark = torch.backends.cudnn.benchmark
    fill_uninitialized = torch.utils.deterministic.fill_uninitialized_memory
    cudnn_det = torch.backends.cudnn.deterministic
    # Model startup state without touching the GPU test process's real handles.
    monkeypatch.setattr(torch.cuda, "is_initialized", lambda: False)
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: False)
    env = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith(("NCCL_", "NVTE_", "CUBLAS_", "MAMBA_", "CAUSAL_CONV1D_", "TRITON_"))
    }
    monkeypatch.setattr(os, "environ", env)
    yield module
    torch.use_deterministic_algorithms(enabled, warn_only=warn_only)
    torch.backends.cudnn.benchmark = benchmark
    torch.utils.deterministic.fill_uninitialized_memory = fill_uninitialized
    torch.backends.cudnn.deterministic = cudnn_det


def test_strict_policy_reports_effective_settings_without_seeding(policy, caplog):
    state = torch.get_rng_state().clone()
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.benchmark = True
    config = {"deterministic_mode": True}
    with caplog.at_level(logging.INFO):
        report = policy.configure_determinism(config)
    assert report["torch"]["deterministic_algorithms"] is True
    assert report["torch"]["warn_only"] is False
    assert report["torch"]["fill_uninitialized_memory"] is False
    assert report["torch"]["cudnn_benchmark"] is False
    assert report["torch"]["cudnn_deterministic"] is True
    assert report["environment"]["CUBLAS_WORKSPACE_CONFIG"] == ":4096:8"
    assert report["environment"]["MAMBA_DETERMINISTIC"] == "1"
    assert report["environment"]["CAUSAL_CONV1D_DETERMINISTIC"] == "1"
    assert report["environment"]["TRITON_CACHE_AUTOTUNING"] is None
    assert json.loads(json.dumps(report)) == report
    assert "Determinism policy:" in caplog.text
    assert torch.equal(state, torch.get_rng_state())
    assert config == {"deterministic_mode": True}


@pytest.mark.parametrize(
    "config",
    [
        {},
        {"deterministic_mode": False},
        {"deterministic_mode": True, "cross_entropy_loss_fusion": True},
        {"deterministic_mode": True, "tp_comm_overlap": True},
        {"deterministic_mode": True, "moe_router_fusion": True},
        {"deterministic_mode": True, "moe_router_aux_loss_fusion": True},
    ],
)
def test_incompatible_config_rejected_before_mutation(policy, config):
    env = dict(os.environ)
    enabled = torch.are_deterministic_algorithms_enabled()
    with pytest.raises(AssertionError):
        policy.configure_determinism(config)
    assert dict(os.environ) == env
    assert torch.are_deterministic_algorithms_enabled() == enabled


def test_fused_topk_with_explicit_unfused_aux_loss_is_supported(policy):
    report = policy.configure_determinism(
        SimpleNamespace(
            deterministic_mode=True, moe_router_fusion=True, moe_router_aux_loss_fusion=False
        )
    )
    assert report["options"]["moe_router_aux_loss_fusion"] is False


@pytest.mark.parametrize(
    "key,value",
    [
        ("NCCL_ALGO", "Tree"),
        ("NCCL_ALGO", "Ring,Tree"),
        ("NCCL_ALGO", ""),
        ("NVTE_ALLOW_NONDETERMINISTIC_ALGO", "1"),
        ("CUBLAS_WORKSPACE_CONFIG", "invalid"),
        ("MAMBA_DETERMINISTIC", "0"),
        ("CAUSAL_CONV1D_DETERMINISTIC", "0"),
        ("TRITON_CACHE_AUTOTUNING", "true"),
        ("TRITON_CACHE_AUTOTUNING", "1"),
    ],
)
def test_invalid_environment_is_atomic(policy, key, value):
    os.environ[key] = value
    env = dict(os.environ)
    with pytest.raises(AssertionError):
        policy.configure_determinism({"deterministic_mode": True})
    assert dict(os.environ) == env
    assert policy._configured_pid is None


def test_valid_launch_overrides_are_preserved_and_reported(policy):
    overrides = {
        "NCCL_ALGO": "Ring,CollnetDirect",
        "CUBLAS_WORKSPACE_CONFIG": ":16:8",
        "TRITON_CACHE_AUTOTUNING": "1",
        "TRITON_CACHE_DIR": "/shared/cache",
        "TRITON_AUTOTUNE_BLOCK_SIZE_M": "64",
        "CUDA_DEVICE_MAX_CONNECTIONS": "32",
    }
    os.environ.update(overrides)
    report = policy.configure_determinism({"deterministic_mode": True})
    for key, value in overrides.items():
        assert os.environ[key] == report["environment"][key] == value


@pytest.mark.parametrize("backend", ["cuda", "distributed"])
def test_late_first_call_cannot_claim_early_setup(policy, monkeypatch, backend):
    monkeypatch.setattr(getattr(torch, backend), "is_initialized", lambda: True)
    env = dict(os.environ)
    enabled = torch.are_deterministic_algorithms_enabled()
    with pytest.raises(RuntimeError, match="before CUDA or process-group"):
        policy.configure_determinism({"deterministic_mode": True})
    assert dict(os.environ) == env
    assert torch.are_deterministic_algorithms_enabled() == enabled


def test_same_policy_can_be_rechecked_after_initialization(policy, monkeypatch):
    config = {"deterministic_mode": True}
    before = policy.configure_determinism(config)
    monkeypatch.setattr(torch.cuda, "is_initialized", lambda: True)
    assert policy.configure_determinism(config) == before
    before["environment"]["NCCL_ALGO"] = "Tree"
    assert policy.configure_determinism(config)["environment"]["NCCL_ALGO"] == "Ring"


@pytest.mark.parametrize(
    "change",
    ["environment", "missing_environment", "torch", "warn_only", "benchmark", "cudnn", "fork"],
)
def test_late_policy_drift_is_rejected(policy, monkeypatch, change):
    policy.configure_determinism({"deterministic_mode": True})
    monkeypatch.setattr(torch.cuda, "is_initialized", lambda: True)
    if change == "environment":
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
    elif change == "missing_environment":
        os.environ.pop("CUBLAS_WORKSPACE_CONFIG")
    elif change == "torch":
        torch.use_deterministic_algorithms(False)
    elif change == "warn_only":
        torch.use_deterministic_algorithms(True, warn_only=True)
    elif change == "benchmark":
        torch.backends.cudnn.benchmark = True
    elif change == "cudnn":
        torch.backends.cudnn.deterministic = False
    else:
        policy._configured_pid -= 1
    with pytest.raises(RuntimeError, match="fresh process"):
        policy.configure_determinism({"deterministic_mode": True})


def test_actual_model_parallel_config_and_training_adapter_share_policy(policy, monkeypatch):
    model_module = load_module(
        "_test_model_parallel_config", "megatron/core/model_parallel_config.py"
    )
    model = model_module.ModelParallelConfig(deterministic_mode=True)
    model_report = policy.configure_determinism(model)
    monkeypatch.setitem(sys.modules, "megatron.core.determinism", policy)
    adapter = load_module("_test_training_determinism", "megatron/training/determinism.py")
    args = SimpleNamespace(cross_entropy_loss_fusion=False, tp_comm_overlap=False)
    assert adapter.apply_determinism_to_args(args) == model_report
    assert vars(args) == {"cross_entropy_loss_fusion": False, "tp_comm_overlap": False}
    assert adapter.apply_determinism_env is policy.apply_determinism_env


def test_optimized_python_still_rejects_invalid_settings():
    script = '''
import importlib.util
spec = importlib.util.spec_from_file_location('policy', 'megatron/core/determinism.py')
policy = importlib.util.module_from_spec(spec)
spec.loader.exec_module(policy)
for action in (
    lambda: policy.validate_determinism_config({'deterministic_mode':True, 'tp_comm_overlap':True}),
    lambda: policy.apply_determinism_env({'NCCL_ALGO':'Tree'}),
):
    try:
        action()
    except AssertionError:
        continue
    raise RuntimeError('Validation was optimized away')
'''
    subprocess.run([sys.executable, "-O", "-c", script], cwd=ROOT, check=True, timeout=60)
