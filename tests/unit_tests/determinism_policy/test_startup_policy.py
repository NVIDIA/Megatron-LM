# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""CPU policy contracts; use --confcutdir here to avoid the GPU suite's conftest.

The public early package imports without the Core GPU dependencies. The separate
library integration test exercises subsequent Core and training imports on GPU.
"""

import ast
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


def clean_environment():
    """Keep inherited device placement while isolating policy-specific overrides."""
    return {
        key: value
        for key, value in os.environ.items()
        if not key.startswith(("NCCL_", "NVTE_", "CUBLAS_", "MAMBA_", "CAUSAL_CONV1D_", "TRITON_"))
    }


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, ROOT / path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def policy(monkeypatch):
    from megatron.determinism import _policy as module

    monkeypatch.setattr(module, "_configured_pid", None)
    monkeypatch.setattr(module, "_configured_environment", None)
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
    torch.utils.deterministic.fill_uninitialized_memory = True
    torch.backends.cudnn.benchmark = True
    config = {"deterministic_mode": True}
    with caplog.at_level(logging.DEBUG):
        report = policy.configure_determinism(config)
    assert report["torch"]["deterministic_algorithms"] is True
    assert report["torch"]["warn_only"] is False
    assert report["torch"]["fill_uninitialized_memory"] is False
    assert torch.utils.deterministic.fill_uninitialized_memory is False
    assert report["torch"]["cudnn_benchmark"] is False
    assert report["torch"]["cudnn_deterministic"] is True
    assert report["environment"]["CUBLAS_WORKSPACE_CONFIG"] == ":4096:8"
    assert report["environment"]["MAMBA_DETERMINISTIC"] == "1"
    assert report["environment"]["CAUSAL_CONV1D_DETERMINISTIC"] == "1"
    assert report["environment"]["TRITON_CACHE_AUTOTUNING"] == "0"
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
    torch.utils.deterministic.fill_uninitialized_memory = True
    with pytest.raises(AssertionError):
        policy.configure_determinism(config)
    assert dict(os.environ) == env
    assert torch.are_deterministic_algorithms_enabled() == enabled
    assert torch.utils.deterministic.fill_uninitialized_memory is True


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
    torch.utils.deterministic.fill_uninitialized_memory = True
    with pytest.raises(AssertionError):
        policy.configure_determinism({"deterministic_mode": True})
    assert dict(os.environ) == env
    assert policy._configured_pid is None
    assert torch.utils.deterministic.fill_uninitialized_memory is True


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
    torch.utils.deterministic.fill_uninitialized_memory = True
    with pytest.raises(RuntimeError, match="before CUDA or process-group"):
        policy.configure_determinism({"deterministic_mode": True})
    assert dict(os.environ) == env
    assert torch.are_deterministic_algorithms_enabled() == enabled
    assert torch.utils.deterministic.fill_uninitialized_memory is True


@pytest.mark.parametrize("backend", ["cuda", "distributed"])
def test_same_policy_can_be_rechecked_after_initialization(policy, monkeypatch, backend):
    config = {"deterministic_mode": True}
    before = policy.configure_determinism(config)
    monkeypatch.setattr(getattr(torch, backend), "is_initialized", lambda: True)
    # Diagnostic fill can be re-enabled after startup; another successful setup
    # restores the training default, as the legacy training helper does.
    torch.utils.deterministic.fill_uninitialized_memory = True
    assert policy.configure_determinism(config) == before
    assert torch.utils.deterministic.fill_uninitialized_memory is False
    before["environment"]["NCCL_ALGO"] = "Tree"
    assert policy.configure_determinism(config)["environment"]["NCCL_ALGO"] == "Ring"


def test_backend_default_cannot_enable_autotuning_after_bootstrap(policy, monkeypatch):
    config = {"deterministic_mode": True}
    before = policy.configure_determinism(config)
    monkeypatch.setattr(torch.cuda, "is_initialized", lambda: True)
    # vLLM uses this default during import, after other backends can initialize CUDA.
    os.environ.setdefault("TRITON_CACHE_AUTOTUNING", "1")
    assert os.environ["TRITON_CACHE_AUTOTUNING"] == "0"
    assert policy.configure_determinism(config) == before
    os.environ["TRITON_CACHE_AUTOTUNING"] = "1"
    os.environ["TRITON_CACHE_DIR"] = "/shared/cache"
    with pytest.raises(RuntimeError, match="fresh process"):
        policy.configure_determinism(config)


@pytest.mark.parametrize(
    "change",
    ["environment", "missing_environment", "torch", "warn_only", "benchmark", "cudnn", "fork"],
)
def test_late_policy_drift_is_rejected(policy, monkeypatch, change):
    policy.configure_determinism({"deterministic_mode": True})
    torch.utils.deterministic.fill_uninitialized_memory = True
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
    assert torch.utils.deterministic.fill_uninitialized_memory is True


def test_actual_model_parallel_config_and_training_adapter_share_policy(policy):
    model_module = load_module(
        "_test_model_parallel_config", "megatron/core/model_parallel_config.py"
    )
    model = model_module.ModelParallelConfig(deterministic_mode=True)
    model_report = policy.configure_determinism(model)
    adapter = load_module("_test_training_determinism", "megatron/training/determinism.py")
    args = SimpleNamespace(cross_entropy_loss_fusion=False, tp_comm_overlap=False)
    assert adapter.apply_determinism_to_args(args) == model_report
    assert vars(args) == {"cross_entropy_loss_fusion": False, "tp_comm_overlap": False}
    assert adapter.apply_determinism_env is policy.apply_determinism_env


def test_training_adapter_rejects_disabled_mode_after_bootstrap(policy):
    policy.configure_determinism({"deterministic_mode": True})
    adapter = load_module("_test_training_determinism", "megatron/training/determinism.py")
    args = SimpleNamespace(
        deterministic_mode=False, cross_entropy_loss_fusion=False, tp_comm_overlap=False
    )
    with pytest.raises(AssertionError, match="deterministic_mode=True"):
        adapter.apply_determinism_to_args(args)


def test_configured_status_tracks_process_intent_even_after_drift(policy):
    assert not policy.is_determinism_configured()
    policy.configure_determinism({"deterministic_mode": True})
    assert policy.is_determinism_configured()
    torch.use_deterministic_algorithms(False)
    assert policy.is_determinism_configured()  # Validation must still reject drift.
    policy._configured_pid -= 1
    assert not policy.is_determinism_configured()


def test_optimized_python_still_rejects_invalid_settings():
    script = '''
import megatron.determinism as policy
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
    subprocess.run(
        [sys.executable, "-O", "-c", script],
        cwd=ROOT,
        env=clean_environment(),
        check=True,
        timeout=60,
    )


def test_public_early_import_needs_no_core_and_does_not_initialize_cuda():
    script = '''
import os
import sys
import torch
before = dict(os.environ)
state = torch.get_rng_state().clone()
enabled = torch.are_deterministic_algorithms_enabled()
import megatron.determinism as policy
assert dict(os.environ) == before
assert torch.are_deterministic_algorithms_enabled() == enabled
assert not policy.is_determinism_configured()
policy.configure_determinism({'deterministic_mode': True})
assert policy.is_determinism_configured()
assert not torch.cuda.is_initialized()
assert not torch.distributed.is_initialized()
assert torch.equal(state, torch.get_rng_state())
assert not any(name == 'megatron.core' or name.startswith(('megatron.core.', 'transformer_engine'))
               for name in sys.modules)
'''
    subprocess.run(
        [sys.executable, "-c", script], cwd=ROOT, env=clean_environment(), check=True, timeout=60
    )


def test_cli_bootstrap_only_applies_when_requested(policy):
    from megatron.determinism import bootstrap_training_determinism

    before = dict(os.environ)
    enabled = torch.are_deterministic_algorithms_enabled()
    torch.utils.deterministic.fill_uninitialized_memory = True
    assert bootstrap_training_determinism(["--train-iters", "2"]) is None
    assert not policy.is_determinism_configured()
    assert dict(os.environ) == before
    assert torch.are_deterministic_algorithms_enabled() == enabled
    assert torch.utils.deterministic.fill_uninitialized_memory is True
    argv = ["--deterministic-mode", "--train-iters", "2"]
    assert bootstrap_training_determinism(argv)["options"]["deterministic_mode"]
    assert torch.utils.deterministic.fill_uninitialized_memory is False
    assert argv == ["--deterministic-mode", "--train-iters", "2"]


def test_root_only_yaml_mode_is_rejected(policy, tmp_path):
    from megatron.determinism import bootstrap_training_determinism

    path = tmp_path / "config.yaml"
    path.write_text("deterministic_mode: true\n")
    with pytest.raises(ValueError, match="model_parallel or language_model"):
        bootstrap_training_determinism(["--yaml-cfg", str(path)])
    assert not policy.is_determinism_configured()


def _training_entrypoints():
    candidates = [
        *ROOT.glob("*.py"),
        *(ROOT / "examples").rglob("*.py"),
        *(ROOT / "tools").rglob("*.py"),
    ]
    return sorted(
        str(path.relative_to(ROOT))
        for path in candidates
        if "__main__" in (source := path.read_text())
        and any(name in source for name in ("initialize_megatron", "pretrain(", "finetune("))
    )


@pytest.mark.parametrize("entrypoint", _training_entrypoints())
def test_every_training_cli_bootstraps_before_gpu_imports(policy, monkeypatch, entrypoint):
    path = ROOT / entrypoint
    tree = ast.parse(path.read_text())
    bootstrap = next(
        (
            index
            for index, node in enumerate(tree.body)
            if isinstance(node, ast.If)
            and any(
                isinstance(child, ast.ImportFrom) and child.module == "megatron.determinism"
                for child in node.body
            )
        ),
        None,
    )
    assert bootstrap is not None, f"Missing early policy: {entrypoint}"
    for node in ast.walk(ast.Module(body=tree.body[:bootstrap], type_ignores=[])):
        if isinstance(node, ast.ImportFrom):
            assert not (node.module or "").startswith(
                ("megatron.core", "megatron.training", "transformer_engine")
            )
    monkeypatch.setattr(sys, "argv", [str(path), "--deterministic-mode"])
    monkeypatch.setattr(sys, "path", list(sys.path))
    prefix = ast.Module(body=tree.body[: bootstrap + 1], type_ignores=[])
    exec(compile(prefix, str(path), "exec"), {"__name__": "__main__", "__file__": str(path)})
    assert policy.is_determinism_configured()


def test_late_drift_error_identifies_environment_key(policy, monkeypatch):
    policy.configure_determinism({"deterministic_mode": True})
    monkeypatch.setattr(torch.cuda, "is_initialized", lambda: True)
    os.environ["CUDA_DEVICE_MAX_CONNECTIONS"] = "32"
    with pytest.raises(RuntimeError, match="CUDA_DEVICE_MAX_CONNECTIONS"):
        policy.configure_determinism({"deterministic_mode": True})


def test_installed_dataset_helpers_load_without_make(monkeypatch):
    path = ROOT / "megatron/core/datasets/utils.py"
    function = next(
        node
        for node in ast.parse(path.read_text()).body
        if isinstance(node, ast.FunctionDef) and node.name == "compile_helpers"
    )
    namespace = {"__file__": str(path), "__package__": "megatron.core.datasets"}
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(path), "exec"), namespace)
    monkeypatch.setattr(os.path, "isfile", lambda _: False)
    loaded = []
    monkeypatch.setattr(importlib, "import_module", lambda name: loaded.append(name))
    monkeypatch.setattr(
        subprocess, "run", lambda *args, **kwargs: pytest.fail("wheel must not run make")
    )
    namespace["compile_helpers"]()
    assert loaded == ["megatron.core.datasets.helpers_cpp"]


@pytest.mark.parametrize(
    "yaml_text,enabled",
    [
        ("deterministic_mode: false", False),
        ("model_parallel:\n  deterministic_mode: true", True),
        ("deterministic_mode: true\nmodel_parallel:\n  deterministic_mode: false", False),
        (
            "model_parallel:\n  deterministic_mode: false\nlanguage_model:\n  deterministic_mode: true",
            True,
        ),
    ],
)
def test_yaml_bootstrap_replaces_cli_and_matches_config_precedence(
    policy, tmp_path, yaml_text, enabled
):
    from megatron.determinism import bootstrap_training_determinism

    path = tmp_path / "config.yaml"
    path.write_text(yaml_text)
    torch.utils.deterministic.fill_uninitialized_memory = True
    report = bootstrap_training_determinism(["--deterministic-mode", "--yaml-cfg", str(path)])
    assert (report is not None) is enabled
    assert policy.is_determinism_configured() is enabled
    assert torch.utils.deterministic.fill_uninitialized_memory is (not enabled)


@pytest.mark.parametrize("module_mode", [False, True])
def test_launcher_preserves_arguments_and_target_imports(tmp_path, module_mode):
    target = tmp_path / "startup_target.py"
    (tmp_path / "startup_sibling.py").write_text("VALUE = 17\n")
    target.write_text('''
import json
import sys
from pathlib import Path
import startup_sibling
from megatron.determinism import is_determinism_configured
assert is_determinism_configured()
assert 'megatron.core' not in sys.modules
assert startup_sibling.VALUE == 17
Path(sys.argv[1]).write_text(json.dumps(sys.argv))
''')
    report_path = tmp_path / "argv.json"
    env = dict(clean_environment(), PYTHONPATH=os.pathsep.join([str(ROOT), str(tmp_path)]))
    command = [sys.executable, "-m", "megatron.determinism"]
    command += ["-m", "startup_target"] if module_mode else [str(target)]
    result = subprocess.run(
        [*command, str(report_path), "--literal", "a b"],
        cwd=tmp_path,
        env=env,
        text=True,
        capture_output=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert json.loads(report_path.read_text()) == [
        str(target),
        str(report_path),
        "--literal",
        "a b",
    ]
