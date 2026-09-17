# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Runtime/configuration contracts; CPU group doubles are not GPU validation."""

import ast
import copy
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Dict, Optional

import pytest
import torch

from tests.unit_tests.determinism.comparison import assert_bit_exact
from tests.unit_tests.determinism.configs import (
    GPT_PARALLELISM_CONFIGS,
    apply_parallelism,
    gb200_compatible_configs,
    moe_overrides,
    required_world_size,
)
from tools.determinism.coverage import (
    DETERMINISTIC,
    NONDETERMINISTIC,
    UNVERIFIED,
    aggregate,
    collect_observations,
    main,
    observe_replay,
    replay_configuration,
)
from tools.determinism.parallelism import normalize_parallelism

pytest_plugins = ["pytester"]
ROOT = Path(__file__).resolve().parents[3]


def matrix_shard(rank=0, world_size=1):
    plan = normalize_parallelism({"TP": 2, "CP": 2})
    case = {
        "declaration": {"op_id": "gpt", "implementation": "model:gpt"},
        "parallelism_plan": plan,
        "test_complete": True,
        "observations": [
            {"status": DETERMINISTIC, "signature": {"parallelism": plan}, "protocol": {}}
        ],
    }
    return {
        "schema_version": 1,
        "evidence_scope": "model",
        "run_id": "run",
        "rank": rank,
        "complete": True,
        "context": {"revision": "a" * 40, "dirty": False, "world_size": world_size},
        "cases": {"gpt-like": copy.deepcopy(case), "llama-like": copy.deepcopy(case)},
    }


@pytest.mark.parametrize(
    "failure", [None, "skip", "missing_runtime", "wrong_runtime", "missing_rank", "mismatch"]
)
def test_pairs_require_all_planned_model_variants_and_actual_configuration(failure):
    rows = [matrix_shard(0, 2), matrix_shard(1, 2)]
    case = rows[1]["cases"]["llama-like"]
    expected = DETERMINISTIC
    if failure == "skip":
        case["test_complete"] = False
        case["observations"] = []
    elif failure == "missing_runtime":
        case["observations"][0]["signature"] = {}
    elif failure == "wrong_runtime":
        case["observations"][0]["signature"] = {"parallelism": normalize_parallelism({"TP": 2})}
    elif failure == "missing_rank":
        rows.pop()
    elif failure == "mismatch":
        case["observations"][0]["status"] = NONDETERMINISTIC
        expected = NONDETERMINISTIC
    if failure and failure != "mismatch":
        expected = UNVERIFIED
    view = aggregate(rows)["parallelism"]
    assert view["counts"]["total"] == 15  # Six axes, one declared value per axis.
    assert view["counts"][expected] == 15
    assert all(pair["cases"] == ["gpt-like", "llama-like"] for pair in view["pairs"])
    assert (
        next(pair for pair in view["pairs"] if pair["values"] == {"TP": 2, "CP": 2})["status"]
        == expected
    )


def test_missing_plan_remains_visible_and_gate_cannot_infer_it_from_name(tmp_path):
    row = matrix_shard()
    for case in row["cases"].values():
        del case["parallelism_plan"]
    (tmp_path / "rank-0.json").write_text(json.dumps(row))
    assert (
        main([str(tmp_path), "--output", str(tmp_path / "report.json"), "--require-parallelism"])
        == 1
    )
    view = json.loads((tmp_path / "report.json").read_text())["parallelism"]
    assert view["unplanned_cases"] == ["gpt-like", "llama-like"]
    assert view["deterministic_percent"] is None


def test_different_plans_across_ranks_are_rejected():
    rows = [matrix_shard(0, 2), matrix_shard(1, 2)]
    rows[1]["cases"]["gpt-like"]["parallelism_plan"] = normalize_parallelism({"TP": 4})
    with pytest.raises(ValueError, match="Different parallelism plans"):
        aggregate(rows)


@pytest.mark.parametrize("plan", [{"typo": 2}, {"CP": 0}, {"TP": True}, {"EP": 2.0}, None])
def test_invalid_parallelism_is_not_silently_normalized(plan):
    with pytest.raises(ValueError):
        normalize_parallelism(plan)


def test_runtime_context_is_restored_between_cases():
    observations = []
    with collect_observations(observations.append):
        with replay_configuration({"parallelism": normalize_parallelism({"TP": 2})}):
            with replay_configuration({"fsdp": {"sharding_strategy": "optim_grads_params"}}):
                with observe_replay({}, {"replays": 2}) as obs:
                    obs["compared_outputs"] = 1
        with observe_replay({}, {"replays": 2}) as obs:
            obs["compared_outputs"] = 1
    assert "parallelism" in observations[0]["signature"]
    assert "fsdp" in observations[0]["signature"]
    assert "parallelism" not in observations[1]["signature"]
    assert "fsdp" not in observations[1]["signature"]


@pytest.mark.parametrize("strategy", [None, "no_shard", "optim_grads_params"])
def test_fsdp_pair_credit_requires_the_actual_sharding_policy(strategy):
    row = matrix_shard()
    for case in row["cases"].values():
        plan = normalize_parallelism({"FSDP": 4})
        case["parallelism_plan"] = plan
        signature = {"parallelism": plan}
        if strategy is not None:
            signature["fsdp"] = {"sharding_strategy": strategy}
        case["observations"][0]["signature"] = signature
    view = aggregate([row])["parallelism"]
    expected = DETERMINISTIC if strategy == "optim_grads_params" else UNVERIFIED
    assert view["counts"][expected] == 15


def test_pytest_plan_and_runtime_remain_independent(pytester, monkeypatch):
    monkeypatch.setenv("PYTHONPATH", str(ROOT))
    monkeypatch.setenv("RANK", "0")
    pytester.makeini("[pytest]\n")
    pytester.makeconftest("""
from tools.determinism import pytest_plugin
pytest_plugin._context = lambda root: {'revision': 'a' * 40, 'dirty': False, 'world_size': 1}
""")
    pytester.makepyfile("""
import pytest
import torch
from tests.unit_tests.determinism.comparison import assert_bit_exact
from tools.determinism.coverage import replay_configuration
from tools.determinism.parallelism import normalize_parallelism

def compare():
    t = torch.ones(2)
    with replay_configuration({'parallelism': normalize_parallelism({'TP': 2})}):
        assert_bit_exact(t, {'weight': t}, t.clone(), {'weight': t.clone()})

@pytest.mark.determinism_model(model_id='gpt')
@pytest.mark.parametrize('parallelism', [{'TP': 2}, {'TP': 4}])
def test_matrix(parallelism):
    compare()

@pytest.mark.determinism_model(model_id='quantized', parallelism={'TP': 2})
def test_fixed():
    compare()
""")
    output = pytester.path / "evidence"
    result = pytester.runpytest_subprocess(
        "-p",
        "tools.determinism.pytest_plugin",
        "--determinism-evidence-dir",
        str(output),
        "--determinism-evidence-scope=model",
    )
    result.assert_outcomes(passed=3)
    report = aggregate([json.loads(path.read_text()) for path in output.glob("rank-*.json")])
    assert report["counts"][DETERMINISTIC] == 3
    assert [row["status"] for row in report["parallelism"]["rows"]].count(DETERMINISTIC) == 2
    assert (
        main([str(output), "--output", str(pytester.path / "report.json"), "--require-parallelism"])
        == 0
    )


def definitions(path, names, namespace):
    """Execute repository definitions without importing the unavailable GPU stack."""
    tree = ast.parse((ROOT / path).read_text())
    nodes = [
        node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in names
    ]
    future = ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)
    exec(
        compile(
            ast.fix_missing_locations(ast.Module(body=[future, *nodes], type_ignores=[])),
            str(path),
            "exec",
        ),
        namespace,
    )
    return namespace


@pytest.mark.parametrize(
    "plan,runtime_changes",
    [
        ({"TP": 2, "CP": 2}, {}),
        ({"PP": 2, "VPP": 2}, {}),
        ({"EP": 2, "FSDP": 4}, {}),
        ({"TP": 2, "CP": 2}, {"CP": 1}),
    ],
)
def test_runner_uses_initialized_groups_for_configuration_and_evidence(plan, runtime_changes):
    actual = normalize_parallelism(plan) | runtime_changes
    calls = []
    getter_axes = {
        "tensor_model_parallel": "TP",
        "pipeline_model_parallel": "PP",
        "virtual_pipeline_model_parallel": "VPP",
        "context_parallel": "CP",
        "expert_model_parallel": "EP",
        "data_parallel": "FSDP",
    }
    groups = SimpleNamespace(
        **{
            f"get_{name}_world_size": (lambda axis=axis: actual[axis])
            for name, axis in getter_axes.items()
        }
    )
    init = []
    ns = definitions(
        "tests/unit_tests/determinism/bit_exact_runner.py",
        {"BitExactRunner"},
        {
            "torch": torch,
            "pytest": pytest,
            "Callable": Callable,
            "parallel_state": groups,
            "Utils": SimpleNamespace(
                world_size=required_world_size(plan),
                destroy_model_parallel=lambda: None,
                initialize_model_parallel=lambda **kwargs: init.append(kwargs),
            ),
            "model_parallel_cuda_manual_seed": lambda seed: None,
            "normalize_parallelism": normalize_parallelism,
            "required_world_size": required_world_size,
            "apply_parallelism": apply_parallelism,
            "moe_overrides": moe_overrides,
            "replay_configuration": replay_configuration,
        },
    )
    runner = ns["BitExactRunner"](lambda: None, lambda: {}, lambda: {}, supports_cp=True)

    def run(overrides, requested):
        calls.append(overrides)
        tensor = torch.ones(2)
        assert_bit_exact(tensor, {"w": tensor}, tensor.clone(), {"w": tensor.clone()})

    runner._run_naive = run
    runner._run_pipeline = run
    observations = []
    with collect_observations(observations.append):
        runner.run({}, plan)
    assert calls[0]["tensor_model_parallel_size"] == actual["TP"]
    assert calls[0]["pipeline_model_parallel_size"] == actual["PP"]
    assert calls[0]["context_parallel_size"] == actual["CP"]
    assert calls[0]["expert_model_parallel_size"] == actual["EP"]
    assert observations[0]["signature"]["parallelism"] == actual
    assert init[0].get("virtual_pipeline_model_parallel_size") == (
        actual["VPP"] if actual["VPP"] > 1 else None
    )


def test_gpt_cp_inputs_cover_sequence_once_and_keep_causal_query_mask(monkeypatch):
    group = object()
    rank = 0
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda g: 2 if g is group else 0)
    monkeypatch.setattr(torch.distributed, "get_rank", lambda g: rank if g is group else -1)
    ns = definitions(
        "megatron/core/utils.py",
        {"_get_batch_on_this_cp_rank_per_sequence_balancing", "get_batch_on_this_cp_rank"},
        {"torch": torch, "Any": Any, "Dict": Dict, "Optional": Optional, "Callable": Callable},
    )

    class CpuAllocations:
        def __getattr__(self, name):
            real = getattr(torch, name)
            if name not in {"randint", "arange", "ones"}:
                return real

            def allocate(*args, **kwargs):
                assert kwargs.pop("device") == "cuda"
                return real(*args, **kwargs)

            return allocate

    gpt = definitions(
        "tests/unit_tests/determinism/correctness/test_gpt_model.py",
        {"make_gpt_inputs"},
        {
            "torch": CpuAllocations(),
            "parallel_state": SimpleNamespace(get_context_parallel_group=lambda: group),
            "get_batch_on_this_cp_rank": ns["get_batch_on_this_cp_rank"],
            "MICRO_BATCH": 4,
            "SEQ_LEN": 32,
            "VOCAB_SIZE": 128,
        },
    )
    batches = []
    for rank in range(2):
        torch.manual_seed(123)
        batches.append(gpt["make_gpt_inputs"]())
    positions = torch.cat([batch["position_ids"][0] for batch in batches])
    assert sorted(positions.tolist()) == list(range(32))
    assert batches[0]["position_ids"][0].tolist() == list(range(8)) + list(range(24, 32))
    for batch in batches:
        assert batch["input_ids"].shape == (4, 16)
        assert batch["attention_mask"].shape == (4, 1, 16, 32)
        expected = torch.arange(32).unsqueeze(0) > batch["position_ids"][0].unsqueeze(1)
        assert torch.equal(batch["attention_mask"][0, 0], expected)


def test_gpt_cp_matrix_preserves_four_and_eight_gpu_selections():
    selected = gb200_compatible_configs(GPT_PARALLELISM_CONFIGS)
    cp = {param.id: param for param in selected if param.values[0].get("CP", 1) > 1}
    assert set(cp) == {"cp2", "tp2-cp2", "pp2-cp2", "tp2-pp2-cp2"}
    assert {
        name
        for name, param in cp.items()
        if any(mark.name == "launch_on_gb200" for mark in param.marks)
    } == {"cp2", "tp2-cp2", "pp2-cp2"}
    assert required_world_size(cp["tp2-pp2-cp2"].values[0]) == 8


@pytest.mark.parametrize("plan", [{"FSDP": 4, "PP": 2}, {"FSDP": 4, "CP": 2}])
def test_unimplemented_fsdp_combinations_skip_before_initialization(plan):
    ns = definitions(
        "tests/unit_tests/determinism/bit_exact_runner.py",
        {"BitExactRunner"},
        {
            "torch": torch,
            "pytest": pytest,
            "normalize_parallelism": normalize_parallelism,
            "required_world_size": required_world_size,
            "Utils": SimpleNamespace(world_size=8),
        },
    )
    runner = ns["BitExactRunner"](lambda: None, lambda: {}, lambda: {}, supports_cp=True)
    with pytest.raises(pytest.skip.Exception, match="Combined FSDP with PP/CP"):
        runner.run({}, plan)


def test_fsdp_gradient_capture_waits_for_reduce_and_reads_local_shards():
    ns = definitions("tests/unit_tests/determinism/utils.py", {"collect_grads"}, {"torch": torch})
    local = torch.tensor([1.0, -0.0])
    parameter = SimpleNamespace(grad=None)
    local_reads = []

    def to_local():
        local_reads.append(True)
        return local

    def finish_grad_sync():
        parameter.grad = SimpleNamespace(to_local=to_local)

    wrapped = SimpleNamespace(
        param_and_grad_buffer=SimpleNamespace(optimizer_named_parameters=[("weight", parameter)]),
        finish_grad_sync=finish_grad_sync,
    )
    snapshot = ns["collect_grads"]([wrapped])
    assert local_reads == [True]
    assert torch.equal(snapshot["chunk0.weight"].view(torch.uint8), local.view(torch.uint8))
    local.fill_(9)
    assert snapshot["chunk0.weight"].tolist() == [1.0, -0.0]


def test_fsdp_factory_uses_full_sharding_and_preserves_process_groups(monkeypatch):
    import sys
    from types import ModuleType

    # Keep production GPU imports isolated while executing the real factory.
    group_collection = object()
    modules = {
        "megatron.core.distributed": {"DistributedDataParallelConfig": SimpleNamespace},
        "megatron.core.process_groups_config": {
            "ProcessGroupCollection": SimpleNamespace(
                use_mpu_process_groups=lambda: group_collection
            )
        },
        "megatron.core.distributed.fsdp.mcore_fsdp_adapter": {
            "FullyShardedDataParallel": lambda **kwargs: SimpleNamespace(**kwargs)
        },
    }
    for name, attributes in modules.items():
        module = ModuleType(name)
        module.__dict__.update(attributes)
        monkeypatch.setitem(sys.modules, name, module)
    ns = definitions("tests/unit_tests/determinism/utils.py", {"maybe_fsdp_wrap"}, {"torch": torch})
    model = SimpleNamespace(config=object())
    wrapped = ns["maybe_fsdp_wrap"](model, {"FSDP": 4})
    assert wrapped.module is model
    assert wrapped.pg_collection is group_collection
    assert wrapped.ddp_config.data_parallel_sharding_strategy == "optim_grads_params"
    assert wrapped.ddp_config.use_megatron_fsdp
    assert wrapped.ddp_config.megatron_fsdp_version == 1
    assert wrapped.ddp_config.overlap_grad_reduce and wrapped.ddp_config.overlap_param_gather
    assert ns["maybe_fsdp_wrap"](model, {}) is model
