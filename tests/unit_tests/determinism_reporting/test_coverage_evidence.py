# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""CPU-only contract tests; run with --confcutdir at this directory."""

import json
from pathlib import Path

import pytest

from tools.determinism.coverage import (
    DETERMINISTIC,
    NONDETERMINISTIC,
    UNVERIFIED,
    ReplayMismatch,
    aggregate,
    collect_observations,
    main,
    observe_replay,
    triton_signature,
)

pytest_plugins = ["pytester"]


def shard(rank=0, world_size=1, status=DETERMINISTIC):
    return {
        "schema_version": 1,
        "run_id": "ci-run-123",
        "rank": rank,
        "complete": True,
        "context": {"revision": "a" * 40, "dirty": False, "world_size": world_size},
        "inventory": {"tested": {}, "missing": {"exempt_reason": "Dependency absent"}},
        "cases": {
            "test_op[bf16]": {
                "declaration": {"op_id": "tested", "implementation": "torch:example"},
                "test_complete": True,
                "observations": [{"status": status, "signature": {}, "protocol": {}}],
            }
        },
    }


def test_complete_rank_coverage_and_inventory_debt():
    report = aggregate([shard(0, 2), shard(1, 2)])
    assert report["counts"] == {"total": 1, DETERMINISTIC: 1, NONDETERMINISTIC: 0, UNVERIFIED: 0}
    assert report["deterministic_percent"] == 100
    assert report["inventory_without_declared_cases"] == {
        "missing": {"exempt_reason": "Dependency absent"}
    }


@pytest.mark.parametrize(
    "failure", ["rank", "session", "test", "observation", "case", "dirty", "stale"]
)
def test_incomplete_or_stale_evidence_cannot_turn_green(failure):
    rows = [shard(0, 2), shard(1, 2)]
    revision = "a" * 40
    if failure == "rank":
        rows.pop()
    elif failure == "session":
        rows[1]["complete"] = False
    elif failure == "test":
        rows[1]["cases"]["test_op[bf16]"]["test_complete"] = False
    elif failure == "observation":
        rows[1]["cases"]["test_op[bf16]"]["observations"] = []
    elif failure == "case":
        rows[1]["cases"] = {}
    elif failure == "dirty":
        for row in rows:
            row["context"]["dirty"] = True
    elif failure == "stale":
        revision = "b" * 40
    assert aggregate(rows, revision)["cases"][0]["status"] == UNVERIFIED


def test_observed_mismatch_survives_xfail_and_missing_peer():
    row = shard(0, 2, NONDETERMINISTIC)
    row["cases"]["test_op[bf16]"]["test_complete"] = False
    assert aggregate([row])["cases"][0]["status"] == NONDETERMINISTIC


@pytest.mark.parametrize("field,value", [("revision", "b" * 40), ("world_size", 8)])
def test_mixed_context_is_rejected(field, value):
    rows = [shard(0, 2), shard(1, 2)]
    rows[1]["context"][field] = value
    with pytest.raises(ValueError, match="Cannot combine"):
        aggregate(rows)


def test_duplicate_ranks_are_not_retries_or_additional_coverage():
    with pytest.raises(ValueError, match="Duplicate"):
        aggregate([shard(), shard()])


def test_triton_configuration_changes_invalidate_shared_evidence(monkeypatch):
    monkeypatch.setenv("TRITON_CACHE_AUTOTUNING", "1")
    monkeypatch.setenv("TRITON_CACHE_DIR", "/shared/cache")
    monkeypatch.setenv("TRITON_AUTOTUNE_BLOCK_SIZE_M", "64")
    rows = [shard(0, 2), shard(1, 2)]
    rows[0]["context"]["environment"] = triton_signature()
    assert rows[0]["context"]["environment"]["TRITON_CACHE_DIR"] == "/shared/cache"
    monkeypatch.setenv("TRITON_AUTOTUNE_BLOCK_SIZE_M", "128")
    rows[1]["context"]["environment"] = triton_signature()
    with pytest.raises(ValueError, match="Cannot combine"):
        aggregate(rows)


def test_empty_selection_has_no_percentage():
    row = shard()
    row["cases"] = {}
    assert aggregate([row])["deterministic_percent"] is None


@pytest.mark.parametrize(
    "error,status",
    [
        (ReplayMismatch("bytes differ"), NONDETERMINISTIC),
        (RuntimeError("CUDA unavailable"), UNVERIFIED),
        (AssertionError("wrong reference"), UNVERIFIED),
    ],
)
def test_only_typed_replay_mismatch_is_nondeterministic(error, status):
    observations = []
    with collect_observations(observations.append), pytest.raises(type(error)):
        with observe_replay({}, {"replays": 3}):
            raise error
    assert observations[0]["status"] == status


def test_success_requires_nonvacuous_observation_and_observer_is_restored():
    observations = []
    with collect_observations(observations.append):
        with observe_replay({}, {"replays": 3}):
            pass
        with observe_replay({}, {"replays": 3}) as observation:
            observation["compared_outputs"] = 1
    with observe_replay({}, {}):
        pass
    assert [item["status"] for item in observations] == [UNVERIFIED, DETERMINISTIC]


def test_forward_backward_requires_gradient_comparisons():
    observations = []
    with collect_observations(observations.append):
        with observe_replay({"phase": "forward_backward"}, {"replays": 3}) as observation:
            observation["compared_outputs"] = 1
        with observe_replay({"phase": "forward_backward"}, {"replays": 3}) as observation:
            observation.update(compared_outputs=1, compared_gradients=1)
    assert [item["status"] for item in observations] == [UNVERIFIED, DETERMINISTIC]


def test_cli_writes_matching_json_and_markdown(tmp_path):
    (tmp_path / "rank-0.json").write_text(json.dumps(shard()))
    target = tmp_path / "report.json"
    assert main([str(tmp_path), "--output", str(target)]) == 0
    assert json.loads(target.read_text())["deterministic_percent"] == 100
    assert "Verification coverage: 100.0%" in target.with_suffix(".md").read_text()


@pytest.mark.parametrize("failure", ["empty", "skipped", "missing_rank", "unrelated_failure"])
def test_ci_gate_requires_passing_comparisons_on_every_rank(tmp_path, failure):
    row = shard(world_size=2 if failure == "missing_rank" else 1)
    if failure == "empty":
        row["cases"] = {}
    elif failure == "skipped":
        row["cases"]["test_op[bf16]"]["observations"] = []
    elif failure == "unrelated_failure":
        row["cases"]["test_op[bf16]"]["test_complete"] = False
    (tmp_path / "rank-0.json").write_text(json.dumps(row))
    target = tmp_path / "report.json"
    assert main([str(tmp_path), "--output", str(target), "--require-verified"]) == 1
    assert target.exists()  # Preserve the unknowns even when the CI gate fails.


def test_required_blackwell_case_cannot_be_replaced_by_an_unrelated_pass(tmp_path):
    row = shard()
    row["evidence_scope"] = "model"
    (tmp_path / "rank-0.json").write_text(json.dumps(row))
    args = [str(tmp_path), "--output", str(tmp_path / "report.json"), "--require-verified"]
    assert main(args + ["--require-case", "*bf16*"]) == 0
    assert main(args + ["--require-case", "*mxfp8*"]) == 1
    row["cases"]["test_op[mxfp8]"] = {
        "declaration": {"op_id": "tested", "implementation": "test:quantized"},
        "test_complete": False,
        "observations": [],
    }
    (tmp_path / "rank-0.json").write_text(json.dumps(row))
    assert main(args) == 0  # The ordinary BF16 case still passed.
    assert main(args + ["--require-case", "*mxfp8*"]) == 1
    assert json.loads((tmp_path / "report.json").read_text())["kind"] == "model_determinism_replay"


def test_model_evidence_cannot_be_combined_with_kernel_coverage():
    rows = [shard(0, 2), shard(1, 2)]
    rows[1]["evidence_scope"] = "model"
    with pytest.raises(ValueError, match="kernel and model"):
        aggregate(rows)


def test_model_pytest_lifecycle_records_actual_comparisons_and_skip_reasons(pytester, monkeypatch):
    monkeypatch.setenv("PYTHONPATH", str(Path(__file__).resolve().parents[3]))
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
pytestmark = pytest.mark.determinism_model(model_id='gpt')

def test_good():
    t = torch.ones(1)
    assert_bit_exact(t, {'w': t}, t.clone(), {'w': t.clone()})

@pytest.mark.skip(reason='unsupported precision')
def test_skipped():
    pass

def test_without_comparison():
    pass
""")
    output = pytester.path / "evidence"
    result = pytester.runpytest_subprocess(
        "-p",
        "tools.determinism.pytest_plugin",
        "--determinism-evidence-dir",
        str(output),
        "--determinism-evidence-scope",
        "model",
    )
    result.assert_outcomes(passed=2, skipped=1)
    report = aggregate([json.loads(path.read_text()) for path in output.glob("rank-*.json")])
    assert report["kind"] == "model_determinism_replay"
    assert report["counts"] == {"total": 3, DETERMINISTIC: 1, NONDETERMINISTIC: 0, UNVERIFIED: 2}
    skipped = next(case for case in report["cases"] if case["case_id"].endswith("test_skipped"))
    assert "unsupported precision" in " ".join(skipped["reasons"])


def test_real_pytest_lifecycle_uses_replay_evidence(pytester, monkeypatch):
    root = Path(__file__).resolve().parents[3]
    monkeypatch.setenv("PYTHONPATH", str(root))
    monkeypatch.setenv("RANK", "0")
    pytester.makeini("[pytest]\n")
    manifest = pytester.path / "tests/unit_tests/determinism/kernels/manifest.py"
    manifest.parent.mkdir(parents=True)
    manifest.write_text(
        "from types import SimpleNamespace\nKERNELS = [SimpleNamespace(name='op', sources=(), exempt_reason='')]\n"
    )
    pytester.makeconftest("""
from tools.determinism import pytest_plugin
pytest_plugin._context = lambda root: {'revision': 'a' * 40, 'dirty': False, 'world_size': 1}
""")
    pytester.makepyfile("""
import pytest
from tools.determinism.coverage import observe_replay, ReplayMismatch
pytestmark = pytest.mark.determinism_case(op_id='op', implementation='test:op')

def test_good():
    with observe_replay({}, {'replays': 3}) as obs:
        obs['compared_outputs'] = 1

@pytest.mark.xfail(reason='known numerical mismatch')
def test_numerical_xfail():
    with observe_replay({}, {'replays': 3}):
        raise ReplayMismatch('different bytes')

@pytest.mark.xfail(reason='missing dependency')
def test_setup_xfail():
    raise RuntimeError('missing dependency')

def test_no_replay():
    pass

@pytest.fixture
def teardown_failure():
    yield
    raise RuntimeError('teardown failed')

def test_failed_teardown(teardown_failure):
    with observe_replay({}, {'replays': 3}) as obs:
        obs['compared_outputs'] = 1
""")
    output = pytester.path / "evidence"
    result = pytester.runpytest_subprocess(
        "-p", "tools.determinism.pytest_plugin", "--determinism-evidence-dir", str(output)
    )
    result.assert_outcomes(passed=3, xfailed=2, errors=1)
    rows = [json.loads(path.read_text()) for path in output.glob("rank-*.json")]
    report = aggregate(rows)
    states = {case["case_id"].split("::")[-1]: case["status"] for case in report["cases"]}
    assert states == {
        "test_good": DETERMINISTIC,
        "test_numerical_xfail": NONDETERMINISTIC,
        "test_setup_xfail": UNVERIFIED,
        "test_no_replay": UNVERIFIED,
        "test_failed_teardown": UNVERIFIED,
    }
