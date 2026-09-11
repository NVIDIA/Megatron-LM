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
