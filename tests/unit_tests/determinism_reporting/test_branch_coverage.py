# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Exercise real coverage contexts, pytest phases, and numerical comparisons on CPU."""

import json
import sys
from pathlib import Path

import pytest

import coverage
from tools.determinism.branch_coverage import BranchRecorder
from tools.determinism.coverage import aggregate, main

pytest_plugins = ["pytester"]


@pytest.fixture
def branch_project(pytester, monkeypatch):
    root = Path(__file__).resolve().parents[3]
    monkeypatch.setenv("PYTHONPATH", str(root))
    monkeypatch.setenv("RANK", "0")
    pytester.makeini("[pytest]\n")
    pytester.makeconftest("""
from tools.determinism import pytest_plugin
pytest_plugin._context = lambda root: {'revision': 'a' * 40, 'dirty': False, 'world_size': 1}
""")
    source = pytester.path / "source"
    source.mkdir()
    (source / "ops.py").write_text(
        "import torch\n"
        "def choose(flag):\n"
        "    if flag:\n"
        "        return torch.ones(2)\n"
        "    return torch.zeros(2)\n"
        "def failed_only(flag):\n"
        "    if flag:\n"
        "        return torch.ones(2)\n"
        "    return torch.zeros(2)\n"
    )
    (source / "unseen.py").write_text(
        "def uncalled(flag):\n    if flag:\n        return 1\n    return 0\n"
    )
    pytester.makepyfile("""
import pytest
from source.ops import choose, failed_only
from tests.unit_tests.determinism.comparison import assert_bit_exact

def compare(t, other):
    assert_bit_exact(t, {'weight': t}, other, {'weight': other})

@pytest.fixture
def setup_branch():
    choose(False)

@pytest.mark.determinism_model(model_id='cpu-pilot')
def test_good(setup_branch):
    compare(choose(True), choose(True))

def test_unmarked():
    choose(False)

@pytest.mark.determinism_model(model_id='cpu-pilot')
def test_without_comparison():
    choose(False)

@pytest.fixture
def bad_teardown():
    yield
    raise RuntimeError('teardown failed')

@pytest.mark.determinism_model(model_id='cpu-pilot')
def test_incomplete(bad_teardown):
    compare(failed_only(True), failed_only(True))

@pytest.mark.determinism_model(model_id='cpu-pilot')
@pytest.mark.skip(reason='unsupported')
def test_skipped():
    pass
""")
    output = pytester.path / "evidence"
    args = [
        "-p",
        "tools.determinism.pytest_plugin",
        "--determinism-evidence-scope=model",
        "--determinism-evidence-dir",
        str(output),
        "--determinism-branch-coverage",
        "--determinism-branch-source=source",
    ]
    return pytester, output, args


@pytest.mark.parametrize("outer", [False, True])
def test_branch_contexts_require_completed_numerical_replay(branch_project, outer):
    pytester, output, args = branch_project
    if outer:
        result = pytester.run(
            sys.executable,
            "-m",
            "coverage",
            "run",
            "--branch",
            "--context=ci",
            "-m",
            "pytest",
            *args,
        )
    else:
        result = pytester.runpytest_subprocess(*args)
    result.assert_outcomes(passed=4, skipped=1, errors=1)
    shards = [json.loads(path.read_text()) for path in output.glob("rank-*.json")]
    assert len(shards) == 1  # Raw coverage JSON must not be mistaken for evidence shards.
    report = aggregate(shards)
    branches = report["branches"]
    assert branches["complete"]
    assert branches["counts"] == {
        "total": 6,
        "observed": 3,
        "passing_replay": 1,
        "uncovered_by_passing_replay": 5,
        "never_observed": 3,
    }
    arcs = {tuple(arc["arc"]): arc for arc in branches["files"]["source/ops.py"]["arcs"]}
    assert arcs[(3, 4)]["passing_replay"]
    assert not arcs[(3, 5)]["passing_replay"]
    assert not arcs[(7, 8)]["passing_replay"]
    assert all(
        "test_unmarked" not in support["case_id"]
        for arc in arcs.values()
        for support in arc["observed_by"]
    )
    assert all(not arc["observed_by"] for arc in branches["files"]["source/unseen.py"]["arcs"])
    assert (
        main([str(output), "--output", str(pytester.path / "report.json"), "--require-branches"])
        == 0
    )
    if outer:
        # The existing CI collector remains alive and retains ordinary unmarked coverage.
        collector = coverage.Coverage(data_file=str(pytester.path / ".coverage"), config_file=False)
        collector.load()
        data = collector.get_data()
        data.set_query_context("ci")
        assert (3, 5) in data.arcs(str(pytester.path / "source/ops.py"))


def test_all_skipped_still_publishes_unexecuted_sources(branch_project):
    pytester, output, args = branch_project
    result = pytester.runpytest_subprocess(*args, "-k", "test_skipped")
    result.assert_outcomes(skipped=1, deselected=4)
    report = aggregate([json.loads(path.read_text()) for path in output.glob("rank-*.json")])
    assert report["branches"]["complete"]
    assert report["branches"]["counts"]["total"] == 6
    assert report["branches"]["counts"]["passing_replay"] == 0
    assert (
        main([str(output), "--output", str(pytester.path / "report.json"), "--require-branches"])
        == 1
    )


def test_statement_only_outer_collector_is_rejected(branch_project):
    pytester, _, args = branch_project
    result = pytester.run(sys.executable, "-m", "coverage", "run", "-m", "pytest", *args)
    assert result.ret != 0
    assert "require coverage run --branch" in result.stderr.str()


def branch_shard(rank=0, world_size=1):
    return {
        "schema_version": 1,
        "run_id": "run",
        "rank": rank,
        "complete": True,
        "context": {"revision": "a" * 40, "dirty": False, "world_size": world_size},
        "cases": {
            "case": {
                "declaration": {"op_id": "op", "implementation": "test"},
                "test_complete": True,
                "observations": [
                    {"status": "verified_deterministic", "signature": {}, "protocol": {}}
                ],
                "branch_arcs": {"op.py": [[1, 2]]} if rank == 0 else {},
            }
        },
        "branches": {
            "complete": True,
            "coverage_version": "7.15.4",
            "files": {"op.py": {"sha256": "b" * 64, "arcs": [[1, 2], [1, 3]]}},
        },
    }


def test_rank_specific_branch_requires_global_replay_completion():
    rows = [branch_shard(0, 2), branch_shard(1, 2)]
    view = aggregate(rows)["branches"]
    assert view["counts"]["passing_replay"] == 1
    assert view["files"]["op.py"]["arcs"][0]["observed_by"][0]["ranks"] == [0]
    assert aggregate(rows[:1])["branches"]["passing_replay_percent"] is None
    rows[1]["cases"]["case"]["test_complete"] = False
    assert aggregate(rows)["branches"]["counts"]["passing_replay"] == 0


@pytest.mark.parametrize("failure", ["dirty", "stale", "missing_collector", "interrupted"])
def test_branch_evidence_cannot_hide_missing_provenance(failure):
    rows = [branch_shard(0, 2), branch_shard(1, 2)]
    revision = "a" * 40
    if failure == "dirty":
        for row in rows:
            row["context"]["dirty"] = True
    elif failure == "stale":
        revision = "c" * 40
    elif failure == "missing_collector":
        del rows[1]["branches"]
    else:
        rows[1]["complete"] = False
    view = aggregate(rows, revision)["branches"]
    assert not view["complete"]
    assert view["counts"]["passing_replay"] == 0
    assert view["passing_replay_percent"] is None


@pytest.mark.parametrize("field,value", [("sha256", "c" * 64), ("arcs", [[1, 2]])])
def test_different_branch_denominators_are_rejected(field, value):
    rows = [branch_shard(0, 2), branch_shard(1, 2)]
    rows[1]["branches"]["files"]["op.py"][field] = value
    with pytest.raises(ValueError, match="catalogs"):
        aggregate(rows)


def test_source_changes_during_collection_are_rejected(tmp_path):
    source = tmp_path / "op.py"
    source.write_text("def f(x):\n    if x:\n        return 1\n    return 0\n")
    recorder = BranchRecorder(tmp_path, ["op.py"], tmp_path / "raw.json")
    source.write_text(source.read_text() + "# changed\n")
    with pytest.raises(ValueError, match="changed"):
        recorder.finish()
