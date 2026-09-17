# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""CPU checks for numerical diagnostics, comparator controls and author gates."""

import ast
import copy
import json
from pathlib import Path

import pytest
import torch

from tests.performance_tests.shell_test_utils.determinism.kernel_case import kernel_policy
from tests.unit_tests.determinism.comparison import _as_bytes, bytes_equal
from tests.unit_tests.determinism_reporting.test_coverage_evidence import shard
from tools.check_kernel_determinism_coverage import load_manifest
from tools.determinism.checks import FAILED, PASSED, check_statuses, collect_checks, observe_check
from tools.determinism.coverage import (
    DETERMINISTIC,
    NONDETERMINISTIC,
    UNVERIFIED,
    ReplayMismatch,
    aggregate,
    collect_observations,
    main,
    observe_replay,
    replay_configuration,
)
from tools.determinism.reference import assert_reference_close, assert_replay_sensitivity

pytest_plugins = ["pytester"]
ROOT = Path(__file__).resolve().parents[3]
SIGNATURE = {"phase": "forward_backward", "implementation": "test:op"}


def reference_check(actual, expected, **kwargs):
    return assert_reference_close(
        actual, expected, signature=SIGNATURE, reference_id="eager-test", **kwargs
    )


def test_reference_reports_outputs_and_every_gradient_including_zero_references():
    expected = ({"out": torch.tensor([0.0, 1.0, 10.0])}, {"x": torch.ones(1), "w": torch.ones(1)})
    actual = copy.deepcopy(expected)
    actual[0]["out"] += torch.tensor([0.0005, 0.01, 0.1])
    check = reference_check(actual, expected, rtol=0.02, atol=0.001)
    assert check["status"] == PASSED
    assert (check["compared_outputs"], check["compared_gradients"]) == (1, 2)
    assert set(check["metrics"]) == {"output:out", "gradient:x", "gradient:w"}
    metric = check["metrics"]["output:out"]
    assert metric["zero_reference_elements"] == 1
    assert metric["max_absolute_error"] == pytest.approx(0.1, abs=1e-6)
    assert metric["max_relative_error"] == pytest.approx(0.01, abs=1e-6)


def test_reference_failure_retains_all_metrics_and_samples_a_tolerance_violation():
    expected = ({"out": torch.tensor([1000.0, 0.0])}, {"w": torch.ones(2)})
    actual = ({"out": torch.tensor([1001.0, 0.1])}, {"w": torch.tensor([1.0, 2.0])})
    checks, replays = [], []
    with collect_checks(checks.append), collect_observations(replays.append):
        with pytest.raises(AssertionError, match="gradient:w"):
            reference_check(actual, expected, rtol=0.01, atol=0.001)
    assert not replays  # Numerical correctness failures are not replay nondeterminism.
    assert checks[0]["status"] == FAILED
    metrics = checks[0]["metrics"]
    assert metrics["output:out"]["violations"] == 1
    assert metrics["output:out"]["sample"]["flat_index"] == 1
    assert metrics["output:out"]["max_absolute_error"] == 1
    assert metrics["gradient:w"]["violations"] == 1


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_equal_nonfinite_values_fail_with_json_safe_diagnostics(value):
    pair = ({"out": torch.tensor([value])}, {"w": torch.ones(1)})
    checks = []
    with collect_checks(checks.append), pytest.raises(AssertionError):
        reference_check(pair, pair, rtol=0, atol=0)
    metric = checks[0]["metrics"]["output:out"]
    assert metric["nonfinite_elements"] == 1
    assert metric["max_absolute_error"] is None
    json.dumps(checks, allow_nan=False)


def test_zero_references_and_overflow_have_explicit_json_safe_error_values():
    zeros = ({"out": torch.zeros(1)}, {"w": torch.ones(1)})
    assert (
        reference_check(zeros, zeros, rtol=0, atol=0)["metrics"]["output:out"]["max_relative_error"]
        is None
    )
    checks = []
    actual = ({"out": torch.tensor([1e308], dtype=torch.float64)}, {"w": torch.ones(1)})
    expected = ({"out": -actual[0]["out"]}, actual[1])
    with collect_checks(checks.append), pytest.raises(AssertionError):
        reference_check(actual, expected, rtol=0, atol=0)
    assert checks[0]["metrics"]["output:out"]["max_absolute_error"] == "inf"
    json.dumps(checks, allow_nan=False)


@pytest.mark.parametrize("problem", ["keys", "shape", "dtype", "gradients", "outputs"])
def test_missing_or_incompatible_reference_tensors_fail(problem):
    pair = ({"out": torch.ones(2)}, {"w": torch.ones(2)})
    expected = copy.deepcopy(pair)
    if problem == "keys":
        expected[1]["other"] = expected[1].pop("w")
    elif problem == "shape":
        expected[0]["out"] = torch.ones(1, 2)
    elif problem == "dtype":
        expected[0]["out"] = torch.ones(2, dtype=torch.bfloat16)
    elif problem == "gradients":
        pair[1].clear()
        expected[1].clear()
    else:
        pair[0].clear()
        expected[0].clear()
    with pytest.raises(AssertionError):
        reference_check(pair, expected, rtol=0, atol=0)


@pytest.mark.parametrize("tolerances", [{"rtol": -1, "atol": 0}, {"rtol": 0, "atol": float("nan")}])
def test_invalid_reference_tolerances_cannot_pass(tolerances):
    pair = ({"out": torch.ones(1)}, {"w": torch.ones(1)})
    with pytest.raises(ValueError, match="tolerances"):
        reference_check(pair, pair, **tolerances)


@pytest.mark.parametrize(
    "error,status", [(AssertionError("bad"), FAILED), (RuntimeError("unavailable"), UNVERIFIED)]
)
def test_observer_classifies_errors_and_restores_nested_sinks(error, status):
    outer, inner = [], []
    with collect_checks(outer.append):
        with collect_checks(inner.append), pytest.raises(type(error)):
            with observe_check("reference", SIGNATURE, {}):
                raise error
        with observe_check("reference", SIGNATURE, {}):
            pass
    with observe_check("reference", SIGNATURE, {}):
        pass
    assert [row["status"] for row in inner] == [status]
    assert [row["status"] for row in outer] == [UNVERIFIED]


def kernel_comparator():
    """Load the actual comparator bodies without the harness's GPU-only imports."""
    path = ROOT / "tests/unit_tests/determinism/kernels/harness.py"
    tree = ast.parse(path.read_text())
    tree.body = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name in ("_assert_replay_matches", "_describe_mismatch")
    ]
    namespace = {
        "torch": torch,
        "bytes_equal": bytes_equal,
        "_as_bytes": _as_bytes,
        "ReplayMismatch": ReplayMismatch,
    }
    exec(compile(tree, str(path), "exec"), namespace)
    return namespace["_assert_replay_matches"]


@pytest.mark.parametrize("different_build", [False, True])
def test_runtime_build_metadata_matches_replay_reference_and_sensitivity(different_build):
    """A real diagnostic build context must survive each evidence producer."""
    actual = ({"out": torch.tensor([1.0, 2.0])}, {"x": torch.tensor([3.0, 4.0])})
    compare = kernel_comparator()
    replays, checks = [], []
    configuration = {"te_extension_sha256": "a" * 64, "diagnostic_dependency_variant": "tested"}
    with collect_observations(replays.append), collect_checks(checks.append):
        with replay_configuration(configuration):
            with observe_replay(SIGNATURE, {"replays": 2}) as observation:
                compare(1, *actual, *copy.deepcopy(actual), "configured replay")
                observation.update(compared_outputs=1, compared_gradients=1)
        reference_configuration = {
            **configuration,
            "te_extension_sha256": ("b" if different_build else "a") * 64,
        }
        with replay_configuration(reference_configuration):
            reference_check(actual, copy.deepcopy(actual), rtol=0, atol=0)
            assert_replay_sensitivity(
                actual,
                lambda candidate: compare(1, *actual, *candidate, "configured control"),
                signature=SIGNATURE,
            )
    assert len(checks) == 2 and all(check["status"] == PASSED for check in checks)
    assert replays[0]["status"] == DETERMINISTIC
    expected = UNVERIFIED if different_build else PASSED
    assert check_statuses(
        [{**check, "rank": 0} for check in checks],
        [{**observation, "rank": 0} for observation in replays],
        complete=True,
        fresh=True,
    ) == {"reference": expected, "sensitivity": expected}
    assert "te_extension_sha256" not in SIGNATURE


@pytest.mark.parametrize("kind", ["reference", "sensitivity"])
def test_nested_runtime_configuration_restores_after_check_failure(kind):
    checks = []
    with collect_checks(checks.append):
        with replay_configuration({"backend_build": "outer", "parallelism": {"TP": 2}}):
            with pytest.raises(RuntimeError, match="unavailable"):
                with replay_configuration({"backend_build": "inner"}):
                    with observe_check(kind, SIGNATURE, {}):
                        raise RuntimeError("unavailable")
            with observe_check(kind, SIGNATURE, {}):
                pass
        with observe_check(kind, SIGNATURE, {}):
            pass
    assert checks[0]["signature"]["backend_build"] == "inner"
    assert checks[0]["signature"]["parallelism"] == {"TP": 2}
    assert checks[1]["signature"]["backend_build"] == "outer"
    assert "backend_build" not in checks[2]["signature"]
    assert "parallelism" not in checks[2]["signature"]
    assert all(check["status"] == UNVERIFIED for check in checks)


def test_sensitivity_exercises_actual_kernel_comparator_without_polluting_replay():
    actual = ({"out": torch.ones(2, 3).t()}, {"x": torch.tensor(0.0), "w": torch.ones(2)})
    saved = copy.deepcopy(actual)
    compare = kernel_comparator()
    replays = []
    with collect_observations(replays.append):
        check = assert_replay_sensitivity(
            actual,
            lambda candidate: compare(1, *actual, *candidate, "control"),
            signature=SIGNATURE,
        )
    assert check["status"] == PASSED
    assert check["detected_perturbations"] == 3
    assert not replays
    assert all(
        bytes_equal(t, saved[i][name])
        for i, tensors in enumerate(actual)
        for name, t in tensors.items()
    )


@pytest.mark.parametrize(
    "behavior,status",
    [("accept_all", FAILED), ("reject_all", FAILED), ("runtime_error", UNVERIFIED)],
)
def test_sensitivity_rejects_broken_comparators(behavior, status):
    actual = ({"out": torch.ones(1)}, {"x": torch.ones(1)})

    def compare(_):
        if behavior == "reject_all":
            raise ReplayMismatch("always rejects")
        if behavior == "runtime_error":
            raise RuntimeError("cannot compare")

    checks = []
    with collect_checks(checks.append), pytest.raises((AssertionError, RuntimeError)):
        assert_replay_sensitivity(actual, compare, signature=SIGNATURE)
    assert checks[0]["status"] == status


def author_shard(rank=0, world_size=1):
    row = shard(rank, world_size)
    row["inventory"]["tested"]["author_tests"] = ["test_op[bf16]"]
    case = row["cases"]["test_op[bf16]"]
    observation = case["observations"][0]
    observation.update(signature=copy.deepcopy(SIGNATURE), compared_outputs=1, compared_gradients=2)
    case["checks"] = [
        {
            "kind": kind,
            "status": PASSED,
            "signature": copy.deepcopy(SIGNATURE),
            "compared_outputs": 1,
            "compared_gradients": 2,
            "detected_perturbations": 3,
        }
        for kind in ("reference", "sensitivity")
    ]
    return row


def test_required_author_ids_match_the_gpu_parameter_matrix_and_gb200_selection():
    """Evaluate real declarations without importing production GPU dependencies."""
    path = ROOT / "tests/unit_tests/determinism/kernels/test_fused_activations.py"
    tree = ast.parse(path.read_text())
    op_ids = next(
        node
        for node in tree.body
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "GATED_OP_IDS" for target in node.targets
        )
    )
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "test_mlp_activation_author_evidence"
    )
    function.body = [ast.copy_location(ast.Pass(), function.body[0])]
    namespace = {"pytest": pytest, "torch": torch, "kernel_policy": kernel_policy}
    exec(
        compile(ast.Module(body=[op_ids, function], type_ignores=[]), str(path), "exec"), namespace
    )
    marks = namespace[function.name].pytestmark
    assert any(mark.name == "launch_on_gb200" for mark in marks)
    parameters = next(mark for mark in marks if mark.name == "parametrize").args[1]
    declared = {
        (
            parameter.marks[0].kwargs["op_id"],
            f"{path.relative_to(ROOT)}::{function.name}[{parameter.id}]",
        )
        for parameter in parameters
    }
    required = {
        (entry.name, node_id) for entry in load_manifest().KERNELS for node_id in entry.author_tests
    }
    assert required == declared
    assert len(required) == 6


def test_author_checks_require_all_ranks_and_matching_replay_counts():
    report = aggregate([author_shard(0, 2), author_shard(1, 2)])
    assert report["cases"][0]["check_status"] == {"reference": PASSED, "sensitivity": PASSED}
    assert report["author_requirements"][0]["status"] == PASSED


def test_one_check_cannot_cover_two_replay_calls_with_the_same_signature():
    row = author_shard()
    case = row["cases"]["test_op[bf16]"]
    case["observations"] *= 2
    assert aggregate([row])["author_requirements"][0]["status"] == UNVERIFIED
    case["checks"] *= 2
    assert aggregate([row])["author_requirements"][0]["status"] == PASSED


@pytest.mark.parametrize(
    "problem",
    [
        "rank",
        "session",
        "teardown",
        "check",
        "signature",
        "count",
        "empty",
        "extra",
        "dirty",
        "stale",
        "replay",
    ],
)
def test_incomplete_or_mismatched_checks_cannot_pass_the_author_gate(problem):
    rows = [author_shard(0, 2), author_shard(1, 2)]
    case = rows[1]["cases"]["test_op[bf16]"]
    revision = "a" * 40
    if problem == "rank":
        rows.pop()
    elif problem == "session":
        rows[1]["complete"] = False
    elif problem == "teardown":
        case["test_complete"] = False
    elif problem == "check":
        case["checks"].pop()
    elif problem == "signature":
        case["checks"][0]["signature"]["implementation"] = "different:op"
    elif problem == "count":
        case["checks"][0]["compared_gradients"] = 1
    elif problem == "empty":
        case["checks"][0]["compared_outputs"] = 0
    elif problem == "extra":
        extra = copy.deepcopy(case["checks"][0])
        extra.update(status=FAILED, signature={"implementation": "unmatched:op"})
        case["checks"].append(extra)
    elif problem == "dirty":
        for row in rows:
            row["context"]["dirty"] = True
    elif problem == "stale":
        revision = "b" * 40
    else:
        case["observations"][0]["status"] = NONDETERMINISTIC
    assert aggregate(rows, revision)["author_requirements"][0]["status"] == UNVERIFIED


def test_matching_failed_reference_survives_incomplete_execution_without_becoming_nondeterminism():
    row = author_shard(0, 2)
    case = row["cases"]["test_op[bf16]"]
    case["checks"][0]["status"] = FAILED
    case["test_complete"] = False
    report = aggregate([row])
    assert report["cases"][0]["status"] == UNVERIFIED
    assert report["cases"][0]["observations"][0]["status"] == DETERMINISTIC
    assert report["author_requirements"][0]["status"] == FAILED


@pytest.mark.parametrize("problem", [None, "missing_case", "wrong_op", "no_requirements", "failed"])
def test_author_cli_preserves_diagnostics_and_requires_exact_manifest_cases(tmp_path, problem):
    row = author_shard()
    if problem == "missing_case":
        row["inventory"]["tested"]["author_tests"].append("test_deleted[fp32]")
    elif problem == "wrong_op":
        row["cases"]["test_op[bf16]"]["declaration"]["op_id"] = "other"
    elif problem == "no_requirements":
        row["inventory"]["tested"].clear()
    elif problem == "failed":
        row["cases"]["test_op[bf16]"]["checks"][0]["status"] = FAILED
    (tmp_path / "rank-0.json").write_text(json.dumps(row))
    target = tmp_path / "report.json"
    assert main([str(tmp_path), "--output", str(target), "--require-author-checks"]) == int(
        problem is not None
    )
    assert target.exists() and target.with_suffix(".md").exists()


def test_pytest_records_author_checks_without_turning_reference_failures_into_replay_failures(
    pytester, monkeypatch
):
    monkeypatch.setenv("PYTHONPATH", str(ROOT))
    monkeypatch.setenv("RANK", "0")
    pytester.makeini("[pytest]\n")
    manifest = pytester.path / "tests/unit_tests/determinism/kernels/manifest.py"
    manifest.parent.mkdir(parents=True)
    manifest.write_text(
        "from types import SimpleNamespace\nKERNELS = [SimpleNamespace(name='op', sources=(), exempt_reason='', author_tests=())]\n"
    )
    pytester.makeconftest("""
from tools.determinism import pytest_plugin
pytest_plugin._context = lambda root: {'revision': 'a' * 40, 'dirty': False, 'world_size': 1}
""")
    pytester.makepyfile("""
import pytest
import torch
from tools.determinism.coverage import observe_replay, ReplayMismatch
from tools.determinism.reference import assert_reference_close, assert_replay_sensitivity
pytestmark = pytest.mark.determinism_case(op_id='op', implementation='test:op')

def exercise(wrong=False):
    signature = {'phase': 'forward_backward'}
    pair = ({'out': torch.ones(1)}, {'w': torch.ones(1)})
    with observe_replay(signature, {'replays': 3}) as observation:
        observation.update(compared_outputs=1, compared_gradients=1)
    expected = ({'out': torch.zeros(1)}, pair[1]) if wrong else pair
    assert_reference_close(pair, expected, signature=signature.copy(), reference_id='eager', rtol=0, atol=0)
    def compare(candidate):
        if any(not torch.equal(t.view(torch.uint8), candidate[i][name].view(torch.uint8))
               for i, values in enumerate(pair) for name, t in values.items()):
            raise ReplayMismatch('bytes changed')
    assert_replay_sensitivity(pair, compare, signature=signature.copy())

def test_good():
    exercise()

@pytest.mark.xfail(reason='known accuracy failure')
def test_wrong_reference():
    exercise(wrong=True)

@pytest.mark.skip(reason='missing hardware')
def test_skipped():
    exercise()

@pytest.fixture
def broken_teardown():
    yield
    raise RuntimeError('teardown failed')

def test_teardown(broken_teardown):
    exercise()
""")
    output = pytester.path / "evidence"
    result = pytester.runpytest_subprocess(
        "-p", "tools.determinism.pytest_plugin", "--determinism-evidence-dir", str(output)
    )
    result.assert_outcomes(passed=2, xfailed=1, skipped=1, errors=1)
    report = aggregate([json.loads(path.read_text()) for path in output.glob("rank-*.json")])
    cases = {case["case_id"].split("::")[-1]: case for case in report["cases"]}
    assert cases["test_good"]["check_status"] == {"reference": PASSED, "sensitivity": PASSED}
    assert cases["test_wrong_reference"]["check_status"]["reference"] == FAILED
    assert cases["test_wrong_reference"]["status"] == UNVERIFIED
    assert cases["test_wrong_reference"]["observations"][0]["status"] == DETERMINISTIC
    for name in ("test_skipped", "test_teardown"):
        assert cases[name]["check_status"] == {"reference": UNVERIFIED, "sensitivity": UNVERIFIED}
