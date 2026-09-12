# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""CPU tests of recipe evidence matching and opt-in capture adapters."""

import copy
import json
import sys
from types import ModuleType, SimpleNamespace

import pytest

from tools.determinism import capture_recipe
from tools.determinism.capture_recipe import Inventory, install_bindings
from tools.determinism.recipe_coverage import (
    DETERMINISTIC,
    NONDETERMINISTIC,
    UNVERIFIED,
    build_report,
    main,
    signature_key,
)


def signature(phase="forward"):
    return {
        "op_id": "swiglu",
        "implementation": "torch.compile:bias_swiglu",
        "phase": phase,
        "deterministic_algorithms": True,
        "runtime": {},
        "inputs": [
            {"shape": [2, 4], "stride": [4, 1], "dtype": "torch.bfloat16", "requires_grad": True}
        ],
    }


def inventory(rank=0, world_size=1):
    return {
        "schema_version": 1,
        "kind": "determinism_inventory",
        "recipe_id": "dense-test",
        "rank": rank,
        "complete": True,
        "truncated": False,
        "context": {"revision": "a" * 40, "dirty": False, "world_size": world_size, "gpu": "H100"},
        "operations": [{"signature": signature(), "calls": 3, "site": "example:swiglu"}],
    }


def evidence(status=DETERMINISTIC, phase="forward_backward"):
    return {
        "schema_version": 1,
        "kind": "determinism_coverage",
        "context": inventory()["context"],
        "run_id": "run-123",
        "ranks_present": [0],
        "cases": [
            {
                "case_id": "test_swiglu[bf16]",
                "status": status,
                "observations": [
                    {
                        "status": status,
                        "rank": 0,
                        "signature": signature(phase),
                        "protocol": {"replays": 3},
                    }
                ],
            }
        ],
    }


def test_matching_forward_evidence_preserves_recipe_replay_boundary():
    report = build_report([inventory()], [evidence()])
    assert report["counts"][DETERMINISTIC] == 1
    assert report["recipe_status"] == "replay_required"
    assert report["operations"][0]["evidence"][0]["case_id"] == "test_swiglu[bf16]"


@pytest.mark.parametrize(
    "dimension",
    ["revision", "gpu", "dtype", "shape", "stride", "mode", "runtime", "implementation", "op_id"],
)
def test_unverified_configuration_is_not_inferred_from_nearby_evidence(dimension):
    request = inventory()
    if dimension in ("revision", "gpu"):
        request["context"][dimension] = "different"
    elif dimension == "dtype":
        request["operations"][0]["signature"]["inputs"][0]["dtype"] = "torch.float32"
    elif dimension in ("shape", "stride"):
        request["operations"][0]["signature"]["inputs"][0][dimension] = [4, 2]
    elif dimension == "mode":
        request["operations"][0]["signature"]["deterministic_algorithms"] = False
    elif dimension == "runtime":
        request["operations"][0]["signature"]["runtime"] = {"autocast": True}
    else:
        request["operations"][0]["signature"][dimension] = "different"
    assert build_report([request], [evidence()])["counts"][UNVERIFIED] == 1


def test_passing_evidence_does_not_erase_a_matching_mismatch():
    report = build_report([inventory()], [evidence(), evidence(NONDETERMINISTIC, "forward")])
    assert report["counts"][NONDETERMINISTIC] == 1
    assert report["recipe_status"] == "known_nondeterministic_operation"


def test_backward_failure_does_not_claim_forward_failure():
    report = build_report([inventory()], [evidence(NONDETERMINISTIC)])
    assert report["counts"][UNVERIFIED] == 1
    request = inventory()
    request["operations"][0]["signature"]["phase"] = "forward_backward"
    assert build_report([request], [evidence(NONDETERMINISTIC)])["counts"][NONDETERMINISTIC] == 1


@pytest.mark.parametrize("problem", ["missing_rank", "incomplete", "truncated", "dirty"])
def test_capture_gaps_cannot_produce_verified_coverage(problem):
    request = inventory()
    proof = evidence()
    if problem == "missing_rank":
        request["context"]["world_size"] = 2
        proof["context"]["world_size"] = 2
    elif problem == "dirty":
        request["context"]["dirty"] = True
    else:
        request["complete" if problem == "incomplete" else "truncated"] = problem != "incomplete"
    report = build_report([request], [proof])
    assert report["counts"][UNVERIFIED] == 1
    assert report["capture_issues"]


def test_repeated_calls_and_ranks_do_not_inflate_configuration_count():
    requests = [inventory(0, 2), inventory(1, 2)]
    proof = evidence()
    proof["context"]["world_size"] = 2
    proof["ranks_present"] = [0, 1]
    observations = proof["cases"][0]["observations"]
    observations.append({**copy.deepcopy(observations[0]), "rank": 1})
    report = build_report(requests, [proof])
    assert report["counts"]["total"] == 1
    assert report["counts"][DETERMINISTIC] == 1
    assert report["operations"][0]["calls"] == 6
    assert report["operations"][0]["ranks"] == [0, 1]
    with pytest.raises(ValueError, match="Duplicate"):
        build_report([inventory(), inventory()], [])


def test_signature_evidence_requires_every_rank():
    requests = [inventory(0, 2), inventory(1, 2)]
    proof = evidence()
    proof["context"]["world_size"] = 2
    proof["ranks_present"] = [0, 1]
    other = copy.deepcopy(proof["cases"][0]["observations"][0])
    other["rank"] = 1
    other["signature"]["inputs"][0]["shape"] = [4, 2]
    proof["cases"][0]["observations"].append(other)
    assert build_report(requests, [proof])["counts"][UNVERIFIED] == 1


def test_empty_inventory_has_no_percentage():
    request = inventory()
    request["operations"] = []
    assert build_report([request], [evidence()])["deterministic_percent"] is None


class Tensor:
    shape = (2, 4)
    dtype = "torch.bfloat16"
    requires_grad = True

    def __init__(self):
        self.hooks = []

    def stride(self):
        return (4, 1)

    def register_hook(self, hook):
        self.hooks.append(hook)


def test_capture_preserves_result_records_backward_and_restores_binding(monkeypatch):
    monkeypatch.setattr(capture_recipe, "runtime_signature", lambda torch: {})
    torch = SimpleNamespace(Tensor=Tensor, are_deterministic_algorithms_enabled=lambda: True)
    recorder = Inventory(torch, 10)
    module = ModuleType("fixture_kernel")
    original = lambda value: value
    module.fn = original
    monkeypatch.setitem(sys.modules, module.__name__, module)
    bindings = [
        {
            "target": "fixture_kernel:fn",
            "op_id": "swiglu",
            "implementation": "torch.compile:bias_swiglu",
        }
    ]
    tensor = Tensor()
    with install_bindings(recorder, bindings):
        assert module.fn(tensor) is tensor
        assert len(recorder.operations) == 1
        assert tensor.hooks[0](tensor) is tensor
        assert len(recorder.operations) == 2
    assert module.fn is original
    assert {item["signature"]["phase"] for item in recorder.operations.values()} == {
        "forward",
        "forward_backward",
    }
    assert signature_key(signature()) in recorder.operations


def test_signature_limit_is_visible():
    torch = SimpleNamespace(Tensor=Tensor, are_deterministic_algorithms_enabled=lambda: True)
    recorder = Inventory(torch, 1)
    recorder.record(signature(), "site")
    recorder.record(signature("forward_backward"), "site")
    assert recorder.truncated
    assert len(recorder.operations) == 1


def test_cli_strict_unknown_and_matching_failure(tmp_path):
    request_dir = tmp_path / "inventory"
    request_dir.mkdir()
    (request_dir / "rank-0.json").write_text(json.dumps(inventory()))
    proof_path = tmp_path / "evidence.json"
    proof_path.write_text(json.dumps(evidence(NONDETERMINISTIC, "forward")))
    output = tmp_path / "report.json"
    args = [str(request_dir), "--evidence", str(proof_path), "--output", str(output), "--strict"]
    assert main(args) == 1
    proof = evidence()
    proof["context"]["revision"] = "b" * 40
    proof_path.write_text(json.dumps(proof))
    assert main(args) == 2
    assert output.with_suffix(".md").is_file()


@pytest.mark.parametrize("exit_code,complete", [(0, True), (1, False)])
def test_capture_entrypoint_preserves_exit_and_incomplete_evidence(
    tmp_path, monkeypatch, exit_code, complete
):
    torch = SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False))
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setattr(capture_recipe, "source_context", lambda torch: inventory()["context"])
    monkeypatch.setenv("RANK", "0")
    bindings = tmp_path / "bindings.json"
    bindings.write_text("[]")
    (tmp_path / "helper.py").write_text(f"EXIT_CODE = {exit_code}\n")
    script = tmp_path / "train.py"
    script.write_text("from helper import EXIT_CODE\nraise SystemExit(EXIT_CODE)\n")
    monkeypatch.delitem(sys.modules, "helper", raising=False)
    output = tmp_path / "capture"
    args = [
        "--bindings",
        str(bindings),
        "--output",
        str(output),
        "--recipe-id",
        "test",
        "--",
        str(script),
    ]
    previous_argv, previous_path = sys.argv[:], sys.path[:]
    if complete:
        assert capture_recipe.main(args) == 0
    else:
        with pytest.raises(SystemExit) as error:
            capture_recipe.main(args)
        assert error.value.code == exit_code
    assert json.loads((output / "rank-0.json").read_text())["complete"] == complete
    assert sys.argv == previous_argv
    assert sys.path == previous_path
