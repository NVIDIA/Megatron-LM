# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""CPU tests of recipe evidence matching and opt-in capture adapters."""

import copy
import json
import os
import subprocess
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import torch
import torch.utils.deterministic

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
        "runtime": {"fill_uninitialized_memory": True},
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


@pytest.fixture
def memory_fill():
    previous = torch.utils.deterministic.fill_uninitialized_memory
    yield
    torch.utils.deterministic.fill_uninitialized_memory = previous


@pytest.mark.parametrize("fill", [False, True])
def test_capture_runtime_records_memory_fill_without_mutation(memory_fill, fill):
    torch.utils.deterministic.fill_uninitialized_memory = fill
    before = capture_recipe.runtime_signature(torch)
    assert before["fill_uninitialized_memory"] is fill
    assert torch.utils.deterministic.fill_uninitialized_memory is fill
    torch.utils.deterministic.fill_uninitialized_memory = not fill
    after = capture_recipe.runtime_signature(torch)
    assert after["fill_uninitialized_memory"] is (not fill)
    assert before != after


@pytest.mark.parametrize("fill", [False, True])
@pytest.mark.parametrize("status", [DETERMINISTIC, NONDETERMINISTIC])
def test_memory_fill_match_requires_the_actual_boolean_setting(fill, status):
    request, proof = inventory(), evidence(status, "forward")
    request["operations"][0]["signature"]["runtime"]["fill_uninitialized_memory"] = fill
    proof["cases"][0]["observations"][0]["signature"]["runtime"]["fill_uninitialized_memory"] = fill
    assert build_report([request], [proof])["counts"][status] == 1
    proof["cases"][0]["observations"][0]["signature"]["runtime"][
        "fill_uninitialized_memory"
    ] = not fill
    assert build_report([request], [proof])["counts"][UNVERIFIED] == 1


@pytest.mark.parametrize("value", [None, "false", 0, 1])
@pytest.mark.parametrize("status", [DETERMINISTIC, NONDETERMINISTIC])
def test_legacy_or_non_boolean_memory_fill_cannot_supply_recipe_evidence(value, status):
    request, proof = inventory(), evidence(status, "forward")
    for item in (request["operations"][0], proof["cases"][0]["observations"][0]):
        if value is None:
            item["signature"]["runtime"].pop("fill_uninitialized_memory")
        else:
            item["signature"]["runtime"]["fill_uninitialized_memory"] = value
    report = build_report([request], [proof])
    assert report["counts"][UNVERIFIED] == 1
    assert not report["operations"][0]["evidence"]
    assert "memory-fill" in report["operations"][0]["reason"].lower()


@pytest.mark.parametrize("missing_from", ["inventory", "evidence", "backward_runtime"])
def test_memory_fill_missing_on_either_side_cannot_match(missing_from):
    request, proof = inventory(), evidence()
    if missing_from == "inventory":
        request["operations"][0]["signature"]["runtime"].pop("fill_uninitialized_memory")
    elif missing_from == "evidence":
        proof["cases"][0]["observations"][0]["signature"]["runtime"].pop(
            "fill_uninitialized_memory"
        )
    else:
        request["operations"][0]["signature"]["backward_runtime"] = {}
        proof["cases"][0]["observations"][0]["signature"]["backward_runtime"] = {}
    assert build_report([request], [proof])["counts"][UNVERIFIED] == 1


def test_backward_policy_change_is_not_relabelled_as_forward_policy(memory_fill):
    torch.utils.deterministic.fill_uninitialized_memory = False
    recorder = Inventory(torch, 10)
    binding = {"target": "fixture:square", "op_id": "square", "implementation": "torch:square"}
    wrapped = recorder.wrap(lambda x: x.square(), binding)
    value = torch.ones(2, requires_grad=True)
    output = wrapped(value)
    torch.utils.deterministic.fill_uninitialized_memory = True
    output.sum().backward(retain_graph=True)
    rows = list(recorder.operations.values())
    backward = next(row for row in rows if row["signature"]["phase"] == "forward_backward")
    assert backward["signature"]["runtime"]["fill_uninitialized_memory"] is False
    assert backward["signature"]["backward_runtime"]["fill_uninitialized_memory"] is True
    # A second traversal under another policy must not disappear behind the
    # duplicate-output hook guard from the first traversal.
    torch.utils.deterministic.fill_uninitialized_memory = False
    output.sum().backward()
    rows = list(recorder.operations.values())
    assert len(rows) == 3
    assert sum("backward_runtime" not in row["signature"] for row in rows) == 2
    request, proof = inventory(), evidence()
    request["operations"] = rows
    proof["cases"][0]["op_id"] = "square"
    proof["cases"][0]["observations"][0]["signature"] = {
        **rows[0]["signature"],
        "phase": "forward_backward",
    }
    report = build_report([request], [proof])
    assert report["counts"][DETERMINISTIC] == 2
    assert report["counts"][UNVERIFIED] == 1


def test_backward_policy_deduplicates_output_hooks_without_changing_gradients(memory_fill):
    torch.utils.deterministic.fill_uninitialized_memory = False
    recorder = Inventory(torch, 10)
    binding = {"target": "fixture:outputs", "op_id": "outputs", "implementation": "torch:outputs"}
    value = torch.ones(2, requires_grad=True)
    outputs = recorder.wrap(lambda x: (x.square(), x + 1), binding)(value)
    sum(output.sum() for output in outputs).backward(retain_graph=True)
    assert torch.equal(value.grad, torch.full_like(value, 3))
    assert len(recorder.operations) == 2
    assert all(row["calls"] == 1 for row in recorder.operations.values())
    torch.utils.deterministic.fill_uninitialized_memory = True
    sum(output.sum() for output in outputs).backward()
    assert torch.equal(value.grad, torch.full_like(value, 6))
    assert len(recorder.operations) == 3
    assert all(row["calls"] == 1 for row in recorder.operations.values())


def test_model_output_gradient_report_is_not_operator_evidence():
    proof = evidence()
    proof["kind"] = "model_determinism_replay"
    with pytest.raises(ValueError, match="Unsupported coverage report"):
        build_report([inventory()], [proof])


def test_module_configuration_requires_an_explicit_matching_adapter():
    request, proof = inventory(), evidence()
    configuration = {"normalization": "RMSNorm", "eps": 1e-5}
    proof["cases"][0]["observations"][0]["signature"]["configuration"] = configuration
    assert build_report([request], [proof])["counts"][UNVERIFIED] == 1
    request["operations"][0]["signature"]["configuration"] = configuration
    assert build_report([request], [proof])["counts"][DETERMINISTIC] == 1
    request["operations"][0]["signature"]["configuration"] = {**configuration, "eps": 1e-6}
    assert build_report([request], [proof])["counts"][UNVERIFIED] == 1


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


@pytest.mark.parametrize(
    "key,value",
    [
        ("TRITON_CACHE_AUTOTUNING", "1"),
        ("TRITON_CACHE_DIR", "/other/cache"),
        ("TRITON_AUTOTUNE_BLOCK_SIZE_M", "128"),
    ],
)
def test_triton_changes_require_new_evidence(monkeypatch, key, value):
    monkeypatch.setenv("TRITON_CACHE_AUTOTUNING", "0")
    monkeypatch.setenv("TRITON_CACHE_DIR", "/shared/cache")
    monkeypatch.setenv("TRITON_AUTOTUNE_BLOCK_SIZE_M", "64")
    request, proof = inventory(), evidence()
    original = capture_recipe.triton_signature()
    request["context"]["environment"] = original
    proof["context"]["environment"] = original
    request["operations"][0]["signature"]["runtime"]["triton"] = original
    proof["cases"][0]["observations"][0]["signature"]["runtime"]["triton"] = original
    assert build_report([request], [proof])["counts"][DETERMINISTIC] == 1
    monkeypatch.setenv(key, value)
    # Even a call-time override after capture startup must stop evidence reuse.
    request["operations"][0]["signature"]["runtime"]["triton"] = capture_recipe.triton_signature()
    assert build_report([request], [proof])["counts"][UNVERIFIED] == 1


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


def ranked_case(phase="forward_backward"):
    """Synthetic collective-shaped metadata, not an executed GPU recipe."""
    requests = [inventory(rank, 4) for rank in range(4)]
    proof = evidence()
    proof["context"]["world_size"] = 4
    proof["ranks_present"] = list(range(4))
    prototype = proof["cases"][0]["observations"][0]
    observations = []
    for rank, request in enumerate(requests):
        observation = copy.deepcopy(prototype)
        observation["rank"] = rank
        recorded = observation["signature"]
        recorded.update(
            op_id="tensor_parallel_mappings",
            implementation="mcore:reduce_from_tensor_model_parallel_region",
            configuration={
                "collective": {
                    "group_ranks": [2 * (rank // 2), 2 * (rank // 2) + 1],
                    "group_rank": rank % 2,
                    "input_sha256": f"{rank:064x}",
                }
            },
        )
        request["operations"][0]["signature"] = {**copy.deepcopy(recorded), "phase": phase}
        observations.append(observation)
    proof["cases"][0]["observations"] = observations
    return requests, proof


@pytest.mark.parametrize("phase", ["forward", "forward_backward"])
def test_complete_rank_assignment_matches_without_generalizing_to_other_ranks(phase):
    requests, proof = ranked_case(phase)
    report = build_report(requests, [proof])
    assert report["counts"] == {"total": 4, DETERMINISTIC: 4, NONDETERMINISTIC: 0, UNVERIFIED: 0}
    assert report["recipe_status"] == "replay_required"
    expected = {
        str(request["rank"]): signature_key(request["operations"][0]["signature"])
        for request in requests
    }
    for operation in report["operations"]:
        match = operation["evidence"][0]
        assert match["ranks"] == operation["ranks"]
        assert match["rank_assignment"] == expected
        assert match["case_id"] == proof["cases"][0]["case_id"]


@pytest.mark.parametrize(
    "problem",
    [
        "swapped_ranks",
        "changed_peer_shape",
        "changed_group",
        "missing_observation",
        "missing_report_rank",
        "changed_protocol",
        "changed_phase",
        "ambiguous_rank",
    ],
)
def test_incomplete_or_different_rank_assignments_stay_unverified(problem):
    requests, proof = ranked_case()
    observations = proof["cases"][0]["observations"]
    if problem == "swapped_ranks":
        requests[0]["operations"], requests[1]["operations"] = (
            requests[1]["operations"],
            requests[0]["operations"],
        )
    elif problem == "changed_peer_shape":
        requests[3]["operations"][0]["signature"]["inputs"][0]["shape"] = [4, 2]
    elif problem == "changed_group":
        requests[3]["operations"][0]["signature"]["configuration"]["collective"]["group_ranks"] = [
            3,
            2,
        ]
    elif problem == "missing_observation":
        observations.pop()
    elif problem == "missing_report_rank":
        proof["ranks_present"].pop()
    elif problem == "changed_protocol":
        observations[3]["protocol"]["replays"] = 4
    elif problem == "changed_phase":
        observations[3]["signature"]["phase"] = "forward"
    else:
        additional = copy.deepcopy(observations[3])
        additional["signature"]["inputs"][0]["shape"] = [4, 2]
        observations.append(additional)
    report = build_report(requests, [proof])
    assert report["counts"][UNVERIFIED] == report["counts"]["total"] == 4


@pytest.mark.parametrize("separation", ["case", "run"])
def test_rank_assignments_cannot_be_assembled_from_separate_cases_or_runs(separation):
    requests, proof = ranked_case()
    other = copy.deepcopy(proof)
    proof["cases"][0]["observations"] = proof["cases"][0]["observations"][:2]
    other["cases"][0]["observations"] = other["cases"][0]["observations"][2:]
    if separation == "case":
        other["cases"][0]["case_id"] = "other-case"
        proof["cases"].extend(other["cases"])
        reports = [proof]
    else:
        other["run_id"] = "other-run"
        reports = [proof, other]
    assert build_report(requests, reports)["counts"][UNVERIFIED] == 4


def test_complete_assignment_cannot_lend_its_peers_to_another_protocol():
    requests, proof = ranked_case()
    observations = proof["cases"][0]["observations"]
    partial = copy.deepcopy(observations[0])
    for observation in observations:
        observation["protocol"]["replays"] = 4
    observations.append(partial)
    report = build_report(requests, [proof])
    assert report["counts"][DETERMINISTIC] == 4
    assert all(
        match["protocol"] == {"replays": 4}
        for operation in report["operations"]
        for match in operation["evidence"]
    )


def test_signature_observed_on_an_additional_rank_cannot_borrow_peer_evidence():
    requests, proof = ranked_case()
    requests[1]["operations"].append(copy.deepcopy(requests[0]["operations"][0]))
    report = build_report(requests, [proof])
    assert report["counts"][DETERMINISTIC] == 3 and report["counts"][UNVERIFIED] == 1
    unknown = next(row for row in report["operations"] if row["status"] == UNVERIFIED)
    assert unknown["ranks"] == [0, 1] and not unknown["evidence"]


def test_one_signature_can_match_a_subset_of_ranks_in_a_complete_assignment():
    requests, proof = ranked_case()
    for request, observation in zip(requests, proof["cases"][0]["observations"]):
        configuration = {"variant": request["rank"] % 2}
        request["operations"][0]["signature"]["configuration"] = configuration
        observation["signature"]["configuration"] = configuration
    report = build_report(requests, [proof])
    assert report["counts"][DETERMINISTIC] == 2
    assert {tuple(row["ranks"]) for row in report["operations"]} == {(0, 2), (1, 3)}


def test_rank_assignment_does_not_erase_matching_negative_evidence():
    requests, proof = ranked_case()
    failed = copy.deepcopy(proof)
    failed["run_id"] = "failing-run"
    failed["cases"][0]["status"] = NONDETERMINISTIC
    failed["cases"][0]["observations"][2]["status"] = NONDETERMINISTIC
    report = build_report(requests, [proof, failed])
    assert report["counts"][DETERMINISTIC] == 3 and report["counts"][NONDETERMINISTIC] == 1
    assert report["recipe_status"] == "known_nondeterministic_operation"


def test_rank_assignment_cli_writes_provenance_and_strict_unknown(tmp_path):
    requests, proof = ranked_case()
    directory = tmp_path / "inventory"
    directory.mkdir()
    for request in requests:
        (directory / f"rank-{request['rank']}.json").write_text(json.dumps(request))
    evidence_path = tmp_path / "evidence.json"
    evidence_path.write_text(json.dumps(proof))
    output = tmp_path / "report.json"
    args = [str(directory), "--evidence", str(evidence_path), "--output", str(output), "--strict"]
    assert main(args) == 0
    assert json.loads(output.read_text())["counts"][DETERMINISTIC] == 4
    requests[2]["operations"][0]["signature"]["configuration"]["collective"]["group_rank"] = 1
    (directory / "rank-2.json").write_text(json.dumps(requests[2]))
    assert main(args) == 2


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
    monkeypatch.setattr(
        capture_recipe, "runtime_signature", lambda torch: {"fill_uninitialized_memory": True}
    )
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
def test_capture_entrypoint_preserves_exit_and_incomplete_evidence(tmp_path, exit_code, complete):
    bindings = tmp_path / "bindings.json"
    bindings.write_text("[]")
    (tmp_path / "helper.py").write_text(f"EXIT_CODE = {exit_code}\n")
    script = tmp_path / "train.py"
    script.write_text("from helper import EXIT_CODE\nraise SystemExit(EXIT_CODE)\n")
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
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; from tools.determinism.capture_recipe import main\n"
            "previous_argv, previous_path = sys.argv[:], sys.path[:]\n"
            "try:\n"
            "    main(sys.argv[1:])\n"
            "finally:\n"
            "    assert sys.argv == previous_argv and sys.path == previous_path\n",
            *args,
        ],
        env={**os.environ, "CUDA_VISIBLE_DEVICES": "", "RANK": "0", "WORLD_SIZE": "1"},
        text=True,
        capture_output=True,
        timeout=60,
    )
    assert result.returncode == exit_code, result.stderr
    assert json.loads((output / "rank-0.json").read_text())["complete"] == complete


@pytest.mark.parametrize("mode", ["cli", "yaml", "yaml-overrides-cli"])
def test_capture_bootstraps_effective_policy_before_binding_import(tmp_path, mode):
    enabled = mode != "yaml-overrides-cli"
    module = tmp_path / "policy_probe.py"
    module.write_text(
        "import os, sys, torch\n"
        "assert 'megatron.core' not in sys.modules\n"
        "assert 'transformer_engine.pytorch' not in sys.modules\n"
        f"assert torch.are_deterministic_algorithms_enabled() is {enabled}\n"
        f"assert torch.backends.cudnn.deterministic is {enabled}\n"
        "assert not torch.cuda.is_initialized()\n"
        "assert not torch.is_deterministic_algorithms_warn_only_enabled()\n"
        + (
            "assert os.environ['TRITON_CACHE_AUTOTUNING'] == '0'\n"
            "assert os.environ['MAMBA_DETERMINISTIC'] == '1'\n"
            "assert os.environ['CAUSAL_CONV1D_DETERMINISTIC'] == '1'\n"
            if enabled
            else "assert 'TRITON_CACHE_AUTOTUNING' not in os.environ\n"
        )
        + "def activation(value):\n    return value.square()\n"
    )
    bindings = tmp_path / "bindings.json"
    bindings.write_text(
        json.dumps(
            [
                {
                    "target": "policy_probe:activation",
                    "op_id": "policy-probe",
                    "implementation": "torch:square",
                }
            ]
        )
    )
    script = tmp_path / "train.py"
    script.write_text(
        "import torch\nfrom policy_probe import activation\n"
        "activation(torch.ones(2, requires_grad=True)).sum().backward()\n"
    )
    command = [str(script)]
    if mode in ("cli", "yaml-overrides-cli"):
        command.append("--deterministic-mode")
    if mode != "cli":
        config = tmp_path / "training.yaml"
        config.write_text(f"language_model:\n  deterministic_mode: {str(enabled).lower()}\n")
        command.extend(["--yaml-cfg", str(config)])
    environment = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith(("NCCL_", "NVTE_", "CUBLAS_", "MAMBA_", "CAUSAL_CONV1D_", "TRITON_"))
    }
    root = Path(__file__).resolve().parents[3]
    environment.update(
        CUDA_VISIBLE_DEVICES="",
        RANK="0",
        WORLD_SIZE="1",
        PYTHONPATH=os.pathsep.join([str(tmp_path), str(root)]),
    )
    output = tmp_path / "capture"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "tools.determinism.capture_recipe",
            "--bindings",
            str(bindings),
            "--output",
            str(output),
            "--recipe-id",
            mode,
            "--",
            *command,
        ],
        env=environment,
        text=True,
        capture_output=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr
    capture = json.loads((output / "rank-0.json").read_text())
    assert capture["complete"] and not capture["truncated"]
    assert {item["signature"]["phase"] for item in capture["operations"]} == {
        "forward",
        "forward_backward",
    }
    assert all(
        item["signature"]["deterministic_algorithms"] is enabled for item in capture["operations"]
    )


@pytest.mark.parametrize("change", ["revision", "dirty", "environment"])
def test_capture_rejects_context_drift_after_successful_training(tmp_path, monkeypatch, change):
    before = inventory()["context"] | {"environment": {"NCCL_ALGO": "Ring"}}
    after = copy.deepcopy(before)
    after[change] = {"revision": "b" * 40, "dirty": True, "environment": {"NCCL_ALGO": "Tree"}}[
        change
    ]
    contexts = iter((before, after))
    monkeypatch.setattr(capture_recipe, "source_context", lambda torch: next(contexts))
    monkeypatch.setenv("RANK", "0")
    bindings = tmp_path / "bindings.json"
    bindings.write_text("[]")
    script = tmp_path / "train.py"
    script.write_text("completed = True\n")
    output = tmp_path / "capture"
    previous_argv, previous_path = sys.argv[:], sys.path[:]
    with pytest.raises(RuntimeError, match="context changed"):
        capture_recipe.main(
            [
                "--bindings",
                str(bindings),
                "--output",
                str(output),
                "--recipe-id",
                "drift",
                "--",
                str(script),
            ]
        )
    assert sys.argv == previous_argv and sys.path == previous_path
    captured = json.loads((output / "rank-0.json").read_text())
    assert not captured["complete"]
    assert captured["context"] == before and captured["context_after"] == after
    # Even an otherwise matching operation cannot be certified by this capture.
    captured["operations"] = inventory()["operations"]
    proof = evidence()
    proof["context"] = before
    report = build_report([captured], [proof])
    assert report["counts"][DETERMINISTIC] == 0
    assert report["counts"][UNVERIFIED] == 1
    assert report["capture_issues"]
    # The consumer also rejects recorded drift if a producer incorrectly sets
    # the completion flag; a passing operation test cannot override it.
    captured["complete"] = True
    assert build_report([captured], [proof])["counts"][UNVERIFIED] == 1


@pytest.mark.parametrize("override_apply", [False, True])
def test_capture_module_alias_to_autograd_apply_preserves_gradients(monkeypatch, override_apply):
    class Square(torch.autograd.Function):
        @staticmethod
        def forward(ctx, value):
            ctx.save_for_backward(value)
            return value.square()

        @staticmethod
        def backward(ctx, gradient):
            (value,) = ctx.saved_tensors
            return gradient * 2 * value

    class ExplicitApply(Square):
        @classmethod
        def apply(cls, *args, **kwargs):
            return super().apply(*args, **kwargs)

    implementation = ExplicitApply if override_apply else Square
    module = ModuleType("autograd_alias_kernel")
    original = module.activation = implementation.apply
    monkeypatch.setitem(sys.modules, module.__name__, module)
    recorder = Inventory(torch, 10)
    binding = {
        "target": "autograd_alias_kernel:activation",
        "op_id": "square",
        "implementation": "autograd:square",
    }
    with install_bindings(recorder, [binding]):
        value = torch.tensor([2.0, -3.0], requires_grad=True)
        result = module.activation(value)
        torch.testing.assert_close(result, torch.tensor([4.0, 9.0]))
        result.sum().backward()
        torch.testing.assert_close(value.grad, torch.tensor([4.0, -6.0]))
    assert module.activation is original
    assert implementation.apply == original
    assert {row["signature"]["phase"] for row in recorder.operations.values()} == {
        "forward",
        "forward_backward",
    }


@pytest.mark.parametrize("kind", ["instance_method", "builtin_method", "classmethod"])
def test_capture_rejects_bound_callables_with_unrecorded_state(monkeypatch, kind):
    class Stateful:
        def operation(self, value):
            return value

        @classmethod
        def class_operation(cls, value):
            return value

    module = ModuleType("stateful_bound_kernel")
    original = module.activation = {
        "instance_method": Stateful().operation,
        "builtin_method": [].append,
        "classmethod": Stateful.class_operation,
    }[kind]
    monkeypatch.setitem(sys.modules, module.__name__, module)
    binding = {
        "target": "stateful_bound_kernel:activation",
        "op_id": "stateful",
        "implementation": "stateful",
    }
    with pytest.raises(ValueError, match="Binding must be"):
        with install_bindings(Inventory(torch, 10), [binding]):
            pass
    assert module.activation is original
