# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""CPU contracts for state artifacts; use --confcutdir to avoid GPU conftest."""

import copy
import io
import json
import struct

import numpy as np
import pytest
import torch

from tools.determinism import training_state as states


@pytest.fixture
def state():
    return {
        "model": {"weight": torch.tensor([1.0, -0.0]), "extra": io.BytesIO(b"metadata")},
        "gradients": {"weight": torch.tensor([0.25, 0.5])},
        "optimizer": {"first_moment": torch.ones(2), "second_moment": torch.ones(2), "step": 1},
        "precision": {"mode": "fp32", "loss_scaler": None},
        "rng": {"numpy": np.array([1, 2], dtype=np.uint32), "torch": torch.get_rng_state()},
        "scheduler": {"step": 1, "lr": 0.01},
        "dataloader": {"position": 4, "permutation": [2, 0, 3, 1]},
    }


@pytest.fixture
def provenance():
    # Deliberately synthetic metadata for protocol unit tests, not GPU evidence.
    return {
        "source_revision": "fixture",
        "source_dirty": False,
        "software": {"torch": str(torch.__version__)},
        "hardware": {"device": "cpu"},
        "recipe": {"name": "synthetic_state_fixture"},
    }


@pytest.fixture
def pair(tmp_path, monkeypatch, state, provenance):
    def write(left=None, right=None, *, steps=(1,), world_size=1, resume_from=None):
        roots = [tmp_path / "reference", tmp_path / "candidate"]
        for i, root in enumerate(roots):
            for step in steps:
                for rank in range(world_size):
                    monkeypatch.setattr(states, "PROCESS_ID", f"fixture-process-{i}-{rank}")
                    states.write_snapshot(
                        root,
                        (left if i == 0 else right) or state,
                        step=step,
                        rank=rank,
                        world_size=world_size,
                        run_id=f"run-{i}",
                        provenance=provenance,
                        resume_from=resume_from if i else None,
                    )
                    if step == steps[-1]:
                        states.complete_rank(
                            root,
                            rank=rank,
                            world_size=world_size,
                            run_id=f"run-{i}",
                            steps=list(steps),
                            provenance=provenance,
                        )
        return roots

    return write


def compare(roots, *, steps=(1,), world_size=1, comparison="fresh"):
    return states.compare_runs(
        *roots, steps=list(steps), world_size=world_size, comparison=comparison
    )


def edit_metadata(root, edit, *, step=1, rank=0):
    path = states.snapshot_path(root, step, rank)
    metadata = json.loads(path.read_text())
    edit(metadata)
    path.write_text(json.dumps(metadata))


def test_equal_complete_state_across_declared_steps_and_ranks(pair):
    result = compare(pair(steps=(1, 3), world_size=2), steps=(1, 3), world_size=2)
    assert result["status"] == "equal"
    assert result["compared_snapshots"] == result["required_snapshots"] == 4
    assert result["first_difference"] is None


@pytest.mark.parametrize("component", states.COMPONENTS)
def test_each_state_component_can_break_equality(pair, state, component):
    changed = copy.deepcopy(state)
    if component in ("model", "gradients"):
        changed[component]["weight"][0] += 1
    elif component == "optimizer":
        # A matching first moment must not hide a changed second moment.
        changed[component]["second_moment"][0] += 1
    elif component == "rng":
        changed[component]["numpy"][0] += 1
    elif component == "precision":
        changed[component]["mode"] = "bf16"
    else:
        changed[component][next(iter(changed[component]))] += 1
    result = compare(pair(right=changed))
    assert result["status"] == "different"
    assert result["first_difference"]["path"][0] == component


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16, torch.complex64])
def test_signed_zero_tensor_bytes_are_not_numerical_equality(pair, state, dtype):
    left = copy.deepcopy(state)
    right = copy.deepcopy(state)
    left["model"]["weight"] = torch.tensor([0.0], dtype=dtype)
    right["model"]["weight"] = torch.tensor([-0.0], dtype=dtype)
    assert torch.equal(left["model"]["weight"], right["model"]["weight"])
    result = compare(pair(left, right))
    assert result["status"] == "different"
    assert result["first_difference"]["flat_element_index"] == 0


def test_nan_payload_is_compared_without_equating_all_nans(pair, state):
    left, right = copy.deepcopy(state), copy.deepcopy(state)
    left["model"]["weight"] = torch.tensor([0x7FC00001], dtype=torch.int32).view(torch.float32)
    right["model"]["weight"] = torch.tensor([0x7FC00002], dtype=torch.int32).view(torch.float32)
    assert torch.isnan(left["model"]["weight"]).all()
    assert torch.isnan(right["model"]["weight"]).all()
    assert compare(pair(left, right))["status"] == "different"


@pytest.mark.parametrize("value", [-0.0, struct.unpack("!d", bytes.fromhex("7ff8000000000001"))[0]])
def test_python_float_payload_is_preserved(pair, state, value):
    left, right = copy.deepcopy(state), copy.deepcopy(state)
    left["scheduler"]["lr"] = 0.0 if value == 0 else float("nan")
    right["scheduler"]["lr"] = value
    assert compare(pair(left, right))["status"] == "different"


def test_layout_is_logical_contents_without_uninitialized_storage_padding(pair, state):
    left, right = copy.deepcopy(state), copy.deepcopy(state)
    value = torch.arange(12.0).reshape(3, 4).t()
    left["model"]["weight"] = value
    right["model"]["weight"] = value.contiguous()
    assert compare(pair(left, right))["status"] == "equal"


def test_first_divergence_orders_step_rank_and_state_before_later_scalar(pair):
    roots = pair(steps=(1, 2), world_size=2)
    path = states.snapshot_path(roots[1], 1, 1).with_suffix(".bin")
    data = bytearray(path.read_bytes())
    metadata = json.loads(path.with_suffix(".json").read_text())
    weight = next(item for item in metadata["index"] if item["path"] == ["model", "weight"])
    data[weight["offset"]] ^= 1
    path.write_bytes(data)
    edit_metadata(
        roots[1],
        lambda m: next(i for i in m["index"] if i["path"] == ["dataloader", "position"]).update(
            value=999
        ),
        step=1,
        rank=1,
    )
    result = compare(roots, steps=(1, 2), world_size=2)
    assert result["status"] == "different"
    assert result["first_difference"] == {
        "step": 1,
        "rank": 1,
        "path": ["model", "weight"],
        "reason": "bytes",
        "byte_offset": 0,
        "flat_element_index": 0,
    }


@pytest.mark.parametrize(
    "failure",
    ["missing", "null", "empty", "empty_string", "empty_buffer", "empty_gradients", "unsupported"],
)
def test_incomplete_capture_cannot_pass(tmp_path, state, provenance, failure):
    if failure == "missing":
        del state["rng"]
    elif failure == "null":
        state["dataloader"] = None
    elif failure == "empty":
        state["precision"] = {}
    elif failure == "empty_string":
        state["precision"] = ""
    elif failure == "empty_buffer":
        state["optimizer"] = b""
    elif failure == "empty_gradients":
        state["gradients"] = {"weight": torch.empty(0)}
    else:
        state["optimizer"]["unknown"] = object()
    with pytest.raises(states.UnverifiedState):
        states.write_snapshot(
            tmp_path, state, step=1, rank=0, world_size=1, run_id="a", provenance=provenance
        )
    metadata = json.loads(states.snapshot_path(tmp_path, 1, 0).read_text())
    assert metadata["status"] == "not_verified"


@pytest.mark.parametrize(
    "failure",
    [
        "missing_rank",
        "truncated",
        "wrong_rank",
        "unknown_kind",
        "missing_child",
        "empty_provenance",
        "same_process",
        "same_run",
        "different_recipe",
    ],
)
def test_incomplete_or_incompatible_evidence_is_unverified(pair, failure):
    roots = pair(world_size=2)
    if failure == "missing_rank":
        states.snapshot_path(roots[1], 1, 1).unlink()
    elif failure == "truncated":
        states.snapshot_path(roots[1], 1, 0).with_suffix(".bin").write_bytes(b"")
    else:

        def modify(meta):
            if failure == "wrong_rank":
                meta["rank"] = 99
            elif failure == "unknown_kind":
                meta["index"][0]["kind"] = "not_a_state_type"
            elif failure == "missing_child":
                meta["index"].pop()
            elif failure == "empty_provenance":
                meta["provenance"] = {}
            elif failure == "same_process":
                meta["process_id"] = "fixture-process-0-0"
            elif failure == "same_run":
                meta["run_id"] = "run-0"
            else:
                meta["provenance"]["recipe"]["name"] = "another_recipe"

        edit_metadata(roots[1], modify)
    assert compare(roots, world_size=2)["status"] == "not_verified"


def test_dirty_source_keeps_observation_but_cannot_pass_gate(pair, provenance):
    provenance["source_dirty"] = True
    result = compare(pair())
    assert result["status"] == "not_verified"
    assert result["comparison_status"] == "equal"


def test_same_process_cannot_be_used_as_fresh_replay(pair):
    roots = pair()
    edit_metadata(roots[1], lambda meta: meta.update(process_id="fixture-process-0-0"))
    path = roots[1] / "complete-rank-000000.json"
    completion = json.loads(path.read_text())
    completion["process_id"] = "fixture-process-0-0"
    path.write_text(json.dumps(completion))
    assert "independent fresh processes" in compare(roots)["reason"]


def test_snapshots_from_failed_worker_are_unverified_even_if_all_steps_exist(pair):
    roots = pair(steps=(1, 2))
    (roots[1] / "complete-rank-000000.json").unlink()
    assert compare(roots, steps=(1, 2))["status"] == "not_verified"


def test_snapshot_is_not_overwritten(pair, state, provenance):
    roots = pair()
    path = states.snapshot_path(roots[0], 1, 0)
    original = path.read_bytes()
    with pytest.raises(FileExistsError):
        states.write_snapshot(
            roots[0], state, step=1, rank=0, world_size=1, run_id="oops", provenance=provenance
        )
    assert path.read_bytes() == original


@pytest.mark.parametrize("failure", [None, "missing", "modified", "identity", "too_late"])
def test_resume_requires_actual_matching_checkpoint(tmp_path, pair, failure):
    checkpoint = states.checkpoint_path(tmp_path / "reference", 1, 0)
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"synthetic checkpoint identity fixture")
    record = states.record_checkpoint(tmp_path / "reference", step=1, rank=0, run_id="run-0")
    if failure == "identity":
        record["checkpoint_sha256"] = "wrong"
    elif failure == "too_late":
        record["step"] = 2
    roots = pair(steps=(2,), resume_from=record)
    if failure == "missing":
        checkpoint.unlink()
    elif failure == "modified":
        checkpoint.write_bytes(b"changed after recording")
    result = compare(roots, steps=(2,), comparison="resume")
    assert result["status"] == ("equal" if failure is None else "not_verified")


def test_nonempty_explicit_denominator_required(tmp_path):
    result = states.compare_runs(tmp_path, tmp_path, steps=[], world_size=1, comparison="fresh")
    assert result["status"] == "not_verified"


def test_resume_cannot_mix_two_valid_checkpoint_identities(tmp_path, pair):
    records = {}
    for step in (1, 2):
        checkpoint = states.checkpoint_path(tmp_path / "reference", step, 0)
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        checkpoint.write_bytes(f"synthetic checkpoint {step}".encode())
        records[step] = states.record_checkpoint(
            tmp_path / "reference", step=step, rank=0, run_id="run-0"
        )
    roots = pair(steps=(3, 4), resume_from=records[2])
    edit_metadata(roots[1], lambda meta: meta.update(resume_from=records[1]), step=4)
    result = compare(roots, steps=(3, 4), comparison="resume")
    assert result["status"] == "not_verified"
    assert "Checkpoint step differs" in result["reason"]
