# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Validation tests for the trace-v1 workload format.

CPU only: no GPU and no process group are required, because the trace format is pure metadata.

Covers the format contract:
  * the generator is deterministic, and the logical workload does not depend on EP size;
  * a valid trace round-trips and its digest is stable;
  * every malformed trace is rejected, for the right reason.

A rejection test that only asserted "an exception was raised" would pass if the validator
rejected everything, so each case also asserts the reason.
"""

from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path

import pytest
import torch

import benchmark as B
from trace import TENSOR_FILES, TraceValidationError, load_trace, save_trace, trace_digest

NUM_EXPERTS = 8
TOPK = 2
TOKENS = 64
HIDDEN = 512
SEED = 4242
WORKLOAD = "W1_hot_expert_set"


def _valid_ids_and_gates(tokens: int = TOKENS, seed: int = SEED, ep_size: int = 1):
    return B.routing_for_step(WORKLOAD, NUM_EXPERTS, TOPK, tokens, 0, seed, ep_size)


def _save(root: Path, name: str, *, tokens: int = TOKENS, world_size: int = 2,
          seed: int = SEED) -> Path:
    ids, gates = _valid_ids_and_gates(tokens=tokens, seed=seed)
    src = (torch.arange(ids.shape[0]) % world_size).to(torch.int32)
    d = root / name
    save_trace(
        d, ids, gates, src, num_experts=NUM_EXPERTS, topk=TOPK, hidden_size=HIDDEN,
        world_size=world_size, seed=seed, workload_id=WORKLOAD, workload_checksum="deadbeef",
        producer_commit="test",
    )
    return d


@pytest.fixture()
def workdir():
    d = Path(tempfile.mkdtemp(prefix="b1trace-"))
    try:
        yield d
    finally:
        shutil.rmtree(d, ignore_errors=True)


# --------------------------------------------------------------------------- determinism

def test_same_seed_regenerates_identical_workload():
    i1, g1 = _valid_ids_and_gates()
    i2, g2 = _valid_ids_and_gates()
    assert torch.equal(i1, i2)
    assert torch.equal(g1, g2)


def test_a_different_step_changes_the_workload():
    i1, _ = B.routing_for_step(WORKLOAD, NUM_EXPERTS, TOPK, TOKENS, 0, SEED, 1)
    i2, _ = B.routing_for_step(WORKLOAD, NUM_EXPERTS, TOPK, TOKENS, 1, SEED, 1)
    assert not torch.equal(i1, i2)


def test_layout_does_not_change_the_logical_workload():
    """The same logical workload must be produced regardless of EP size."""
    i1, g1 = _valid_ids_and_gates(ep_size=1)
    i4, g4 = _valid_ids_and_gates(ep_size=4)
    assert torch.equal(i1, i4)
    assert torch.equal(g1, g4)


def test_gates_are_normalized_and_slots_distinct():
    ids, gates = _valid_ids_and_gates()
    assert bool(torch.isfinite(gates).all())
    assert bool((gates >= 0).all())
    torch.testing.assert_close(gates.sum(dim=1), torch.ones(gates.shape[0]))
    for row in ids:
        assert len(set(row.tolist())) == TOPK


# --------------------------------------------------------------------------- round trip

def test_valid_trace_round_trips(workdir):
    d = _save(workdir, "t")
    man, tensors = load_trace(d)
    assert man.schema_version == 1
    assert man.logical_num_tokens == TOKENS
    assert set(tensors) == set(TENSOR_FILES)
    assert tuple(tensors["expert_ids"].shape) == (TOKENS, TOPK)


def test_digest_is_stable_across_identical_traces(workdir):
    a = _save(workdir, "a")
    b = _save(workdir, "b")
    assert trace_digest(a) == trace_digest(b)


def test_refuses_to_overwrite_an_existing_trace(workdir):
    _save(workdir, "t")
    with pytest.raises(TraceValidationError, match="refusing to overwrite"):
        _save(workdir, "t")


# --------------------------------------------------------------------------- rejections

def _copy(workdir: Path, tag: str) -> Path:
    src = _save(workdir, f"src-{tag}")
    dst = workdir / f"case-{tag}"
    shutil.copytree(src, dst)
    return dst


def test_rejects_tampered_tensor(workdir):
    d = _copy(workdir, "tamper")
    p = d / "tensors" / "gates.pt"
    torch.save(torch.load(p, weights_only=True) * 0.5, p)
    with pytest.raises(TraceValidationError, match="checksum mismatch"):
        load_trace(d)


def test_rejects_unsupported_schema_version(workdir):
    d = _copy(workdir, "schema")
    m = json.loads((d / "manifest.json").read_text())
    m["schema_version"] = 99
    (d / "manifest.json").write_text(json.dumps(m))
    with pytest.raises(TraceValidationError, match="unsupported schema_version"):
        load_trace(d)


def test_rejects_missing_required_field(workdir):
    d = _copy(workdir, "missing")
    m = json.loads((d / "manifest.json").read_text())
    m.pop("seed")
    (d / "manifest.json").write_text(json.dumps(m))
    with pytest.raises(TraceValidationError, match="missing required field"):
        load_trace(d)


def test_rejects_missing_tensor_file(workdir):
    d = _copy(workdir, "nofile")
    (d / "tensors" / "gates.pt").unlink()
    with pytest.raises(TraceValidationError, match="missing tensor file"):
        load_trace(d)


def test_rejects_missing_manifest(workdir):
    d = _copy(workdir, "nomanifest")
    (d / "manifest.json").unlink()
    with pytest.raises(TraceValidationError, match="missing manifest"):
        load_trace(d)


@pytest.mark.parametrize(
    "name, ids, gates, reason",
    [
        (
            "dup_slots",
            torch.zeros((8, 2), dtype=torch.int64),
            torch.full((8, 2), 0.5, dtype=torch.float32),
            "repeat an expert",
        ),
        (
            "nan_gate",
            torch.stack([torch.arange(2, dtype=torch.int64) for _ in range(8)]),
            torch.tensor([[float("nan"), 0.0]] * 8, dtype=torch.float32),
            "NaN or Inf",
        ),
        (
            "negative_gate",
            torch.stack([torch.arange(2, dtype=torch.int64) for _ in range(8)]),
            torch.tensor([[-0.5, 1.5]] * 8, dtype=torch.float32),
            "negative",
        ),
        (
            "unnormalized_gate",
            torch.stack([torch.arange(2, dtype=torch.int64) for _ in range(8)]),
            torch.full((8, 2), 0.5, dtype=torch.float32) * 2.0,
            "do not sum to 1",
        ),
    ],
)
def test_rejects_invalid_tensor_content(workdir, name, ids, gates, reason):
    d = workdir / f"bad-{name}"
    src = torch.zeros(ids.shape[0], dtype=torch.int32)
    save_trace(
        d, ids, gates, src, num_experts=NUM_EXPERTS, topk=TOPK, hidden_size=HIDDEN,
        world_size=1, seed=1, workload_id=name, workload_checksum="x", producer_commit="test",
    )
    with pytest.raises(TraceValidationError, match=reason):
        load_trace(d)


def test_rejects_out_of_range_expert_id(workdir):
    d = workdir / "badid"
    ids = torch.zeros((8, 2), dtype=torch.int64)
    ids[:, 1] = 99
    gates = torch.full((8, 2), 0.5, dtype=torch.float32)
    save_trace(
        d, ids, gates, torch.zeros(8, dtype=torch.int32), num_experts=NUM_EXPERTS, topk=TOPK,
        hidden_size=HIDDEN, world_size=1, seed=1, workload_id="badid",
        workload_checksum="x", producer_commit="test",
    )
    with pytest.raises(TraceValidationError, match="out of range"):
        load_trace(d)


def test_rejects_out_of_range_source_rank(workdir):
    d = workdir / "badsrc"
    ids = torch.stack([torch.arange(2, dtype=torch.int64) for _ in range(8)])
    gates = torch.full((8, 2), 0.5, dtype=torch.float32)
    save_trace(
        d, ids, gates, torch.full((8,), 7, dtype=torch.int32), num_experts=NUM_EXPERTS,
        topk=TOPK, hidden_size=HIDDEN, world_size=1, seed=1, workload_id="badsrc",
        workload_checksum="x", producer_commit="test",
    )
    with pytest.raises(TraceValidationError, match="source_rank out of range"):
        load_trace(d)


def test_rejects_zero_token_trace(workdir):
    d = workdir / "empty"
    save_trace(
        d, torch.zeros((0, 2), dtype=torch.int64), torch.zeros((0, 2), dtype=torch.float32),
        torch.zeros(0, dtype=torch.int32), num_experts=NUM_EXPERTS, topk=TOPK,
        hidden_size=HIDDEN, world_size=1, seed=1, workload_id="empty",
        workload_checksum="x", producer_commit="test",
    )
    with pytest.raises(TraceValidationError, match="zero logical tokens"):
        load_trace(d)
