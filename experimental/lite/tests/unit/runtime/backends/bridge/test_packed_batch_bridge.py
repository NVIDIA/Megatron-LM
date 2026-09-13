# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Unit coverage for the PackedBatch -> bridge forward metadata boundary.

CPU-only. The public ``PackedBatch`` contract stays free of transient THD metadata
(``packed_seq_params``); the bridge runtime renders Megatron-Core THD kwargs at the
forward call using the same canonical packing as the native lite protocols, so the
mlite-vs-bridge comparison is fair and context-parallel-correct.
"""

from __future__ import annotations

import torch
from examples.bench.session import _infinite_packed_batches
from megatron.lite.primitive.parallel.thd import (
    pack_nested_thd,
    reconstruct_packed_from_cp_parts,
)
from megatron.lite.runtime.backends.bridge.runtime import (
    _bridge_forward_kwargs_from_packed_batch,
    _nested_from_packed,
)
from megatron.lite.runtime.contracts.data import Batch, PackedBatch


def _packed_batch() -> PackedBatch:
    return PackedBatch(
        input_ids=torch.arange(8, dtype=torch.long),
        labels=torch.arange(100, 108, dtype=torch.long),
        seq_lens=torch.tensor([3, 5], dtype=torch.int64),
    )


def test_packed_batch_contract_carries_no_transient_thd_metadata() -> None:
    # The contract keeps its optional extensibility slots (position_ids for custom
    # layouts, routed_experts for router replay, extras for multimodal) but must
    # never carry the transient Megatron-Core THD metadata the bridge derives.
    batch = _packed_batch()
    assert hasattr(batch, "position_ids")
    assert hasattr(batch, "routed_experts")
    assert hasattr(batch, "extras")
    assert batch.position_ids is None
    assert not hasattr(batch, "packed_seq_params")
    assert not hasattr(batch, "to_bridge_dict")


def test_packed_batch_is_batch_subclass() -> None:
    assert issubclass(PackedBatch, Batch)


def test_bridge_forward_kwargs_are_transient_bridge_metadata() -> None:
    batch = _packed_batch()
    out = _bridge_forward_kwargs_from_packed_batch(batch)

    assert set(out) == {"input_ids", "labels", "position_ids", "packed_seq_params"}
    assert out["input_ids"].shape == (1, 8)
    assert out["labels"].shape == (1, 8)
    assert out["position_ids"].shape == (1, 8)
    # tp=cp=1 -> no padding, input ids unchanged.
    assert torch.equal(out["input_ids"].reshape(-1), batch.input_ids)
    # Labels are rolled one position left per sequence (last token zeroed), exactly
    # like the native pack_thd_forward_kwargs path, so both backends train on the
    # same shifted targets.
    assert torch.equal(
        out["labels"].reshape(-1),
        torch.tensor([101, 102, 0, 104, 105, 106, 107, 0]),
    )
    assert torch.equal(
        out["position_ids"].reshape(-1),
        torch.tensor([0, 1, 2, 0, 1, 2, 3, 4]),
    )

    psp = out["packed_seq_params"]
    assert psp.qkv_format == "thd"
    assert torch.equal(psp.cu_seqlens_q, torch.tensor([0, 3, 8], dtype=torch.int32))
    assert psp.max_seqlen_q == 5


def test_bridge_forward_kwargs_carry_loss_mask_only_inside_bridge() -> None:
    batch = _packed_batch()
    batch.loss_mask = torch.tensor([1, 1, 0, 1, 1, 1, 0, 1], dtype=torch.long)
    out = _bridge_forward_kwargs_from_packed_batch(batch)
    assert "loss_mask" in out
    # loss_mask is rolled with the labels so it masks the shifted targets.
    assert torch.equal(
        out["loss_mask"].reshape(-1),
        torch.tensor([1, 0, 0, 1, 1, 0, 1, 0]),
    )


def test_bridge_forward_kwargs_are_context_parallel_correct() -> None:
    # cp_size=2 must pad to the zigzag (2*cp) alignment, keep full cu_seqlens, and
    # hand each rank only its half of the tokens — the exact behaviour the previous
    # hand-rolled implementation lacked.
    batch = _packed_batch()
    seq_lens = batch.seq_lens

    # Full CP-aligned reference (padded but not yet CP-split).
    ref = pack_nested_thd(
        _nested_from_packed(batch.input_ids, seq_lens),
        tp_size=1,
        cp_size=2,
        cp_rank=0,
        split_cp=False,
    )
    full_padded = int(ref.cu_seqlens_padded[-1].item())
    assert full_padded == 12  # [3->4, 5->8] padded to 2*cp alignment

    rank0 = _bridge_forward_kwargs_from_packed_batch(batch, cp_size=2, cp_rank=0)
    rank1 = _bridge_forward_kwargs_from_packed_batch(batch, cp_size=2, cp_rank=1)

    for local in (rank0, rank1):
        assert local["input_ids"].shape == (1, full_padded // 2)
        assert local["position_ids"].shape == (1, full_padded // 2)
        psp = local["packed_seq_params"]
        # cu_seqlens stays full (CP-aligned), not the unpadded [0, 3, 8].
        assert torch.equal(psp.cu_seqlens_q, ref.cu_seqlens_padded)
        assert int(getattr(psp, "local_cp_size", 1)) == 2

    # The two rank-local zigzag shards reconstruct the full padded sequence.
    recon = reconstruct_packed_from_cp_parts(
        [rank0["input_ids"][0], rank1["input_ids"][0]],
        cu_seqlens_padded=ref.cu_seqlens_padded,
        cp_size=2,
        dim=0,
    )
    assert torch.equal(recon, ref.input_ids[0])


def test_infinite_packed_batches_shape_and_determinism() -> None:
    gen_a = _infinite_packed_batches(vocab_size=32, seq_len=6, device="cpu", seed=7)
    gen_b = _infinite_packed_batches(vocab_size=32, seq_len=6, device="cpu", seed=7)

    a = next(gen_a)
    b = next(gen_b)
    assert isinstance(a, PackedBatch)
    assert a.input_ids.shape == (6,)
    assert a.labels.shape == (6,)
    assert torch.equal(a.seq_lens, torch.tensor([6], dtype=torch.int64))
    # No transient THD metadata baked into bench data.
    assert a.position_ids is None
    assert torch.equal(a.input_ids, b.input_ids)
    assert torch.equal(a.labels, b.labels)
