# Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from types import SimpleNamespace

import pytest
import torch

from megatron.core.packed_seq_params import PackedSeqParams, pad_sequence_for_thd
from megatron.core.parallel_state import get_context_parallel_group
from megatron.core.transformer import mtp_sequence_roll
from megatron.core.transformer import multi_token_prediction as multi_token_prediction_module
from megatron.core.transformer.mtp_sequence_roll import (
    ContiguousPackedCPRollContext,
    ContiguousPackedCPRollHalos,
    ContiguousPackedSeqRollPlan,
    LocalRollContext,
    MTPSequenceRollField,
    ZigzagPackedCPRollContext,
    prepare_mtp_sequence_roll_context,
    prepare_mtp_sequence_roll_fields,
    roll_tensor,
)
from megatron.core.transformer.multi_token_prediction import (
    MultiTokenPredictionBlock,
    process_mtp_loss,
)
from megatron.core.transformer.transformer_config import TransformerConfig
from tests.unit_tests.test_utilities import Utils

pytestmark = pytest.mark.launch_on_gb200


@pytest.mark.parametrize(
    "name",
    [
        "MTPSequenceRollHalos",
        "ContiguousPackedCPRollHalos",
        "MTPSequenceRollContext",
        "ContiguousPackedSeqRollPlan",
        "ContiguousPackedCPRollContext",
        "prepare_mtp_sequence_roll_context",
        "roll_tensor",
    ],
)
def test_sequence_roll_legacy_exports(name):
    """Moving the implementation must preserve the established import path."""
    assert getattr(multi_token_prediction_module, name) is getattr(mtp_sequence_roll, name)


def _packed_params(cu_seqlens, *, partition_mode="contiguous"):
    return PackedSeqParams(
        cu_seqlens_q=cu_seqlens,
        cu_seqlens_kv=cu_seqlens,
        max_seqlen_q=8,
        max_seqlen_kv=8,
        qkv_format="thd",
        cp_partition_mode=partition_mode,
    )


def _packed_absolute_reference(source, boundaries, offset, fill_value):
    expected = torch.full_like(source, fill_value)
    physical_ends = list(boundaries[1:])
    if not physical_ends or physical_ends[-1] < source.size(-1):
        physical_ends.append(source.size(-1))
    start = 0
    for end in physical_ends:
        end = int(end)
        if offset < end - start:
            expected[..., start : end - offset] = source[..., start + offset : end]
        start = end
    return expected


@pytest.mark.parametrize("offset", [1, 2, 7])
def test_local_unpacked_absolute_rows_preserve_layout_and_source(offset):
    ids = torch.arange(16, dtype=torch.long).view(2, 8)
    hidden = torch.arange(48, dtype=torch.float32).view(8, 2, 3)
    padding = torch.zeros_like(ids, dtype=torch.bool)
    ids_before = ids.clone()
    hidden_before = hidden.clone()

    bare = prepare_mtp_sequence_roll_context(ids, None, None)
    assert isinstance(bare, LocalRollContext)
    context = bare.prepare_fields(
        [
            MTPSequenceRollField("ids", ids, -1, 0, 0),
            MTPSequenceRollField("hidden", hidden, 0, 1, -1.0),
            MTPSequenceRollField("padding", padding, -1, 0, True),
        ],
        max_offset=7,
    )

    expected_ids = torch.zeros_like(ids)
    expected_ids[:, : 8 - offset] = ids[:, offset:]
    expected_hidden = torch.full_like(hidden, -1.0)
    expected_hidden[: 8 - offset] = hidden[offset:]
    expected_padding = torch.ones_like(padding)
    expected_padding[:, : 8 - offset] = padding[:, offset:]

    assert context.keys == ("ids", "hidden", "padding")
    assert context.max_offset == 7
    assert context.address("ids", offset).row_indices.dtype == torch.int32
    assert context.address("ids", offset).halo is None
    assert torch.equal(context.materialize("ids", offset), expected_ids)
    assert torch.equal(context.materialize("hidden", offset), expected_hidden)
    assert torch.equal(context.materialize("padding", offset), expected_padding)
    assert context.materialize("ids", 0).data_ptr() == ids.data_ptr()
    assert not hasattr(context._prepared_fields[0], "original")
    assert torch.equal(context.materialize_all("ids")[offset - 1], expected_ids)
    assert torch.equal(ids, ids_before)
    assert torch.equal(hidden, hidden_before)


@pytest.mark.parametrize("offset", [1, 2, 7])
def test_local_packed_absolute_rows_respect_boundaries_and_implicit_tail(offset):
    tokens = torch.arange(1, 11, dtype=torch.long).view(1, 10)
    padding = torch.zeros_like(tokens, dtype=torch.bool)
    cu_seqlens = torch.tensor([0, 3, 7], dtype=torch.int32)
    packed = _packed_params(cu_seqlens)

    bare = prepare_mtp_sequence_roll_context(tokens, None, packed)
    assert isinstance(bare, LocalRollContext)
    context = bare.prepare_fields(
        [
            MTPSequenceRollField("tokens", tokens, -1, 0, 0),
            MTPSequenceRollField("padding", padding, -1, 0, True),
        ],
        max_offset=7,
    )

    assert torch.equal(
        context.materialize("tokens", offset),
        _packed_absolute_reference(tokens, [0, 3, 7], offset, 0),
    )
    assert torch.equal(
        context.materialize("padding", offset),
        _packed_absolute_reference(padding, [0, 3, 7], offset, True),
    )
    address = context.address("tokens", offset)
    assert address.row_indices.dtype == torch.int32
    assert address.valid_rows.dtype == torch.bool


@pytest.mark.internal
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_thd_cp1_prepared_rows_capture_and_replay():
    """Packed CP1 row addressing remains data-driven across CUDA Graph replay."""
    source = torch.arange(1, 9, dtype=torch.long, device="cuda").view(1, 8)
    cu_seqlens = torch.tensor([0, 3, 8], dtype=torch.int32, device="cuda")
    packed = _packed_params(cu_seqlens)
    bare = prepare_mtp_sequence_roll_context(source, None, packed)
    assert isinstance(bare, LocalRollContext)
    context = bare.prepare_fields([MTPSequenceRollField("tokens", source, -1, 0, 0)], max_offset=3)

    warmup_stream = torch.cuda.Stream()
    warmup_stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(warmup_stream):
        for _ in range(3):
            context.materialize_all("tokens")
    torch.cuda.current_stream().wait_stream(warmup_stream)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = context.materialize_all("tokens")
    graph.replay()
    torch.cuda.synchronize()
    for offset, actual in enumerate(captured, 1):
        assert torch.equal(actual, _packed_absolute_reference(source, [0, 3, 8], offset, 0))

    source.copy_(torch.arange(21, 29, dtype=torch.long, device="cuda").view(1, 8))
    graph.replay()
    torch.cuda.synchronize()
    for offset, actual in enumerate(captured, 1):
        assert torch.equal(actual, _packed_absolute_reference(source, [0, 3, 8], offset, 0))


def test_local_context_does_not_change_existing_roll_dispatch():
    tokens = torch.tensor([[1, 2, 3, 4, 5, 6]], dtype=torch.long)
    unpacked_context = prepare_mtp_sequence_roll_context(tokens, None, None)
    expected_unpacked = roll_tensor([tokens], packed_seq_params=None)[0]
    actual_unpacked = roll_tensor([tokens], packed_seq_params=None, roll_context=unpacked_context)[
        0
    ]
    assert torch.equal(actual_unpacked, expected_unpacked)

    cu_seqlens = torch.tensor([0, 2, 6], dtype=torch.int32)
    packed = _packed_params(cu_seqlens)
    packed_context = prepare_mtp_sequence_roll_context(tokens, None, packed)
    expected_packed = roll_tensor([tokens], packed_seq_params=packed)[0]
    actual_packed = roll_tensor([tokens], packed_seq_params=packed, roll_context=packed_context)[0]
    assert torch.equal(actual_packed, expected_packed)


def test_helper_keeps_unsupported_shapes_and_zigzag_on_fallback():
    class FakeCPGroup:
        def __init__(self, size):
            self._size = size

        def size(self):
            return self._size

    assert prepare_mtp_sequence_roll_context(torch.zeros(4), None, None) is None
    packed = _packed_params(torch.tensor([0, 4], dtype=torch.int32), partition_mode="zigzag")
    assert prepare_mtp_sequence_roll_context(torch.zeros((1, 4)), FakeCPGroup(2), packed) is None


def test_contiguous_prepare_is_atomic_deterministic_and_one_hop(monkeypatch):
    class FakeWork:
        def wait(self):
            return None

    class FakeCPGroup:
        pass

    plan = ContiguousPackedSeqRollPlan(
        invalid_next=torch.tensor([False, True, False, False]),
        sequence_length=4,
        device=torch.device("cpu"),
        cp_group=FakeCPGroup(),
        recv_rank=1,
        send_rank=None,
        has_sequences=True,
        right_halo_valid_count=torch.tensor(2, dtype=torch.long),
    )
    bare = ContiguousPackedCPRollContext(plan=plan, batch_size=1)
    ids = torch.tensor([[1, 2, 3, 4]], dtype=torch.long)
    padding = torch.zeros_like(ids, dtype=torch.bool)
    grouped_calls = []

    def fake_p2p_op(op, tensor, peer, group=None):
        return SimpleNamespace(op=op, tensor=tensor, peer=peer, group=group)

    def fake_batch_isend_irecv(ops):
        grouped_calls.append(ops)
        # Sorted key order is ids (long), then padding (bool), independent of the
        # declaration order below.
        recv_ops = [op for op in ops if op.op is torch.distributed.irecv]
        assert [op.tensor.dtype for op in recv_ops] == [torch.long, torch.bool]
        recv_ops[0].tensor.copy_(torch.tensor([[5], [6], [7]], dtype=torch.long))
        recv_ops[1].tensor.zero_()
        return [FakeWork() for _ in ops]

    monkeypatch.setattr(torch.distributed, "P2POp", fake_p2p_op)
    monkeypatch.setattr(torch.distributed, "batch_isend_irecv", fake_batch_isend_irecv)
    context = bare.prepare_fields(
        [
            MTPSequenceRollField("padding", padding, -1, 0, True),
            MTPSequenceRollField("ids", ids, -1, 0, 0),
        ],
        max_offset=3,
    )

    assert len(grouped_calls) == 1
    assert context.keys == ("padding", "ids")
    assert context.address("ids", 3).halo is not None
    expected_by_offset = (
        torch.tensor([[2, 0, 4, 5]]),
        torch.tensor([[0, 0, 5, 6]]),
        torch.tensor([[0, 0, 6, 0]]),
    )
    assert all(
        torch.equal(actual, expected)
        for actual, expected in zip(context.materialize_all("ids"), expected_by_offset)
    )

    grouped_calls.clear()
    unsupported = bare.prepare_fields([MTPSequenceRollField("ids", ids, -1, 0, 0)], max_offset=5)
    assert unsupported is bare
    assert grouped_calls == []


def test_contiguous_validates_whole_group_before_p2p(monkeypatch):
    plan = ContiguousPackedSeqRollPlan(
        invalid_next=torch.zeros(4, dtype=torch.bool),
        sequence_length=4,
        device=torch.device("cpu"),
        cp_group=object(),
        recv_rank=1,
        send_rank=None,
        has_sequences=True,
        right_halo_valid_count=torch.tensor(3),
    )
    context = ContiguousPackedCPRollContext(plan=plan, batch_size=1)
    monkeypatch.setattr(
        torch.distributed,
        "batch_isend_irecv",
        lambda _: pytest.fail("Field validation must finish before P2P starts."),
    )
    with pytest.raises(ValueError, match="sequence length"):
        context.prepare_fields(
            [
                MTPSequenceRollField("good", torch.zeros((1, 4)), -1, 0),
                MTPSequenceRollField("bad", torch.zeros((1, 3)), -1, 0),
            ],
            max_offset=5,
        )


def test_contiguous_rejects_grad_fields_before_p2p(monkeypatch):
    plan = ContiguousPackedSeqRollPlan(
        invalid_next=torch.zeros(4, dtype=torch.bool),
        sequence_length=4,
        device=torch.device("cpu"),
        cp_group=object(),
        recv_rank=1,
        send_rank=None,
        has_sequences=True,
        right_halo_valid_count=torch.tensor(3),
    )
    context = ContiguousPackedCPRollContext(plan=plan, batch_size=1)
    monkeypatch.setattr(
        torch.distributed,
        "batch_isend_irecv",
        lambda _: pytest.fail("Gradient validation must finish before P2P starts."),
    )
    with pytest.raises(ValueError, match="does not support fields that require gradients"):
        context.prepare_fields(
            [MTPSequenceRollField("hidden", torch.zeros((1, 4), requires_grad=True), -1, 0)],
            max_offset=5,
        )


def test_empty_prepare_replaces_fields_with_bare_sibling():
    tokens = torch.arange(4, dtype=torch.long).view(1, 4)
    local = LocalRollContext(
        sequence_length=4, batch_size=1, device=torch.device("cpu")
    ).prepare_fields([MTPSequenceRollField("tokens", tokens, -1, 0, 0)], max_offset=2)
    cleared_local = local.prepare_fields([], max_offset=2)
    assert cleared_local is not local
    assert cleared_local.keys == ()
    assert cleared_local.max_offset == 0
    assert cleared_local.sequence_length == local.sequence_length
    assert cleared_local.batch_size == local.batch_size

    plan = ContiguousPackedSeqRollPlan(
        invalid_next=torch.zeros(4, dtype=torch.bool),
        sequence_length=4,
        device=torch.device("cpu"),
        cp_group=object(),
        recv_rank=None,
        send_rank=None,
        has_sequences=True,
        right_halo_valid_count=torch.tensor(0),
    )
    halos = ContiguousPackedCPRollHalos(input_ids=torch.zeros((1, 2), dtype=torch.long))
    contiguous = ContiguousPackedCPRollContext(plan=plan, halos=halos, batch_size=1).prepare_fields(
        [MTPSequenceRollField("tokens", tokens, -1, 0, 0)], max_offset=2
    )
    cleared_contiguous = contiguous.prepare_fields([], max_offset=5)
    assert cleared_contiguous is not contiguous
    assert cleared_contiguous.keys == ()
    assert cleared_contiguous.max_offset == 0
    assert cleared_contiguous.plan is plan
    assert cleared_contiguous.halos is halos
    assert cleared_contiguous.batch_size == contiguous.batch_size


def test_late_bound_siblings_reuse_local_and_contiguous_geometry(monkeypatch):
    tokens = torch.arange(4, dtype=torch.long).view(1, 4)
    mask = torch.zeros_like(tokens, dtype=torch.bool)

    local = LocalRollContext(
        sequence_length=4, batch_size=1, device=torch.device("cpu")
    ).prepare_fields((MTPSequenceRollField("tokens", tokens, -1, 0, 0),), max_offset=3)
    local_indices = local._roll_row_indices
    local_valid = local._roll_valid_rows
    monkeypatch.setattr(
        mtp_sequence_roll,
        "_build_local_roll_geometry",
        lambda **_: pytest.fail("Local late binding should reuse row geometry."),
    )
    rebound_local = local.prepare_fields(
        (MTPSequenceRollField("mask", mask, -1, 0, True),), max_offset=2
    )
    assert rebound_local._roll_row_indices is local_indices
    assert rebound_local._roll_valid_rows is local_valid
    assert rebound_local.max_offset == 3

    plan = ContiguousPackedSeqRollPlan(
        invalid_next=torch.zeros(4, dtype=torch.bool),
        sequence_length=4,
        device=torch.device("cpu"),
        cp_group=object(),
        recv_rank=None,
        send_rank=None,
        has_sequences=True,
        right_halo_valid_count=torch.tensor(0),
    )
    contiguous = ContiguousPackedCPRollContext(plan=plan, batch_size=1).prepare_fields(
        (MTPSequenceRollField("tokens", tokens, -1, 0, 0),), max_offset=3
    )
    contiguous_indices = contiguous._roll_row_indices
    contiguous_valid = contiguous._roll_valid_rows
    monkeypatch.setattr(
        mtp_sequence_roll,
        "_build_contiguous_packed_cp_roll_geometry",
        lambda *_, **__: pytest.fail("Contiguous late binding should reuse row geometry."),
    )
    rebound_contiguous = contiguous.prepare_fields(
        (MTPSequenceRollField("mask", mask, -1, 0, True),), max_offset=2
    )
    assert rebound_contiguous._roll_row_indices is contiguous_indices
    assert rebound_contiguous._roll_valid_rows is contiguous_valid
    assert rebound_contiguous.max_offset == 3


def test_source_identity_guard_reprepares_stale_fields_atomically():
    stale = torch.tensor([[1, 2, 3, 4]], dtype=torch.long)
    current = torch.tensor([[11, 12, 13, 14]], dtype=torch.long)
    bare = prepare_mtp_sequence_roll_context(stale, None, None)
    assert isinstance(bare, LocalRollContext)
    context = bare.prepare_fields(
        [MTPSequenceRollField("input_ids", stale, -1, 0, 0)], max_offset=2
    )

    assert context.is_prepared_for_fields([MTPSequenceRollField("input_ids", stale, -1, 0, 0)])
    assert not context.is_prepared_for_fields(
        [MTPSequenceRollField("input_ids", current, -1, 0, 0)]
    )
    rebound = prepare_mtp_sequence_roll_fields(
        context, [MTPSequenceRollField("input_ids", current, -1, 0, 0)], max_offset=2
    )
    assert rebound is not None
    assert rebound is not context
    assert rebound.is_prepared_for_fields([MTPSequenceRollField("input_ids", current, -1, 0, 0)])
    assert torch.equal(rebound.materialize("input_ids", 1), torch.tensor([[12, 13, 14, 0]]))

    stale.add_(100)
    assert not context.is_prepared_for_fields([MTPSequenceRollField("input_ids", stale, -1, 0, 0)])


def test_source_identity_guard_safely_falls_back_for_inference_tensors():
    with torch.inference_mode():
        source = torch.tensor([[1, 2, 3, 4]], dtype=torch.long)
        field = MTPSequenceRollField("input_ids", source, -1, 0, 0)
        bare = prepare_mtp_sequence_roll_context(source, None, None)
        assert isinstance(bare, LocalRollContext)
        context = bare.prepare_fields([field], max_offset=2)

        # Preparation itself remains useful, but an inference tensor has no
        # version counter with which to prove that a later reuse is fresh.
        assert torch.equal(context.materialize("input_ids", 1), torch.tensor([[2, 3, 4, 0]]))
        assert not context.is_prepared_for_fields([field])
        assert prepare_mtp_sequence_roll_fields(context, [field], max_offset=2) is None


def _packed_global_aligned_rows(source, cu_seqlens, offset, fill_value):
    """Return a global packed-sequence oracle for a left shift by ``offset``."""
    expected = torch.full_like(source, fill_value)
    for start, end in zip(cu_seqlens[:-1], cu_seqlens[1:]):
        start = int(start)
        end = int(end)
        if offset < end - start:
            expected[..., start : end - offset] = source[..., start + offset : end]
    return expected


def _materialize_address(address, fill_value):
    """Materialize one public sequence-roll address without using context helpers."""
    source = address.source
    if address.halo is not None:
        source = torch.cat((source, address.halo), dim=0)
    payload_shape = source.shape[2:]
    selected = source.reshape(-1, *payload_shape).index_select(
        0, address.row_indices.reshape(-1).long()
    )
    selected = selected.reshape(*address.row_indices.shape, *payload_shape)
    valid_rows = address.valid_rows
    while valid_rows.dim() < selected.dim():
        valid_rows = valid_rows.unsqueeze(-1)
    return torch.where(valid_rows, selected, selected.new_full((), fill_value))


class _FakeZigzagCPGroup:
    """Minimal effective CP group for host-only certificate and route tests."""

    def __init__(self, size):
        self._size = size

    def size(self):
        return self._size


def _certified_zigzag_params(cu_seqlens, cp_group, *, certificate, local_cp_size=None):
    """Build zigzag packed-CP metadata with an explicit scheduler certificate."""
    return PackedSeqParams(
        qkv_format="thd",
        cu_seqlens_q=cu_seqlens,
        cu_seqlens_kv=cu_seqlens,
        cu_seqlens_q_padded=cu_seqlens,
        cu_seqlens_kv_padded=cu_seqlens,
        max_seqlen_q=int(cu_seqlens[-1]),
        max_seqlen_kv=int(cu_seqlens[-1]),
        cp_partition_mode="zigzag",
        local_cp_size=local_cp_size,
        cp_group=cp_group,
        zigzag_cp_min_chunk_size=certificate,
    )


def test_uncertified_and_zero_certificate_keep_zigzag_roll_fallback():
    """Missing/zero certificates describe unsupported optimization, not invalid data."""
    static_group = _FakeZigzagCPGroup(4)
    dynamic_group = _FakeZigzagCPGroup(2)
    source = torch.arange(8).view(1, 8)
    cu_seqlens = torch.tensor([0, 16], dtype=torch.int32)

    for certificate in (None, 0):
        packed = _certified_zigzag_params(
            cu_seqlens, dynamic_group, certificate=certificate, local_cp_size=2
        )
        # The dynamic group carried by PackedSeqParams is the effective group;
        # neither an absent nor a failed certificate creates a prepared context.
        assert prepare_mtp_sequence_roll_context(source, static_group, packed) is None

    negative = _certified_zigzag_params(cu_seqlens, dynamic_group, certificate=-1, local_cp_size=2)
    with pytest.raises(ValueError, match="certificate cannot be negative"):
        prepare_mtp_sequence_roll_context(source, static_group, negative)


def test_zigzag_certificate_survives_padding_and_tightens_for_dummy_sequence():
    """Appending a shorter physical sequence conservatively lowers the certificate."""
    cu_seqlens = torch.tensor([0, 16], dtype=torch.int32)
    packed = PackedSeqParams(
        qkv_format="thd",
        cu_seqlens_q=cu_seqlens,
        cu_seqlens_kv=cu_seqlens.clone(),
        cu_seqlens_q_padded=cu_seqlens.clone(),
        cu_seqlens_kv_padded=cu_seqlens.clone(),
        max_seqlen_q=16,
        max_seqlen_kv=16,
        local_cp_size=2,
        cp_partition_mode="zigzag",
        zigzag_cp_min_chunk_size=4,
    )

    _, _, _, _, padded, _ = pad_sequence_for_thd(
        torch.ones(1, 8),
        None,
        None,
        None,
        packed,
        target_len=12,
        tail_padding_policy="append_dummy_seq",
        cp_size=2,
        cp_rank=0,
    )

    assert padded.cu_seqlens_q_padded.tolist() == [0, 16, 24]
    assert padded.zigzag_cp_min_chunk_size == 2


def test_zigzag_certificate_is_ignored_by_cuda_graph_argument_matching():
    """A scheduler runtime certificate must not select another graph runner."""
    from megatron.core.transformer.cuda_graphs import ArgMetadata, _CudaGraphRunner

    reference = PackedSeqParams(
        qkv_format="thd", cp_partition_mode="zigzag", zigzag_cp_min_chunk_size=2
    )
    current = PackedSeqParams(
        qkv_format="thd", cp_partition_mode="zigzag", zigzag_cp_min_chunk_size=8
    )
    runner = _CudaGraphRunner.__new__(_CudaGraphRunner)
    runner.fwd_graph_input_arg_metas = [ArgMetadata(reference)]
    runner.fwd_graph_input_kwarg_metas = {}

    assert runner.get_mismatch_errors((current,), {}) == []
    current.cp_partition_mode = "contiguous"
    assert runner.get_mismatch_errors((current,), {})


def test_certified_zigzag_rejects_effective_cp_group_mismatch():
    source = torch.arange(8).view(1, 8)
    dynamic_group = _FakeZigzagCPGroup(2)
    packed = _certified_zigzag_params(
        torch.tensor([0, 16], dtype=torch.int32), dynamic_group, certificate=4, local_cp_size=4
    )

    with pytest.raises(ValueError, match="must match the effective CP group size"):
        prepare_mtp_sequence_roll_context(source, _FakeZigzagCPGroup(4), packed)


@pytest.mark.parametrize(
    ("boundaries", "message"),
    [
        ((1, 16), "must start at zero"),
        ((0, 12, 8), "must be nondecreasing"),
        ((0, 12), "must cover the physical buffer exactly"),
        ((0, 14, 16), "physical chunk divisibility"),
        ((0, 4, 16), "overstates the minimum physical chunk size"),
    ],
)
def test_certified_zigzag_validates_runtime_metadata_before_p2p(monkeypatch, boundaries, message):
    cp_group = _FakeZigzagCPGroup(2)
    source = torch.arange(8).view(1, 8)
    packed = _certified_zigzag_params(
        torch.tensor(boundaries, dtype=torch.int32), cp_group, certificate=2, local_cp_size=2
    )
    bare = prepare_mtp_sequence_roll_context(source, cp_group, packed)
    assert isinstance(bare, ZigzagPackedCPRollContext)
    monkeypatch.setattr(torch.distributed, "get_rank", lambda group=None: 0)
    monkeypatch.setattr(torch.distributed, "get_process_group_ranks", lambda group: [0, 1])
    monkeypatch.setattr(
        torch.distributed,
        "batch_isend_irecv",
        lambda _: pytest.fail("Invalid certified metadata must fail before P2P."),
    )

    with pytest.raises(RuntimeError, match=message):
        bare.prepare_fields((MTPSequenceRollField("tokens", source, -1, 0, 0),), max_offset=2)


def test_certified_zigzag_multi_hop_fallback_is_atomic(monkeypatch):
    """A consumer wider than the certificate performs no partial prepare or P2P."""
    cp_group = _FakeZigzagCPGroup(2)
    source = torch.arange(8).view(1, 8)
    packed = _certified_zigzag_params(
        torch.tensor([0, 16], dtype=torch.int32), cp_group, certificate=2, local_cp_size=2
    )
    bare = prepare_mtp_sequence_roll_context(source, cp_group, packed)
    assert isinstance(bare, ZigzagPackedCPRollContext)
    monkeypatch.setattr(
        torch.distributed,
        "batch_isend_irecv",
        lambda _: pytest.fail("Unsupported multi-hop preparation must not start P2P."),
    )

    fields = (MTPSequenceRollField("tokens", source, -1, 0, 0),)
    assert bare.prepare_fields(fields, max_offset=3) is bare
    assert prepare_mtp_sequence_roll_fields(bare, fields, max_offset=3) is None
    assert bare.keys == ()
    assert bare.max_offset == 0


def test_certified_zigzag_routes_and_late_binding_reuse_geometry(monkeypatch):
    """Front/back routes use opposite peers and sibling fields share exact geometry."""

    class FakeWork:
        def wait(self):
            return None

    cp_group = _FakeZigzagCPGroup(4)
    monkeypatch.setattr(torch.distributed, "get_rank", lambda group=None: 1)
    monkeypatch.setattr(
        torch.distributed, "get_process_group_ranks", lambda group: [10, 20, 30, 40]
    )

    captured_calls = []

    def fake_p2p_op(op, tensor, peer, group=None):
        return SimpleNamespace(op=op, tensor=tensor, peer=peer, group=group)

    def fake_batch_isend_irecv(ops):
        captured_calls.append(tuple(ops))
        assert [(op.op, op.peer) for op in ops] == [
            (torch.distributed.irecv, 30),
            (torch.distributed.isend, 10),
            (torch.distributed.irecv, 10),
            (torch.distributed.isend, 30),
        ]
        recv_ops = [op for op in ops if op.op is torch.distributed.irecv]
        if recv_ops[0].tensor.dtype == torch.long:
            send_ops = [op for op in ops if op.op is torch.distributed.isend]
            assert torch.equal(send_ops[0].tensor, torch.tensor([[4], [5], [6]], dtype=torch.long))
            assert torch.equal(
                send_ops[1].tensor, torch.tensor([[24], [25], [26]], dtype=torch.long)
            )
            recv_ops[0].tensor.copy_(torch.tensor([[8], [9], [10]], dtype=torch.long))
            recv_ops[1].tensor.copy_(torch.tensor([[28], [29], [30]], dtype=torch.long))
        else:
            recv_ops[0].tensor.zero_()
            recv_ops[1].tensor.zero_()
        return [FakeWork() for _ in ops]

    monkeypatch.setattr(torch.distributed, "P2POp", fake_p2p_op)
    monkeypatch.setattr(torch.distributed, "batch_isend_irecv", fake_batch_isend_irecv)

    # CP4 rank 1 owns global front chunk [4, 8) and mirrored back chunk [24, 28).
    source = torch.tensor([[4, 5, 6, 7, 24, 25, 26, 27]], dtype=torch.long)
    packed = _certified_zigzag_params(
        torch.tensor([0, 32], dtype=torch.int32), cp_group, certificate=4, local_cp_size=4
    )
    bare = prepare_mtp_sequence_roll_context(source, cp_group, packed)
    assert isinstance(bare, ZigzagPackedCPRollContext)
    prepared = bare.prepare_fields(
        (MTPSequenceRollField("tokens", source, -1, 0, 0),), max_offset=3
    )

    assert torch.equal(
        prepared.materialize("tokens", 1),
        torch.tensor([[5, 6, 7, 8, 25, 26, 27, 28]], dtype=torch.long),
    )
    assert torch.equal(
        prepared.materialize("tokens", 3),
        torch.tensor([[7, 8, 9, 10, 27, 28, 29, 30]], dtype=torch.long),
    )
    first_indices = prepared._roll_row_indices
    first_valid = prepared._roll_valid_rows
    first_transport = prepared._roll_transport
    assert first_indices is not None and first_valid is not None and first_transport is not None

    # A late-bound TV/mask sibling with a smaller requested depth must reuse the
    # already-built row map and route metadata rather than reconstruct either.
    monkeypatch.setattr(
        mtp_sequence_roll,
        "_build_zigzag_packed_cp_roll_geometry",
        lambda **_: pytest.fail("Late binding should reuse certified zigzag geometry."),
    )
    mask = torch.zeros_like(source, dtype=torch.bool)
    rebound = prepared.prepare_fields(
        (MTPSequenceRollField("mask", mask, -1, 0, True),), max_offset=2
    )

    assert rebound.keys == ("mask",)
    assert rebound.max_offset == 3
    assert rebound._roll_row_indices is first_indices
    assert rebound._roll_valid_rows is first_valid
    assert rebound._roll_transport is first_transport
    assert len(captured_calls) == 2


def _canonical_mtp_test_layout(tensor, sequence_dim, batch_dim):
    sequence_dim %= tensor.ndim
    batch_dim %= tensor.ndim
    remaining = tuple(dim for dim in range(tensor.ndim) if dim not in (sequence_dim, batch_dim))
    return tensor.permute(sequence_dim, batch_dim, *remaining)


def _restore_mtp_test_layout(tensor, source, sequence_dim, batch_dim):
    sequence_dim %= source.ndim
    batch_dim %= source.ndim
    permutation = (sequence_dim, batch_dim) + tuple(
        dim for dim in range(source.ndim) if dim not in (sequence_dim, batch_dim)
    )
    inverse = [0] * source.ndim
    for canonical_dim, original_dim in enumerate(permutation):
        inverse[original_dim] = canonical_dim
    return tensor.permute(inverse)


def _packed_sequence_roll_oracle(
    source, offset, *, sequence_dim, batch_dim, fill_value, boundaries
):
    """Select global shifted rows without using production roll/address helpers."""
    canonical = _canonical_mtp_test_layout(source, sequence_dim, batch_dim)
    output = torch.full_like(canonical, fill_value)
    for start, end in zip(boundaries[:-1], boundaries[1:]):
        valid_length = max(end - start - offset, 0)
        if valid_length:
            output[start : start + valid_length].copy_(
                canonical[start + offset : start + offset + valid_length]
            )
    return _restore_mtp_test_layout(output, source, sequence_dim, batch_dim)


def _zigzag_shard_oracle(source, boundaries, cp_size, cp_rank, *, sequence_dim, batch_dim):
    """Produce one zigzag packed-CP shard without production partition helpers."""
    canonical = _canonical_mtp_test_layout(source, sequence_dim, batch_dim)
    local_segments = []
    for start, end in zip(boundaries[:-1], boundaries[1:]):
        if start == end:
            continue
        segment = canonical[start:end]
        assert segment.size(0) % (2 * cp_size) == 0
        chunks = segment.chunk(2 * cp_size, dim=0)
        local_segments.extend((chunks[cp_rank], chunks[2 * cp_size - cp_rank - 1]))
    local = torch.cat(local_segments, dim=0).contiguous()
    return _restore_mtp_test_layout(local, source, sequence_dim, batch_dim)


@pytest.mark.internal
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_certified_zigzag_cp2_matches_global_oracle(monkeypatch):
    """A real CP2 group validates both physical zigzag boundary directions."""
    Utils.initialize_model_parallel(tensor_model_parallel_size=1, context_parallel_size=2)
    try:
        cp_group = get_context_parallel_group()
        cp_rank = torch.distributed.get_rank(group=cp_group)
        cp_size = cp_group.size()
        assert cp_size == 2
        boundaries = (0, 16)
        max_offset = 3
        full_ids = torch.arange(1, 17, dtype=torch.long, device="cuda").view(1, -1)
        full_mask = (full_ids % 3 == 0).to(torch.float32)
        local_ids = _zigzag_shard_oracle(
            full_ids, boundaries, cp_size, cp_rank, sequence_dim=1, batch_dim=0
        )
        local_mask = _zigzag_shard_oracle(
            full_mask, boundaries, cp_size, cp_rank, sequence_dim=1, batch_dim=0
        )
        cu_seqlens = torch.tensor(boundaries, dtype=torch.int32, device="cuda")
        packed = PackedSeqParams(
            qkv_format="thd",
            cu_seqlens_q=cu_seqlens,
            cu_seqlens_kv=cu_seqlens,
            cu_seqlens_q_padded=cu_seqlens,
            cu_seqlens_kv_padded=cu_seqlens,
            max_seqlen_q=16,
            max_seqlen_kv=16,
            cp_partition_mode="zigzag",
            local_cp_size=cp_size,
            cp_group=cp_group,
            zigzag_cp_min_chunk_size=4,
        )
        bare = prepare_mtp_sequence_roll_context(local_ids, cp_group, packed)
        assert isinstance(bare, ZigzagPackedCPRollContext)

        original_batch_isend_irecv = torch.distributed.batch_isend_irecv
        p2p_calls = []

        def counted_batch_isend_irecv(ops):
            p2p_calls.append(len(ops))
            return original_batch_isend_irecv(ops)

        monkeypatch.setattr(torch.distributed, "batch_isend_irecv", counted_batch_isend_irecv)
        context = bare.prepare_fields(
            (
                MTPSequenceRollField("ids", local_ids, 1, 0, 0),
                MTPSequenceRollField("mask", local_mask, 1, 0, 0),
            ),
            max_offset=max_offset,
        )
        assert p2p_calls == [4]

        for offset in range(1, max_offset + 1):
            for key, full_source, fill_value in (("ids", full_ids, 0), ("mask", full_mask, 0)):
                global_expected = _packed_sequence_roll_oracle(
                    full_source,
                    offset,
                    sequence_dim=1,
                    batch_dim=0,
                    fill_value=fill_value,
                    boundaries=boundaries,
                )
                local_expected = _zigzag_shard_oracle(
                    global_expected, boundaries, cp_size, cp_rank, sequence_dim=1, batch_dim=0
                )
                torch.testing.assert_close(
                    context.materialize(key, offset), local_expected, rtol=0, atol=0
                )
        assert p2p_calls == [4]
    finally:
        Utils.destroy_model_parallel()


@pytest.mark.internal
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_certified_zigzag_cp4_dynamic_interleaved_group_matches_global_oracle(monkeypatch):
    """A real interleaved CP4 group prepares all fields with one neighbor exchange."""
    if Utils.world_size < 8 or Utils.world_size % 4 != 0:
        pytest.skip("This test requires at least eight ranks and a world size divisible by four.")

    Utils.initialize_model_parallel(tensor_model_parallel_size=1)
    world_size = torch.distributed.get_world_size()
    num_groups = world_size // 4
    rank_lists = [
        list(range(group_index, world_size, num_groups)) for group_index in range(num_groups)
    ]
    cp_groups = [torch.distributed.new_group(ranks) for ranks in rank_lists]
    global_rank = torch.distributed.get_rank()
    cp_group = cp_groups[global_rank % num_groups]
    cp_rank = torch.distributed.get_rank(group=cp_group)
    try:
        cp_size = 4
        max_offset = 2
        valid_boundaries = (0, 13, 33, 33)
        padded_boundaries = (0, 16, 40, 40)
        global_sequence_length = padded_boundaries[-1]

        full_padding = torch.ones((1, global_sequence_length), dtype=torch.bool, device="cuda")
        for valid_start, valid_end, padded_start in zip(
            valid_boundaries[:-1], valid_boundaries[1:], padded_boundaries[:-1]
        ):
            full_padding[:, padded_start : padded_start + valid_end - valid_start] = False
        full_ids = torch.arange(1, global_sequence_length + 1, device="cuda").view(1, -1)
        full_ids.masked_fill_(full_padding, 0)
        full_target = (
            torch.arange(global_sequence_length * 3, dtype=torch.float32, device="cuda")
            .view(global_sequence_length, 1, 3)
            .to(torch.bfloat16)
        )
        full_target.masked_fill_(full_padding.transpose(0, 1).unsqueeze(-1), 0)

        local_ids = _zigzag_shard_oracle(
            full_ids, padded_boundaries, cp_size, cp_rank, sequence_dim=1, batch_dim=0
        )
        local_padding = _zigzag_shard_oracle(
            full_padding, padded_boundaries, cp_size, cp_rank, sequence_dim=1, batch_dim=0
        )
        local_target = _zigzag_shard_oracle(
            full_target, padded_boundaries, cp_size, cp_rank, sequence_dim=0, batch_dim=1
        )

        cu_seqlens = torch.tensor(valid_boundaries, dtype=torch.int32, device="cuda")
        padded_cu_seqlens = torch.tensor(padded_boundaries, dtype=torch.int32, device="cuda")
        packed = PackedSeqParams(
            qkv_format="thd",
            cu_seqlens_q=cu_seqlens,
            cu_seqlens_kv=cu_seqlens,
            cu_seqlens_q_padded=padded_cu_seqlens,
            cu_seqlens_kv_padded=padded_cu_seqlens,
            max_seqlen_q=24,
            max_seqlen_kv=24,
            cp_partition_mode="zigzag",
            local_cp_size=cp_size,
            cp_group=cp_group,
            zigzag_cp_min_chunk_size=2,
        )
        # The explicit group is deliberately unrelated: dynamic metadata must win.
        static_cp_group = get_context_parallel_group()
        bare = prepare_mtp_sequence_roll_context(local_ids, static_cp_group, packed)
        assert isinstance(bare, ZigzagPackedCPRollContext)
        assert bare.cp_group is cp_group

        original_batch_isend_irecv = torch.distributed.batch_isend_irecv
        p2p_calls = []

        def counted_batch_isend_irecv(ops):
            p2p_calls.append(len(ops))
            return original_batch_isend_irecv(ops)

        monkeypatch.setattr(torch.distributed, "batch_isend_irecv", counted_batch_isend_irecv)
        fields = [
            MTPSequenceRollField("ids", local_ids, 1, 0, 0),
            MTPSequenceRollField("padding", local_padding, 1, 0, True),
            MTPSequenceRollField("target", local_target, 0, 1, 0),
        ]
        if cp_rank % 2:
            fields.reverse()
        context = bare.prepare_fields(fields, max_offset=max_offset)

        neighbor_count = int(cp_rank > 0) + int(cp_rank + 1 < cp_size)
        assert p2p_calls == [2 * neighbor_count * len(fields)]
        for offset in range(1, max_offset + 1):
            for key, full_source, sequence_dim, batch_dim, fill_value in (
                ("ids", full_ids, 1, 0, 0),
                ("padding", full_padding, 1, 0, True),
                ("target", full_target, 0, 1, 0),
            ):
                global_expected = _packed_sequence_roll_oracle(
                    full_source,
                    offset,
                    sequence_dim=sequence_dim,
                    batch_dim=batch_dim,
                    fill_value=fill_value,
                    boundaries=padded_boundaries,
                )
                local_expected = _zigzag_shard_oracle(
                    global_expected,
                    padded_boundaries,
                    cp_size,
                    cp_rank,
                    sequence_dim=sequence_dim,
                    batch_dim=batch_dim,
                )
                torch.testing.assert_close(
                    context.materialize(key, offset), local_expected, rtol=0, atol=0
                )

        # Every depth consumes only the already-prepared row map and halos.
        assert p2p_calls == [2 * neighbor_count * len(fields)]
    finally:
        torch.distributed.destroy_process_group(cp_group)
        Utils.destroy_model_parallel()


class TestContiguousPackedCPPreparedRollRowsDistributed:
    """Real CP tests for generic sequence-roll fields and their legacy roll oracle."""

    def teardown_method(self):
        Utils.destroy_model_parallel()

    def test_cp2_true_shards_mixed_dtype_one_grouped_exchange(self, monkeypatch):
        """One preparation serves every absolute depth without another P2P.

        Every CP rank receives a true contiguous slice of the same global packed
        microbatch.  Ranks deliberately declare the two fields in opposite order;
        transport therefore has to use the field-key order rather than declaration
        order to pair the int64 and float32 messages safely.
        """
        cp_size = 2
        Utils.initialize_model_parallel(tensor_model_parallel_size=1, context_parallel_size=cp_size)
        cp_group = get_context_parallel_group()
        cp_rank = torch.distributed.get_rank(group=cp_group)
        assert cp_group.size() == cp_size

        batch_size = 2
        local_sequence_length = 6
        global_sequence_length = cp_size * local_sequence_length
        max_offset = local_sequence_length
        global_rows = torch.arange(global_sequence_length, device="cuda")
        global_ids = torch.stack((global_rows + 10, global_rows + 110)).to(torch.long)
        global_scores = torch.stack(
            (global_rows.to(torch.float32) + 0.25, global_rows.to(torch.float32) + 100.25)
        )
        local_slice = slice(cp_rank * local_sequence_length, (cp_rank + 1) * local_sequence_length)
        local_ids = global_ids[:, local_slice].contiguous()
        local_scores = global_scores[:, local_slice].contiguous()
        source_ids = local_ids.clone()
        source_scores = local_scores.clone()

        # Three physical sequences [0, 5), [5, 9), [9, 12). The middle sequence
        # crosses the CP boundary, while both ranks also own a local sequence end.
        cu_seqlens = torch.tensor([0, 5, 9, 12], dtype=torch.int32, device="cuda")
        packed_seq_params = PackedSeqParams(
            cu_seqlens_q=cu_seqlens,
            cu_seqlens_kv=cu_seqlens,
            cu_seqlens_q_padded=cu_seqlens,
            cu_seqlens_kv_padded=cu_seqlens,
            max_seqlen_q=5,
            max_seqlen_kv=5,
            qkv_format="thd",
            cp_partition_mode="contiguous",
        )
        bare_context = prepare_mtp_sequence_roll_context(local_ids, cp_group, packed_seq_params)
        assert isinstance(bare_context, ContiguousPackedCPRollContext)

        batch_isend_irecv = torch.distributed.batch_isend_irecv
        grouped_p2p_calls = 0
        grouped_p2p_op_counts = []

        def counted_batch_isend_irecv(p2p_ops):
            nonlocal grouped_p2p_calls
            grouped_p2p_calls += 1
            grouped_p2p_op_counts.append(len(p2p_ops))
            return batch_isend_irecv(p2p_ops)

        monkeypatch.setattr(torch.distributed, "batch_isend_irecv", counted_batch_isend_irecv)
        fields = [
            MTPSequenceRollField("ids", local_ids, -1, 0, 0),
            MTPSequenceRollField("scores", local_scores, -1, 0, -7.5),
        ]
        if cp_rank % 2:
            fields.reverse()
        context = bare_context.prepare_fields(fields, max_offset=max_offset)

        assert context is not bare_context
        assert context.keys == tuple(field.key for field in fields)
        assert grouped_p2p_calls == 1
        assert grouped_p2p_op_counts == [2]
        assert context.max_offset == local_sequence_length

        expected_ids_by_offset = []
        expected_scores_by_offset = []
        for offset in range(1, max_offset + 1):
            expected_ids = _packed_global_aligned_rows(global_ids, cu_seqlens.tolist(), offset, 0)[
                :, local_slice
            ]
            expected_scores = _packed_global_aligned_rows(
                global_scores, cu_seqlens.tolist(), offset, -7.5
            )[:, local_slice]
            expected_ids_by_offset.append(expected_ids)
            expected_scores_by_offset.append(expected_scores)

            ids_address = context.address("ids", offset)
            scores_address = context.address("scores", offset)
            assert ids_address.row_indices.dtype == torch.int32
            assert scores_address.row_indices.dtype == torch.int32
            assert ids_address.valid_rows.dtype == torch.bool
            assert scores_address.valid_rows.dtype == torch.bool
            assert ids_address.halo is not None
            assert scores_address.halo is not None
            assert torch.equal(_materialize_address(ids_address, 0).permute(1, 0), expected_ids)
            assert torch.equal(
                _materialize_address(scores_address, -7.5).permute(1, 0), expected_scores
            )
            assert torch.equal(context.materialize("ids", offset), expected_ids)
            assert torch.equal(context.materialize("scores", offset), expected_scores)

        assert all(
            torch.equal(actual, expected)
            for actual, expected in zip(context.materialize_all("ids"), expected_ids_by_offset)
        )
        assert all(
            torch.equal(actual, expected)
            for actual, expected in zip(
                context.materialize_all("scores"), expected_scores_by_offset
            )
        )
        # Addressing and materialization are communication-free after prepare().
        assert grouped_p2p_calls == 1

        if cp_rank == cp_size - 1:
            assert torch.equal(context.materialize("ids", max_offset), torch.zeros_like(local_ids))
            assert torch.equal(
                context.materialize("scores", max_offset), torch.full_like(local_scores, -7.5)
            )

        # The established cumulative roll path is an independent distributed
        # oracle. It performs one grouped P2P per depth, unlike the prepared path.
        grouped_p2p_calls = 0
        grouped_p2p_op_counts.clear()
        rolled_ids = local_ids
        rolled_scores = local_scores
        for offset in range(1, max_offset + 1):
            rolled_ids, rolled_scores = roll_tensor(
                [rolled_ids, rolled_scores],
                shifts=-1,
                dims=-1,
                cp_group=cp_group,
                packed_seq_params=packed_seq_params,
                fill_values=[0, -7.5],
                roll_context=bare_context,
            )
            assert torch.equal(rolled_ids, expected_ids_by_offset[offset - 1])
            assert torch.equal(rolled_scores, expected_scores_by_offset[offset - 1])
        assert grouped_p2p_calls == max_offset
        assert grouped_p2p_op_counts == [2] * max_offset
        assert torch.equal(local_ids, source_ids)
        assert torch.equal(local_scores, source_scores)

    def test_cp2_mtp_forward_direct_matches_legacy_roll(self):
        """MTP block pre-alignment is identical to cumulative consumer rolling."""
        cp_size = 2
        Utils.initialize_model_parallel(tensor_model_parallel_size=1, context_parallel_size=cp_size)
        cp_group = get_context_parallel_group()
        cp_rank = torch.distributed.get_rank(group=cp_group)

        batch_size = 2
        local_sequence_length = 6
        global_sequence_length = cp_size * local_sequence_length
        num_depths = 3
        global_rows = torch.arange(global_sequence_length, device="cuda")
        global_ids = torch.stack((global_rows + 10, global_rows + 110)).long()
        global_positions = torch.stack((global_rows, global_rows + 100)).long()
        global_padding = torch.zeros(
            (batch_size, global_sequence_length), dtype=torch.bool, device="cuda"
        )
        global_padding[0, 7] = True
        global_padding[1, 10] = True
        local_slice = slice(cp_rank * local_sequence_length, (cp_rank + 1) * local_sequence_length)
        local_ids = global_ids[:, local_slice].contiguous()
        local_positions = global_positions[:, local_slice].contiguous()
        local_padding = global_padding[:, local_slice].contiguous()

        cu_seqlens = torch.tensor([0, 5, 9, 12], dtype=torch.int32, device="cuda")
        packed_seq_params = PackedSeqParams(
            cu_seqlens_q=cu_seqlens,
            cu_seqlens_kv=cu_seqlens,
            cu_seqlens_q_padded=cu_seqlens,
            cu_seqlens_kv_padded=cu_seqlens,
            max_seqlen_q=5,
            max_seqlen_kv=5,
            qkv_format="thd",
            cp_partition_mode="contiguous",
        )
        bare_context = prepare_mtp_sequence_roll_context(local_ids, cp_group, packed_seq_params)
        assert isinstance(bare_context, ContiguousPackedCPRollContext)
        direct_context = bare_context.prepare_fields(
            (
                MTPSequenceRollField("input_ids", local_ids, -1, 0, 0),
                MTPSequenceRollField("position_ids", local_positions, -1, 0, 0),
                MTPSequenceRollField("padding_mask", local_padding, -1, 0, True),
            ),
            max_offset=num_depths,
        )

        class RollAwareLayer(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.calls = []

            def forward(
                self,
                hidden_states,
                input_ids,
                position_ids,
                padding_mask,
                packed_seq_params,
                sequence_roll_context,
                roll_depth,
                _inputs_pre_aligned=False,
                **kwargs,
            ):
                del kwargs
                if not _inputs_pre_aligned:
                    input_ids, position_ids, padding_mask = roll_tensor(
                        [input_ids, position_ids, padding_mask],
                        shifts=-1,
                        dims=-1,
                        cp_group=cp_group,
                        packed_seq_params=packed_seq_params,
                        fill_values=[0, 0, True],
                        roll_context=sequence_roll_context,
                        sequence_fields=["input_ids", "position_ids", "padding_mask"],
                        roll_depth=roll_depth,
                    )
                self.calls.append(
                    (
                        input_ids.clone(),
                        position_ids.clone(),
                        padding_mask.clone(),
                        _inputs_pre_aligned,
                    )
                )
                aligned_value = input_ids.transpose(0, 1).unsqueeze(-1).to(hidden_states.dtype)
                aligned_value = aligned_value + position_ids.transpose(0, 1).unsqueeze(-1).to(
                    hidden_states.dtype
                )
                hidden_states = hidden_states + aligned_value
                return hidden_states, input_ids, position_ids, padding_mask

        def run(sequence_roll_context, sequence_roll_padding_mask=None):
            config = SimpleNamespace(
                pipeline_model_parallel_size=1, mtp_num_layers=num_depths, mtp_detach_heads=False
            )
            block = MultiTokenPredictionBlock.__new__(MultiTokenPredictionBlock)
            torch.nn.Module.__init__(block)
            block.config = config
            block.vp_stage = None
            block.mtp_use_repeated_layer = False
            block.layers = torch.nn.ModuleList(RollAwareLayer() for _ in range(num_depths))
            output = block.forward(
                input_ids=local_ids,
                position_ids=local_positions,
                hidden_states=torch.zeros(local_sequence_length, batch_size, 1, device="cuda"),
                attention_mask=torch.empty(0, device="cuda"),
                padding_mask=local_padding,
                sequence_roll_padding_mask=sequence_roll_padding_mask,
                packed_seq_params=packed_seq_params,
                sequence_roll_context=sequence_roll_context,
                embedding=SimpleNamespace(add_position_embedding=True),
            )
            calls = [layer.calls[0] for layer in block.layers]
            return output, calls

        legacy_output, legacy_calls = run(bare_context)
        direct_output, direct_calls = run(direct_context, local_padding)
        assert torch.equal(direct_output, legacy_output)
        for depth, (legacy_call, direct_call) in enumerate(
            zip(legacy_calls, direct_calls), start=1
        ):
            expected_ids = _packed_global_aligned_rows(global_ids, cu_seqlens.tolist(), depth, 0)[
                :, local_slice
            ]
            expected_positions = _packed_global_aligned_rows(
                global_positions, cu_seqlens.tolist(), depth, 0
            )[:, local_slice]
            expected_padding = _packed_global_aligned_rows(
                global_padding, cu_seqlens.tolist(), depth, True
            )[:, local_slice]
            assert not legacy_call[3]
            assert direct_call[3]
            for legacy_value, direct_value, expected in zip(
                legacy_call[:3],
                direct_call[:3],
                (expected_ids, expected_positions, expected_padding),
            ):
                assert torch.equal(legacy_value, expected)
                assert torch.equal(direct_value, expected)

    @pytest.mark.parametrize("derived_labels", [False, True], ids=["sft", "rl"])
    def test_cp2_ce_direct_matches_legacy_roll(self, derived_labels):
        """SFT and RL CE consumers preserve labels, masks, output, and gradients."""
        cp_size = 2
        Utils.initialize_model_parallel(tensor_model_parallel_size=1, context_parallel_size=cp_size)
        cp_group = get_context_parallel_group()
        cp_rank = torch.distributed.get_rank(group=cp_group)

        batch_size = 2
        local_sequence_length = 6
        global_sequence_length = cp_size * local_sequence_length
        num_depths = 3
        hidden_size = 4
        global_rows = torch.arange(global_sequence_length, device="cuda")
        global_input_ids = torch.stack((global_rows + 10, global_rows + 110)).long()
        global_labels = torch.stack((global_rows + 1010, global_rows + 1110)).long()
        global_loss_mask = torch.ones(
            (batch_size, global_sequence_length), dtype=torch.float32, device="cuda"
        )
        global_loss_mask[0, 3] = 0
        global_loss_mask[1, 7] = 0
        local_slice = slice(cp_rank * local_sequence_length, (cp_rank + 1) * local_sequence_length)
        local_input_ids = global_input_ids[:, local_slice].contiguous()
        local_labels = global_labels[:, local_slice].contiguous()
        local_loss_mask = global_loss_mask[:, local_slice].contiguous()

        cu_seqlens = torch.tensor([0, 5, 9, 12], dtype=torch.int32, device="cuda")
        packed_seq_params = PackedSeqParams(
            cu_seqlens_q=cu_seqlens,
            cu_seqlens_kv=cu_seqlens,
            cu_seqlens_q_padded=cu_seqlens,
            cu_seqlens_kv_padded=cu_seqlens,
            max_seqlen_q=5,
            max_seqlen_kv=5,
            qkv_format="thd",
            cp_partition_mode="contiguous",
        )
        reference = local_input_ids if derived_labels else local_labels
        bare_context = prepare_mtp_sequence_roll_context(reference, cp_group, packed_seq_params)
        assert isinstance(bare_context, ContiguousPackedCPRollContext)

        config = TransformerConfig(
            hidden_size=hidden_size, num_layers=2, num_attention_heads=2, mtp_num_layers=num_depths
        )
        config.mtp_loss_scaling_factor = 1.0
        hidden_template = (
            torch.arange(
                (1 + num_depths) * local_sequence_length * batch_size * hidden_size,
                dtype=torch.float32,
                device="cuda",
            ).view((1 + num_depths) * local_sequence_length, batch_size, hidden_size)
            / 100.0
            + 1.0
        )

        class OutputLayer:
            gather_output = True

            def __call__(self, hidden, weight=None, runtime_gather_output=None):
                del weight, runtime_gather_output
                return hidden, None

        def run(sequence_roll_context):
            hidden = hidden_template.clone().requires_grad_(True)
            seen_labels = []

            def compute_language_model_loss(current_labels, logits):
                seen_labels.append(current_labels.clone())
                return (
                    logits.square().mean(dim=-1).transpose(0, 1)
                    + current_labels.to(logits.dtype) * 1.0e-4
                )

            output = process_mtp_loss(
                hidden_states=hidden,
                labels=None if derived_labels else local_labels,
                loss_mask=local_loss_mask,
                output_layer=OutputLayer(),
                output_weight=None,
                runtime_gather_output=None,
                is_training=False,
                compute_language_model_loss=compute_language_model_loss,
                config=config,
                cp_group=cp_group,
                packed_seq_params=packed_seq_params,
                input_ids=local_input_ids if derived_labels else None,
                sequence_roll_context=sequence_roll_context,
            )
            output.sum().backward()
            return output.detach(), hidden.grad.detach(), seen_labels

        legacy_output, legacy_grad, legacy_labels = run(None)
        direct_output, direct_grad, direct_labels = run(bare_context)
        assert torch.equal(direct_output, legacy_output)
        assert torch.equal(direct_grad, legacy_grad)
        assert len(direct_labels) == num_depths

        global_label_source = global_input_ids if derived_labels else global_labels
        first_offset = 2 if derived_labels else 1
        direct_grad_chunks = torch.chunk(direct_grad, 1 + num_depths, dim=0)
        for depth, (legacy_current, direct_current) in enumerate(zip(legacy_labels, direct_labels)):
            offset = first_offset + depth
            expected_labels = _packed_global_aligned_rows(
                global_label_source, cu_seqlens.tolist(), offset, 0
            )[:, local_slice]
            expected_mask = _packed_global_aligned_rows(
                global_loss_mask, cu_seqlens.tolist(), offset, 0
            )[:, local_slice]
            assert torch.equal(legacy_current, expected_labels)
            assert torch.equal(direct_current, expected_labels)
            active_grad = direct_grad_chunks[depth + 1].abs().sum(dim=-1).transpose(0, 1) > 0
            assert torch.equal(active_grad, expected_mask.bool())


@pytest.mark.parametrize("sequence_parallel", [False, True])
@pytest.mark.parametrize("has_padding_mask", [False, True])
@pytest.mark.parametrize("prepared", [False, True], ids=["legacy", "prepared"])
def test_mtp_block_scatters_only_prepared_global_padding_masks(
    monkeypatch, sequence_parallel, has_padding_mask, prepared
):
    """Prepared rows scatter after alignment; legacy keeps its existing local mask."""
    sequence_length = 8
    local_sequence_length = 4 if sequence_parallel else sequence_length
    input_ids = torch.arange(sequence_length).view(1, sequence_length)
    position_ids = input_ids.clone()
    raw_padding_mask = (
        torch.tensor([[False, False, False, False, False, False, True, True]])
        if has_padding_mask
        else None
    )
    local_padding_mask = (
        raw_padding_mask[:, :local_sequence_length].contiguous()
        if raw_padding_mask is not None
        else None
    )
    sequence_roll_context = None
    if prepared:
        bare_context = prepare_mtp_sequence_roll_context(input_ids, None, None)
        fields = [
            MTPSequenceRollField("input_ids", input_ids, -1, 0, 0),
            MTPSequenceRollField("position_ids", position_ids, -1, 0, 0),
        ]
        if raw_padding_mask is not None:
            fields.append(MTPSequenceRollField("padding_mask", raw_padding_mask, -1, 0, True))
        sequence_roll_context = bare_context.prepare_fields(fields, max_offset=1)

    tp_group = object()
    scatter_calls = []

    def fake_scatter(mask, group=None):
        scatter_calls.append((mask.clone(), group))
        return mask[:local_sequence_length]

    monkeypatch.setattr(
        "megatron.core.transformer.multi_token_prediction.tensor_parallel."
        "scatter_to_sequence_parallel_region",
        fake_scatter,
    )

    class RecordingLayer(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.tp_group = tp_group
            self.call = None

        def forward(self, hidden_states, padding_mask=None, **kwargs):
            self.call = (
                padding_mask,
                kwargs.get("_inputs_pre_aligned", False),
                "_inputs_pre_aligned" in kwargs,
            )
            return (hidden_states, kwargs["input_ids"], kwargs["position_ids"], padding_mask)

    config = SimpleNamespace(
        pipeline_model_parallel_size=1,
        mtp_num_layers=1,
        mtp_detach_heads=False,
        sequence_parallel=sequence_parallel,
    )
    block = MultiTokenPredictionBlock.__new__(MultiTokenPredictionBlock)
    torch.nn.Module.__init__(block)
    block.config = config
    block.vp_stage = None
    block.mtp_use_repeated_layer = False
    block.layers = torch.nn.ModuleList((RecordingLayer(),))

    block.forward(
        input_ids=input_ids,
        position_ids=position_ids,
        hidden_states=torch.zeros(sequence_length, 1, 1),
        attention_mask=torch.empty(0),
        padding_mask=local_padding_mask,
        sequence_roll_padding_mask=raw_padding_mask if prepared else None,
        sequence_roll_context=sequence_roll_context,
        embedding=SimpleNamespace(add_position_embedding=True),
    )

    seen_padding_mask, inputs_pre_aligned, has_pre_aligned_kwarg = block.layers[0].call
    assert inputs_pre_aligned is prepared
    assert has_pre_aligned_kwarg is prepared
    if not has_padding_mask:
        assert seen_padding_mask is None
        assert scatter_calls == []
    elif not prepared:
        assert seen_padding_mask is local_padding_mask
        assert scatter_calls == []
    else:
        expected_global = torch.ones_like(raw_padding_mask)
        expected_global[:, :-1] = raw_padding_mask[:, 1:]
        expected_local = expected_global[:, :local_sequence_length]
        assert torch.equal(seen_padding_mask, expected_local)
        if sequence_parallel:
            assert len(scatter_calls) == 1
            assert scatter_calls[0][1] is tp_group
            assert torch.equal(scatter_calls[0][0], expected_global.transpose(0, 1))
        else:
            assert scatter_calls == []
