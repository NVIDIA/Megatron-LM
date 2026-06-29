# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
from __future__ import annotations

import pytest
import torch
from megatron.lite.primitive.parallel import (
    ParallelState,
    build_pipeline_chunk_layout,
    pack_nested_thd,
    parallel_state_from_model,
    prepare_packed_thd_for_context_parallel,
    reconstruct_packed_from_cp_parts,
    roll_packed_thd_left,
    split_packed_to_cp_local,
    zigzag_position_ids_for_cp,
    zigzag_reconstruct_from_cp_parts,
    zigzag_slice_for_cp,
    zigzag_split_for_cp,
)

pytestmark = pytest.mark.mlite


def test_cp_zigzag_split_slice_and_reconstruct_match():
    tensor = torch.arange(16).reshape(1, 8, 2)
    parts = [zigzag_split_for_cp(tensor, rank, cp_size=2, seq_dim=1) for rank in range(2)]

    assert torch.equal(parts[0], tensor[:, [0, 1, 6, 7], :])
    assert torch.equal(parts[1], tensor[:, [2, 3, 4, 5], :])
    assert torch.equal(zigzag_slice_for_cp(tensor, 0, cp_size=2, seq_dim=1), parts[0])
    assert torch.equal(zigzag_slice_for_cp(tensor, 1, cp_size=2, seq_dim=1), parts[1])
    assert torch.equal(zigzag_reconstruct_from_cp_parts(parts, seq_dim=1), tensor)


def test_cp_position_ids_follow_zigzag_order():
    assert torch.equal(
        zigzag_position_ids_for_cp(8, cp_rank=0, cp_size=2, device=torch.device("cpu")),
        torch.tensor([[0, 1, 6, 7]]),
    )
    assert torch.equal(
        zigzag_position_ids_for_cp(8, cp_rank=1, cp_size=2, device=torch.device("cpu")),
        torch.tensor([[2, 3, 4, 5]]),
    )


# The pipeline layout wires to Megatron-core's layer-layout machinery, so these
# tests need megatron.core importable (present in the mlite GPU/smoke containers).
_mcore_layout = pytest.importorskip(
    "megatron.core.transformer.pipeline_parallel_layer_layout"
)


def _ranks(pp_size: int, pp_layout=None) -> list[ParallelState]:
    return [
        ParallelState(
            pp_size=pp_size,
            pp_rank=r,
            pp_is_first=(r == 0),
            pp_is_last=(r == pp_size - 1),
            pp_layout=pp_layout,
        )
        for r in range(pp_size)
    ]


def _layout_indices(num_layers: int, pp_size: int, *, pp_layout=None, **kw) -> list[list[int]]:
    return [
        build_pipeline_chunk_layout(num_layers, ps, **kw).layer_indices
        for ps in _ranks(pp_size, pp_layout)
    ]


def _mtp_flags(num_layers: int, pp_size: int, *, pp_layout=None, **kw) -> list[bool]:
    return [
        build_pipeline_chunk_layout(num_layers, ps, **kw).has_mtp
        for ps in _ranks(pp_size, pp_layout)
    ]


def _assert_full_contiguous_cover(indices: list[list[int]], num_layers: int) -> None:
    """Every decoder 0..N-1 placed exactly once, in order, each stage a contiguous run."""
    assert [i for stage in indices for i in stage] == list(range(num_layers))
    for stage in indices:
        assert not stage or stage == list(range(stage[0], stage[0] + len(stage)))


def test_pp_layout_marks_stage_boundaries():
    rank0, rank1 = _ranks(2)
    assert build_pipeline_chunk_layout(8, rank0).layer_indices == [0, 1, 2, 3]
    assert build_pipeline_chunk_layout(8, rank0).has_embed is True
    assert build_pipeline_chunk_layout(8, rank0).has_head is False
    assert build_pipeline_chunk_layout(8, rank1).layer_indices == [4, 5, 6, 7]
    assert build_pipeline_chunk_layout(8, rank1).has_embed is False
    assert build_pipeline_chunk_layout(8, rank1).has_head is True


def test_pp_layout_vpp_is_not_supported_yet():
    # pp-only: requesting VPP raises rather than silently mis-splitting.
    rank0 = ParallelState(pp_size=2, pp_rank=0, pp_is_first=True, pp_is_last=False)
    with pytest.raises(NotImplementedError):
        build_pipeline_chunk_layout(8, rank0, vpp=2)
    with pytest.raises(NotImplementedError):
        build_pipeline_chunk_layout(8, rank0, vpp=2, vpp_chunk_id=1)


def test_pp_layout_auto_balances_non_divisible_counts():
    # account_for_embedding/loss: the embedding/loss stages each give up a decoder,
    # so 6/pp4 balances to [1,2,2,1] (not [2,2,1,1]) and never raises "not divisible".
    assert _layout_indices(6, 4) == [[0], [1, 2], [3, 4], [5]]
    assert _layout_indices(6, 4, vpp=1) == _layout_indices(6, 4)  # vpp=1 == pp-only
    # An MTP head occupies the last stage's slot -> it gives up its decoder: [2,2,2,0].
    assert _layout_indices(6, 4, num_mtp_layers=1) == [[0, 1], [2, 3], [4, 5], []]


def test_pp_layout_accounts_for_head_tail_even_when_divisible():
    # Embedding/loss occupy head/tail slots, so 8/pp4 is [2,3,2,1] not [2,2,2,2].
    assert _layout_indices(8, 4) == [[0, 1], [2, 3, 4], [5, 6], [7]]
    assert _layout_indices(8, 4, num_mtp_layers=1) == [[0, 1], [2, 3, 4], [5, 6, 7], []]
    assert _layout_indices(8, 2) == [[0, 1, 2, 3], [4, 5, 6, 7]]


# --- Correctness matrix: real delivery counts x MTP{off,on} x PP{2,4,8} ---
_REAL_LAYERS = {"deepseek_v4": 43, "kimi_k2": 61, "glm5": 78}
_CORRECTNESS_CASES = [(m, pp, mtp) for m in _REAL_LAYERS for pp in (2, 4, 8) for mtp in (0, 1)]


def _mcore_reference_decoder_ids(num_layers: int, pp: int, mtp: int) -> list[list[int]]:
    """Per-stage decoder ids from mcore's PipelineParallelLayerLayout, built directly
    from the canonical unit sequence — the independent reference the glue must match."""
    from megatron.core.transformer.enums import LayerType
    from megatron.core.transformer.pipeline_parallel_layer_layout import (
        PipelineParallelLayerLayout,
    )

    units = ["embedding"] + ["decoder"] * num_layers + ["mtp"] * mtp + ["loss"]
    base, rem = divmod(len(units), pp)
    rows, pos = [], 0
    for size in (base + (1 if s < rem else 0) for s in range(pp)):
        rows.append(units[pos : pos + size])
        pos += size
    ref = PipelineParallelLayerLayout(rows, pipeline_model_parallel_size=pp)
    return [ref.get_layer_id_list(LayerType.decoder, vp_stage=0, pp_rank=r) for r in range(pp)]


@pytest.mark.parametrize("model, pp, mtp", _CORRECTNESS_CASES)
def test_auto_layout_is_bit_equal_to_mcore(model, pp, mtp):
    # Faithful reuse: the glue's per-stage decoder ids are exactly mcore's for the same
    # canonical layout, so correctness is inherited from mcore (not re-derived).
    n = _REAL_LAYERS[model]
    assert _layout_indices(n, pp, num_mtp_layers=mtp) == _mcore_reference_decoder_ids(n, pp, mtp)


@pytest.mark.parametrize("model, pp, mtp", _CORRECTNESS_CASES)
def test_auto_layout_is_legal_and_balanced(model, pp, mtp):
    n = _REAL_LAYERS[model]
    chunks = [build_pipeline_chunk_layout(n, ps, num_mtp_layers=mtp) for ps in _ranks(pp)]
    # complete + ordered, no missing/dup decoder -> sum == num_layers
    assert [i for c in chunks for i in c.layer_indices] == list(range(n))
    # embedding only on stage 0; loss/head only on the last stage
    assert [c.has_embed for c in chunks] == [r == 0 for r in range(pp)]
    assert [c.has_head for c in chunks] == [r == pp - 1 for r in range(pp)]
    # MTP placed exactly `mtp` times, on the final (head) stage
    assert sum(c.has_mtp for c in chunks) == mtp and chunks[-1].has_mtp == bool(mtp)
    # balanced: per-stage unit cells (decoders + embedding/loss/mtp slots) differ by <=1
    cells = [len(c.layer_indices) + c.has_embed + c.has_head + (mtp if c.has_mtp else 0) for c in chunks]
    assert max(cells) - min(cells) <= 1


def test_pp_layout_custom_string_mode_is_used_verbatim():
    # Advanced users pin an explicit mcore layout string (custom mode) instead of the
    # auto split. "Ettt|t|t|tL" front-loads 3 decoders onto stage 0; mcore parses
    # E=embedding t=decoder L=loss. This is NOT the auto [1,2,2,1] for 6/pp4.
    custom = _layout_indices(6, 4, pp_layout="Ettt|t|t|tL")
    assert [len(stage) for stage in custom] == [3, 1, 1, 1]
    _assert_full_contiguous_cover(custom, 6)
    assert custom != _layout_indices(6, 4)  # custom overrides auto

    # A custom string carrying MTP after the decoders validates against num_mtp_layers.
    mtp_custom = _layout_indices(6, 4, pp_layout="Ett|tt|t|tmL", num_mtp_layers=1)
    assert [len(stage) for stage in mtp_custom] == [2, 2, 1, 1]
    _assert_full_contiguous_cover(mtp_custom, 6)


def test_pp_layout_custom_string_rejects_vpp_multi_chunk():
    # A custom layout with more stages than pp implies VPP, which is not supported.
    with pytest.raises(NotImplementedError):
        _layout_indices(6, 2, pp_layout="Et|t|t|ttL")  # 4 stages over pp2 -> vpp=2


def test_pp_layout_marks_mtp_stage_from_the_layout_not_a_fixed_rank():
    # auto: the MTP slot sits just before loss, so it lands on the final (head) stage
    # -- has_mtp marks exactly that stage, and shifts the decoder split [2,2,2,0].
    assert _mtp_flags(6, 4, num_mtp_layers=1) == [False, False, False, True]
    assert _layout_indices(6, 4, num_mtp_layers=1) == [[0, 1], [2, 3], [4, 5], []]
    # no MTP requested -> no stage owns MTP.
    assert _mtp_flags(6, 4) == [False, False, False, False]
    # single stage owns the MTP too.
    assert build_pipeline_chunk_layout(
        5, ParallelState(pp_size=1, pp_rank=0, pp_is_first=True, pp_is_last=True), num_mtp_layers=1
    ).has_mtp is True


def test_pp_layout_custom_string_mtp_lands_on_the_designated_stage():
    # The `m` token co-located with loss on the final stage: has_mtp marks that stage
    # (== the head stage), driven by the layout, not a hard-coded rank.
    flags = _mtp_flags(6, 4, pp_layout="Ett|tt|t|tmL", num_mtp_layers=1)
    assert flags == [False, False, False, True]
    assert _layout_indices(6, 4, pp_layout="Ett|tt|t|tmL", num_mtp_layers=1) == [
        [0, 1], [2, 3], [4], [5]
    ]


@pytest.mark.xfail(
    reason="Standalone MTP (a custom pp_layout placing `m` off the final/head stage) is a "
    "planned follow-up: mlite's MTP shares the output head on the loss stage, so it "
    "currently raises NotImplementedError instead of building MTP on the `m` stage. "
    "When cross-stage standalone MTP lands, this should pass and the marker be removed.",
    strict=True,
    raises=NotImplementedError,
)
def test_pp_layout_custom_string_standalone_mtp_builds_on_designated_stage():
    # DESIRED (follow-up): `m` on a non-final stage builds MTP on that stage.
    # "E|ttttttm|L" over pp3 -> [E], [t*6, m], [L]; MTP belongs on the middle stage.
    assert _mtp_flags(6, 3, pp_layout="E|ttttttm|L", num_mtp_layers=1) == [False, True, False]


def test_pp_layout_single_stage_owns_all_layers():
    ps = ParallelState(pp_size=1, pp_rank=0, pp_is_first=True, pp_is_last=True)
    layout = build_pipeline_chunk_layout(5, ps)
    assert layout.layer_indices == [0, 1, 2, 3, 4]
    assert layout.has_embed and layout.has_head


def test_virtual_pipeline_rank_is_tracked_on_lite_parallel_state():
    from megatron.lite.primitive.parallel.pipeline import _set_virtual_pipeline_rank

    ps = ParallelState(pp_size=2, pp_rank=1, pp_is_first=False, pp_is_last=True)

    _set_virtual_pipeline_rank(ps, chunk_id=1, num_chunks=2)

    assert ps.virtual_pipeline_size == 2
    assert ps.virtual_pipeline_rank == 1

    _set_virtual_pipeline_rank(ps, chunk_id=None, num_chunks=2)

    assert ps.virtual_pipeline_size is None
    assert ps.virtual_pipeline_rank is None


def test_thd_roll_keeps_sequence_boundaries():
    cu_seqlens = torch.tensor([0, 4, 8], dtype=torch.int32)
    rolled, token_sum = roll_packed_thd_left(torch.arange(8), cu_seqlens_padded=cu_seqlens, dims=0)

    assert torch.equal(rolled, torch.tensor([1, 2, 3, 0, 5, 6, 7, 0]))
    assert token_sum.item() == 24


def test_thd_cp_split_and_reconstruct_roundtrip():
    cu_seqlens = torch.tensor([0, 8], dtype=torch.int32)
    tensor = torch.arange(8)
    parts = [
        split_packed_to_cp_local(
            tensor, cu_seqlens_padded=cu_seqlens, cp_size=2, cp_rank=rank, dim=0
        )
        for rank in range(2)
    ]

    assert torch.equal(parts[0], torch.tensor([0, 1, 6, 7]))
    assert torch.equal(parts[1], torch.tensor([2, 3, 4, 5]))
    assert torch.equal(
        reconstruct_packed_from_cp_parts(parts, cu_seqlens_padded=cu_seqlens, cp_size=2, dim=0),
        tensor,
    )


def test_plain_thd_batch_is_split_by_protocol_context_parallel_helper():
    ids = torch.nested.as_nested_tensor(
        [torch.arange(1, 6), torch.arange(11, 18)],
        layout=torch.jagged,
    )
    labels = torch.nested.as_nested_tensor(
        [torch.arange(101, 106), torch.arange(111, 118)],
        layout=torch.jagged,
    )
    loss_mask = torch.nested.as_nested_tensor(
        [torch.ones(5), torch.ones(7)],
        layout=torch.jagged,
    )
    packed = pack_nested_thd(
        ids,
        cp_size=2,
        split_cp=False,
        labels=labels,
        loss_mask=loss_mask,
    )

    assert packed.input_ids.shape == (1, 16)
    assert packed.cp_size == 2
    assert packed.packed_seq_params.local_cp_size is None

    local_params, local_tensors = prepare_packed_thd_for_context_parallel(
        packed.packed_seq_params,
        (packed.input_ids, packed.labels, packed.loss_mask, packed.position_ids),
        cp_size=2,
        cp_rank=0,
    )

    expected_ids = split_packed_to_cp_local(
        packed.input_ids,
        cu_seqlens_padded=packed.cu_seqlens_padded,
        cp_size=2,
        cp_rank=0,
        dim=1,
    )
    expected_pos = split_packed_to_cp_local(
        packed.position_ids,
        cu_seqlens_padded=packed.cu_seqlens_padded,
        cp_size=2,
        cp_rank=0,
        dim=1,
    )
    local_ids, local_labels, local_loss_mask, local_pos = local_tensors
    assert torch.equal(local_ids, expected_ids)
    assert torch.equal(local_pos, expected_pos)
    assert local_labels is not None
    assert local_loss_mask is not None
    assert local_params.local_cp_size == 2
    assert local_params.cp_rank == 0


def test_parallel_state_from_model_unwraps_ddp_style_module():
    class Model:
        ps = ParallelState(cp_size=2, cp_rank=1)

    class Wrapper:
        module = Model()

    assert parallel_state_from_model(Wrapper()).cp_rank == 1


def test_protocol_context_parallel_helper_keeps_pre_split_thd_batch_idempotent():
    ids = torch.nested.as_nested_tensor([torch.arange(8)], layout=torch.jagged)
    packed = pack_nested_thd(ids, cp_size=2, cp_rank=1)

    local_params, local_tensors = prepare_packed_thd_for_context_parallel(
        packed.packed_seq_params,
        (packed.input_ids, packed.position_ids),
        cp_size=2,
        cp_rank=1,
    )

    assert local_params is packed.packed_seq_params
    assert torch.equal(local_tensors[0], packed.input_ids)
    assert local_params.local_cp_size == 2


def test_protocol_context_parallel_helper_is_noop_without_packed_thd_params():
    tensor = torch.arange(8)

    local_params, local_tensors = prepare_packed_thd_for_context_parallel(
        None,
        (tensor,),
        cp_size=2,
        cp_rank=0,
    )

    assert local_params is None
    assert local_tensors[0] is tensor
