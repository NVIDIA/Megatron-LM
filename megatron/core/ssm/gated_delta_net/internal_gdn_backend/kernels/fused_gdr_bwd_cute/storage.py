"""Shared-memory and tensor-memory resource contracts for fused GDR backward."""

from dataclasses import dataclass
from typing import Any

import cutlass
import cutlass.cute as cute

_BT = 64
_TMA_VECTOR_HEADS = 4
_DK = 128
_DV = 128
_THREADS_PER_CTA = 384
_BUFFER_ALIGN_BYTES = 128
_SM100_OPTIN_SMEM_BYTES = 232_448

TMEM_COLUMNS = 480
TMEM_ALLOCATION_COLUMNS = (256, 128, 32, 64)
TMEM_RANGES = (
    # name, first column, one-past-last column, first phase, last phase
    ("p", 0, 32, 0, 7),
    ("dk", 0, 64, 7, 16),
    ("dv", 64, 128, 1, 6),
    ("u", 192, 256, 3, 6),
    ("dq", 192, 256, 8, 11),
    ("vprime", 416, 480, 5, 6),
    ("dog", 416, 480, 6, 14),
    ("dh_left", 256, 320, 0, 16),
    ("da", 128, 160, 6, 14),
    ("a", 160, 192, 5, 6),
    ("dp", 160, 192, 6, 8),
    ("at", 160, 192, 13, 14),
    ("dh_right", 320, 384, 0, 16),
    ("mask", 384, 416, 0, 16),
)

# Keep the extracted declarations source-compatible with the original kernel.
_TMEM_COLUMNS = TMEM_COLUMNS
_TMEM_ALLOCATION_COLUMNS = TMEM_ALLOCATION_COLUMNS
_TMEM_RANGES = TMEM_RANGES


@dataclass(frozen=True)
class LayoutBudget:
    """Physical CTA storage required by the structured kernel shell."""

    smem_bytes: int
    tmem_columns: int
    threads: int


def _validate_tmem_ranges() -> None:
    """Prove that spatial aliases have disjoint completion phases."""

    assert sum(TMEM_ALLOCATION_COLUMNS) == TMEM_COLUMNS
    assert all(
        columns > 0 and columns & (columns - 1) == 0
        for columns in TMEM_ALLOCATION_COLUMNS
    )
    for name, begin, end, phase_begin, phase_end in TMEM_RANGES:
        assert 0 <= begin < end <= TMEM_COLUMNS, name
        assert 0 <= phase_begin < phase_end <= 16, name
    for index, lhs in enumerate(TMEM_RANGES):
        for rhs in TMEM_RANGES[index + 1 :]:
            lhs_name, lhs_begin, lhs_end, lhs_phase, lhs_phase_end = lhs
            rhs_name, rhs_begin, rhs_end, rhs_phase, rhs_phase_end = rhs
            columns_overlap = max(lhs_begin, rhs_begin) < min(
                lhs_end, rhs_end
            )
            phases_overlap = max(lhs_phase, rhs_phase) < min(
                lhs_phase_end, rhs_phase_end
            )
            assert not (columns_overlap and phases_overlap), (
                f"live TMEM aliases overlap: {lhs_name} and {rhs_name}"
            )
    assert max(allocation[2] for allocation in TMEM_RANGES) <= TMEM_COLUMNS


_validate_tmem_ranges()


@cute.struct
class SharedStorage:
    """Non-overlaid live SMEM and independent one-stage pipeline storage."""

    # Seven runtime load edges coordinate eight logical input tensors: q and k
    # share one TMA transaction edge, while the other six own one edge each.
    load_qk_mbar_ptr: cute.struct.MemRange[cutlass.Int64, 2]
    load_v_mbar_ptr: cute.struct.MemRange[cutlass.Int64, 2]
    load_g_mbar_ptr: cute.struct.MemRange[cutlass.Int64, 2]
    load_h_mbar_ptr: cute.struct.MemRange[cutlass.Int64, 2]
    load_a_mbar_ptr: cute.struct.MemRange[cutlass.Int64, 2]
    load_do_mbar_ptr: cute.struct.MemRange[cutlass.Int64, 2]
    load_beta_mbar_ptr: cute.struct.MemRange[cutlass.Int64, 2]

    # AQ publishes one complete shared dQ tile to the TMA warp. The TMA warp
    # releases the empty phase only after cp.async.bulk.wait_group proves the
    # shared source is no longer in use.
    dq_smem_tile_ready_mbar_ptr: cute.struct.MemRange[cutlass.Int64, 2]
    # SK and AQ publish that all current-iteration TMEM reads are done before
    # the MMA warp overwrites persistent accumulator ranges next iteration.
    all_tmem_readers_done_mbar_ptr: cute.struct.MemRange[cutlass.Int64, 2]
    # AQ publishes completion of the W computation after its final U and V
    # reads. SK uses that one result before overwriting either allocation.
    aq_w_inputs_read_done_mbar_ptr: cute.struct.MemRange[cutlass.Int64, 2]
    # SK publishes the materialized dO-gamma shared tile consumed by AQ.
    dog_smem_ready_mbar_ptr: cute.struct.MemRange[cutlass.Int64, 2]
    # SK publishes completion of every dK-derived sDg update before AQ
    # merges q*dQ. The earlier write ownership has its own dedicated edge.
    dg_rmw_done_mbar_ptr: cute.struct.MemRange[cutlass.Int64, 2]

    # Cross-iteration physical-buffer reuse permissions. These are deliberately
    # independent from compute-ready stage edges: each becomes full only after
    # the last reader of that exact SMEM allocation has completed.
    reuse_v_tma_mbar_ptr: cute.struct.MemRange[cutlass.Int64, 2]
    reuse_a_tma_mbar_ptr: cute.struct.MemRange[cutlass.Int64, 2]
    reuse_do_tma_mbar_ptr: cute.struct.MemRange[cutlass.Int64, 2]
    reuse_beta_tma_mbar_ptr: cute.struct.MemRange[cutlass.Int64, 2]
    # SK publishes after the final stage-14 readers release sTmp21; AQ waits on
    # the previous iteration before materializing the next dVprime tile.
    reuse_tmp21_aq_mbar_ptr: cute.struct.MemRange[cutlass.Int64, 2]

    # One empty/full mbarrier pair per pinned-schedule MMA dependency.
    mma_p_done_mbar_ptr: cute.struct.MemRange[cutlass.Int64, 2]
    mma_dvprime_state_done_mbar_ptr: cute.struct.MemRange[cutlass.Int64, 2]
    mma_dvprime_pgamma_done_mbar_ptr: cute.struct.MemRange[cutlass.Int64, 2]
    mma_u_state_done_mbar_ptr: cute.struct.MemRange[cutlass.Int64, 2]
    mma_dv_done_mbar_ptr: cute.struct.MemRange[cutlass.Int64, 2]
    mma_vprime_done_mbar_ptr: cute.struct.MemRange[cutlass.Int64, 2]
    mma_dag_done_mbar_ptr: cute.struct.MemRange[cutlass.Int64, 2]
    mma_dpg_done_mbar_ptr: cute.struct.MemRange[cutlass.Int64, 2]
    mma_dk_state_done_mbar_ptr: cute.struct.MemRange[cutlass.Int64, 2]
    mma_dq_state_done_mbar_ptr: cute.struct.MemRange[cutlass.Int64, 2]
    mma_dk_state_g_done_mbar_ptr: cute.struct.MemRange[cutlass.Int64, 2]
    mma_dq_dp_done_mbar_ptr: cute.struct.MemRange[cutlass.Int64, 2]
    mma_dh_k_left_done_mbar_ptr: cute.struct.MemRange[cutlass.Int64, 2]
    mma_dh_k_right_done_mbar_ptr: cute.struct.MemRange[cutlass.Int64, 2]
    mma_dk_dp_done_mbar_ptr: cute.struct.MemRange[cutlass.Int64, 2]
    mma_da_left_done_mbar_ptr: cute.struct.MemRange[cutlass.Int64, 2]
    mma_da_right_done_mbar_ptr: cute.struct.MemRange[cutlass.Int64, 2]
    mma_at_done_mbar_ptr: cute.struct.MemRange[cutlass.Int64, 2]
    mma_dh_q_left_done_mbar_ptr: cute.struct.MemRange[cutlass.Int64, 2]
    mma_dh_q_right_done_mbar_ptr: cute.struct.MemRange[cutlass.Int64, 2]
    mma_dk_da_done_mbar_ptr: cute.struct.MemRange[cutlass.Int64, 2]

    # One empty/full mbarrier pair per semantic MMA-input readiness edge.
    p_inputs_ready_mbar_ptr: cute.struct.MemRange[cutlass.Int64, 2]
    dvprime_state_inputs_ready_mbar_ptr: cute.struct.MemRange[cutlass.Int64, 2]
    dvprime_pgamma_inputs_ready_mbar_ptr: cute.struct.MemRange[cutlass.Int64, 2]
    g_smem_reuse_mbar_ptr: cute.struct.MemRange[cutlass.Int64, 2]
    dv_inputs_ready_mbar_ptr: cute.struct.MemRange[cutlass.Int64, 2]
    vprime_dag_inputs_ready_mbar_ptr: cute.struct.MemRange[cutlass.Int64, 2]
    sdg_ownership_order_mbar_ptr: cute.struct.MemRange[cutlass.Int64, 2]
    dpg_inputs_ready_mbar_ptr: cute.struct.MemRange[cutlass.Int64, 2]
    dk_state_inputs_ready_mbar_ptr: cute.struct.MemRange[cutlass.Int64, 2]
    u_readers_done_before_dq_state_mma_mbar_ptr: cute.struct.MemRange[
        cutlass.Int64, 2
    ]
    dk_state_g_inputs_ready_mbar_ptr: cute.struct.MemRange[cutlass.Int64, 2]
    dq_dp_inputs_ready_mbar_ptr: cute.struct.MemRange[cutlass.Int64, 2]
    dh_k_inputs_ready_mbar_ptr: cute.struct.MemRange[cutlass.Int64, 2]
    da_left_at_inputs_ready_mbar_ptr: cute.struct.MemRange[cutlass.Int64, 2]
    da_right_inputs_ready_mbar_ptr: cute.struct.MemRange[cutlass.Int64, 2]
    dh_q_inputs_ready_mbar_ptr: cute.struct.MemRange[cutlass.Int64, 2]
    dk_da_inputs_ready_mbar_ptr: cute.struct.MemRange[cutlass.Int64, 2]

    # Four allocation-result tokens for the legal 480-column split.
    tmem_holding_buf: cute.struct.MemRange[
        cutlass.Int32, len(TMEM_ALLOCATION_COLUMNS)
    ]
    sDvOffset: cute.struct.Align[
        cute.struct.MemRange[cutlass.Int32, 128],
        16,
    ]

    # Four 64x128 BF16 input tiles.
    sDo: cute.struct.Align[
        cute.struct.MemRange[cutlass.BFloat16, _BT * _DV],
        _BUFFER_ALIGN_BYTES,
    ]
    sQ: cute.struct.Align[
        cute.struct.MemRange[cutlass.BFloat16, _BT * _DK],
        _BUFFER_ALIGN_BYTES,
    ]
    sK: cute.struct.Align[
        cute.struct.MemRange[cutlass.BFloat16, _BT * _DK],
        _BUFFER_ALIGN_BYTES,
    ]
    sV: cute.struct.Align[
        cute.struct.MemRange[cutlass.BFloat16, _BT * _DV],
        _BUFFER_ALIGN_BYTES,
    ]

    # Three 64x64 BF16 tiles.
    sA: cute.struct.Align[
        cute.struct.MemRange[cutlass.BFloat16, _BT * _BT],
        _BUFFER_ALIGN_BYTES,
    ]
    # P-gamma, then dP scratch across non-overlapping phases.
    sTmp11: cute.struct.Align[
        cute.struct.MemRange[cutlass.BFloat16, _BT * _BT],
        _BUFFER_ALIGN_BYTES,
    ]
    # A-gamma-beta, dAb, dA, then final dA_s scratch.
    sTmp12: cute.struct.Align[
        cute.struct.MemRange[cutlass.BFloat16, _BT * _BT],
        _BUFFER_ALIGN_BYTES,
    ]

    # Two 128x128 BF16 state tiles.
    sH: cute.struct.Align[
        cute.struct.MemRange[cutlass.BFloat16, _DK * _DV],
        _BUFFER_ALIGN_BYTES,
    ]
    # Raw incoming dH snapshot used by state-gradient MMAs.
    sTmp41: cute.struct.Align[
        cute.struct.MemRange[cutlass.BFloat16, _DK * _DV],
        _BUFFER_ALIGN_BYTES,
    ]

    # Four 64x128 BF16 scratch/output tiles.
    # Dedicated dQ staging tile consumed by the TMA store warp.
    sDqkv: cute.struct.Align[
        cute.struct.MemRange[cutlass.BFloat16, _BT * _DK],
        _BUFFER_ALIGN_BYTES,
    ]
    # dVprime, Vprime, then dO-gamma scratch across ordered phases.
    sTmp21: cute.struct.Align[
        cute.struct.MemRange[cutlass.BFloat16, _BT * _DK],
        _BUFFER_ALIGN_BYTES,
    ]
    # W tile, then AQ reduction scratch after W readers finish.
    sTmp22: cute.struct.Align[
        cute.struct.MemRange[cutlass.BFloat16, _BT * _DK],
        _BUFFER_ALIGN_BYTES,
    ]
    # dV-gamma tile consumed by the dH and dK paths.
    sTmp23: cute.struct.Align[
        cute.struct.MemRange[cutlass.BFloat16, _BT * _DK],
        _BUFFER_ALIGN_BYTES,
    ]

    # TMA needs at least 128 contiguous bits. Load four adjacent heads for
    # each token and let the consumer select its head lane.
    sGTma: cute.struct.Align[
        cute.struct.MemRange[cutlass.Float32, _TMA_VECTOR_HEADS * _BT],
        16,
    ]
    sExpG: cute.struct.MemRange[cutlass.Float32, _BT]
    sScaledExpG: cute.struct.MemRange[cutlass.Float32, _BT]
    sReverseExpG: cute.struct.MemRange[cutlass.Float32, _BT]
    sBetaTma: cute.struct.Align[
        cute.struct.MemRange[cutlass.Float32, _TMA_VECTOR_HEADS * _BT],
        16,
    ]
    sDg: cute.struct.MemRange[cutlass.Float32, _BT]
    sDb: cute.struct.MemRange[cutlass.Float32, _BT]
    sReduce: cute.struct.MemRange[cutlass.Float32, 2 * _BT]
    sReduce2: cute.struct.MemRange[cutlass.Float32, 2 * _BT]


assert SharedStorage.size_in_bytes() <= _SM100_OPTIN_SMEM_BYTES


def get_layout_budget() -> LayoutBudget:
    """Return the real decorated-struct size and fixed SM100 CTA budgets."""

    return LayoutBudget(
        smem_bytes=SharedStorage.size_in_bytes(),
        tmem_columns=TMEM_COLUMNS,
        threads=_THREADS_PER_CTA,
    )


@dataclass(frozen=True)
class CanonicalSharedViews:
    """Canonical SMEM aliases used by TMA and consumer warp groups."""

    q: Any
    q_transposed: Any
    k: Any
    k_transposed: Any
    v: Any
    do: Any
    do_padded: Any
    a: Any
    h: Any
    tmp11_base_ptr: Any
    tmp12: Any
    dqkv: Any
    tmp21: Any
    tmp21_left_square_transposed: Any
    tmp21_right_square_transposed: Any
    tmp22: Any
    tmp22_padded: Any
    tmp23: Any
    tmp23_left_square_transposed: Any
    tmp23_right_square_transposed: Any
    tmp41: Any
    tmp41_padded: Any


@dataclass(frozen=True)
class PackedSharedViews:
    """Named packed-layout SMEM aliases for the individual MMA operations."""

    tmp11_p1: Any
    tmp12_p1: Any
    do_p1: Any
    tmp21_p1: Any
    q_p1: Any
    tmp11_p2: Any
    tmp12_p2: Any
    tmp22_p2: Any
    k_p2: Any
    k_p8: Any
    tmp41_p8: Any
    h_p8: Any
    tmp21_p9: Any
    do_p9: Any
    tmp23_p9: Any
    tmp41_p9: Any
    h_p9: Any
    q_p0: Any
    k_p0: Any
    tmp21_p0a: Any
    tmp22_p0b: Any
    do_p0: Any
    tmp21_p0b: Any
    a_p6: Any
    tmp12_p6: Any
    tmp12_p7: Any
    a_p7: Any
    k_left_p7_a: Any
    k_left_p7_b: Any
    k_right_p7_a: Any
    k_right_p7_b: Any


@dataclass(frozen=True)
class VectorSharedViews:
    """Vector and reduction SMEM views."""

    g: Any
    exp_g: Any
    scaled_exp_g: Any
    rev_exp_g: Any
    beta: Any
    dg: Any
    db: Any
    dv_offset: Any
    reduce: Any
    reduce2: Any


@dataclass(frozen=True)
class SharedViews:
    """Trace-time bundle of all named SMEM aliases."""

    canonical: CanonicalSharedViews
    packed: PackedSharedViews
    vectors: VectorSharedViews


def make_shared_views(
    storage,
    *,
    token_layout,
    token_transposed_layout,
    square_layout,
    square_transposed_layout,
    state_layout,
    packed_dvprime_pg_a,
    packed_dvprime_pg_b,
    packed_vprime_a,
    packed_vprime_b,
    packed_dvprime_state_a,
    packed_dvprime_state_b,
    packed_dq_state_a,
    packed_dq_state_b,
    packed_p_a,
    packed_p_b,
    packed_da_left_a,
    packed_da_left_b,
    packed_at_a,
    packed_at_b,
    io_dtype,
) -> SharedViews:
    """Build every SMEM alias without changing its layout or byte offset."""

    def get(member, layout):
        return member.get_tensor(
            layout.outer, swizzle=layout.inner
        )

    canonical = CanonicalSharedViews(
        q=get(storage.sQ, token_layout),
        q_transposed=get(storage.sQ, token_transposed_layout),
        k=get(storage.sK, token_layout),
        k_transposed=get(storage.sK, token_transposed_layout),
        v=get(storage.sV, token_layout),
        do=get(storage.sDo, token_layout),
        do_padded=cute.make_tensor(
            storage.sDo.data_ptr(),
            cute.make_layout((_BT, _BT), stride=(_BT + 2, 1)),
        ),
        a=get(storage.sA, square_layout),
        h=get(storage.sH, state_layout),
        tmp11_base_ptr=storage.sTmp11.data_ptr(),
        tmp12=get(storage.sTmp12, square_layout),
        dqkv=get(storage.sDqkv, token_layout),
        tmp21=get(storage.sTmp21, token_layout),
        tmp21_left_square_transposed=get(
            storage.sTmp21, square_transposed_layout
        ),
        tmp21_right_square_transposed=cute.make_tensor(
            cute.recast_ptr(
                storage.sTmp21.data_ptr() + _BT * (_DV // 2),
                square_transposed_layout.inner,
                dtype=io_dtype,
            ),
            square_transposed_layout.outer,
        ),
        tmp22=get(storage.sTmp22, token_layout),
        tmp22_padded=cute.make_tensor(
            storage.sTmp22.data_ptr(),
            cute.make_layout((_BT, _BT), stride=(_BT + 2, 1)),
        ),
        tmp23=get(storage.sTmp23, token_layout),
        tmp23_left_square_transposed=get(
            storage.sTmp23, square_transposed_layout
        ),
        tmp23_right_square_transposed=cute.make_tensor(
            cute.recast_ptr(
                storage.sTmp23.data_ptr() + _BT * (_DV // 2),
                square_transposed_layout.inner,
                dtype=io_dtype,
            ),
            square_transposed_layout.outer,
        ),
        tmp41=get(storage.sTmp41, state_layout),
        tmp41_padded=cute.make_tensor(
            storage.sTmp41.data_ptr(),
            cute.make_layout((_BT, _BT), stride=(_BT + 2, 1)),
        ),
    )
    packed = PackedSharedViews(
        tmp11_p1=get(storage.sTmp11, packed_dvprime_pg_a),
        tmp12_p1=get(storage.sTmp12, packed_dvprime_pg_a),
        do_p1=get(storage.sDo, packed_dvprime_pg_b),
        tmp21_p1=get(storage.sTmp21, packed_dvprime_pg_b),
        q_p1=get(storage.sQ, packed_dvprime_pg_b),
        tmp11_p2=get(storage.sTmp11, packed_vprime_a),
        tmp12_p2=get(storage.sTmp12, packed_vprime_a),
        tmp22_p2=get(storage.sTmp22, packed_vprime_b),
        k_p2=get(storage.sK, packed_vprime_b),
        k_p8=get(storage.sK, packed_dvprime_state_a),
        tmp41_p8=get(storage.sTmp41, packed_dvprime_state_b),
        h_p8=get(storage.sH, packed_dvprime_state_b),
        tmp21_p9=get(storage.sTmp21, packed_dq_state_a),
        do_p9=get(storage.sDo, packed_dq_state_a),
        tmp23_p9=get(storage.sTmp23, packed_dq_state_a),
        tmp41_p9=get(storage.sTmp41, packed_dq_state_b),
        h_p9=get(storage.sH, packed_dq_state_b),
        q_p0=get(storage.sQ, packed_p_a),
        k_p0=get(storage.sK, packed_p_b),
        tmp21_p0a=get(storage.sTmp21, packed_p_a),
        tmp22_p0b=get(storage.sTmp22, packed_p_b),
        do_p0=get(storage.sDo, packed_p_a),
        tmp21_p0b=get(storage.sTmp21, packed_p_b),
        a_p6=get(storage.sA, packed_da_left_a),
        tmp12_p6=get(storage.sTmp12, packed_da_left_b),
        tmp12_p7=get(storage.sTmp12, packed_at_a),
        a_p7=get(storage.sA, packed_at_b),
        k_left_p7_a=get(storage.sK, packed_at_a),
        k_left_p7_b=get(storage.sK, packed_at_b),
        k_right_p7_a=cute.make_tensor(
            cute.recast_ptr(
                storage.sK.data_ptr() + _BT * (_DK // 2),
                packed_at_a.inner,
                dtype=io_dtype,
            ),
            packed_at_a.outer,
        ),
        k_right_p7_b=cute.make_tensor(
            cute.recast_ptr(
                storage.sK.data_ptr() + _BT * (_DK // 2),
                packed_at_b.inner,
                dtype=io_dtype,
            ),
            packed_at_b.outer,
        ),
    )
    vector_layout = cute.make_layout((_BT,), stride=(1,))
    vector_tma_staged_layout = cute.make_layout(
        (_TMA_VECTOR_HEADS, _BT, 1),
        stride=(1, _TMA_VECTOR_HEADS, _TMA_VECTOR_HEADS * _BT),
    )
    reduce_layout = cute.make_layout((_BT, 2), stride=(1, _BT))
    vectors = VectorSharedViews(
        g=storage.sGTma.get_tensor(vector_tma_staged_layout),
        exp_g=storage.sExpG.get_tensor(vector_layout),
        scaled_exp_g=storage.sScaledExpG.get_tensor(vector_layout),
        rev_exp_g=storage.sReverseExpG.get_tensor(vector_layout),
        beta=storage.sBetaTma.get_tensor(vector_tma_staged_layout),
        dg=storage.sDg.get_tensor(vector_layout),
        db=storage.sDb.get_tensor(vector_layout),
        dv_offset=storage.sDvOffset.get_tensor(
            cute.make_layout((128,), stride=(1,))
        ),
        reduce=storage.sReduce.get_tensor(reduce_layout),
        reduce2=storage.sReduce2.get_tensor(reduce_layout),
    )
    return SharedViews(canonical=canonical, packed=packed, vectors=vectors)


@dataclass(frozen=True)
class TmemViews:
    """Named TMEM pointers, MMA accumulators, and direct-copy aliases."""

    primary_ptr: Any
    secondary_ptr: Any
    mask_ptr: Any
    vprime_ptr: Any
    dog_ptr: Any
    dq_ptr: Any
    da_ptr: Any
    square_ptr: Any
    u_acc: Any
    vprime_acc: Any
    dog_acc: Any
    dq_acc: Any
    dk_acc: Any
    dv_acc: Any
    dh_left_acc: Any
    dh_right_acc: Any
    p_copy_acc: Any
    dp_copy_acc: Any
    a_copy_acc: Any
    da_copy_acc: Any
    mask_acc: Any


def _make_tmem_accumulator(tmem_ptr, tiled_mma, shape, offset):
    accumulator_shape = tiled_mma.partition_shape_C(shape)
    fake = tiled_mma.make_fragment_C(
        cute.append(accumulator_shape, 1)
    )
    return cute.make_tensor(tmem_ptr + offset, fake.layout)


def make_tmem_views(tmem_ptrs, mma_64x128, mma_128x64) -> TmemViews:
    """Build the named TMEM aliases after kernel-owned allocation."""

    primary_ptr, secondary_ptr, mask_ptr, vprime_ptr = tmem_ptrs
    dog_ptr = vprime_ptr
    dq_ptr = primary_ptr
    da_ptr = primary_ptr + 128
    square_ptr = primary_ptr + 160
    square_copy_layout = cute.make_layout(
        ((16, 4), (32, 2)),
        stride=((65536, 32 * 65536), (1, 16 * 65536)),
    )
    return TmemViews(
        primary_ptr=primary_ptr,
        secondary_ptr=secondary_ptr,
        mask_ptr=mask_ptr,
        vprime_ptr=vprime_ptr,
        dog_ptr=dog_ptr,
        dq_ptr=dq_ptr,
        da_ptr=da_ptr,
        square_ptr=square_ptr,
        u_acc=_make_tmem_accumulator(
            primary_ptr, mma_64x128, (64, 128), 192
        ),
        vprime_acc=_make_tmem_accumulator(
            vprime_ptr, mma_64x128, (64, 128), 0
        ),
        dog_acc=_make_tmem_accumulator(
            dog_ptr, mma_64x128, (64, 128), 0
        ),
        dq_acc=_make_tmem_accumulator(
            dq_ptr, mma_64x128, (64, 128), 192
        ),
        dk_acc=_make_tmem_accumulator(
            primary_ptr, mma_64x128, (64, 128), 0
        ),
        dv_acc=_make_tmem_accumulator(
            primary_ptr, mma_64x128, (64, 128), 64
        ),
        dh_left_acc=_make_tmem_accumulator(
            secondary_ptr, mma_128x64, (128, 64), 0
        ),
        dh_right_acc=_make_tmem_accumulator(
            secondary_ptr, mma_128x64, (128, 64), 64
        ),
        p_copy_acc=cute.make_tensor(primary_ptr, square_copy_layout),
        dp_copy_acc=cute.make_tensor(square_ptr, square_copy_layout),
        a_copy_acc=cute.make_tensor(square_ptr, square_copy_layout),
        da_copy_acc=cute.make_tensor(da_ptr, square_copy_layout),
        mask_acc=cute.make_tensor(mask_ptr, square_copy_layout),
    )


__all__ = [
    "CanonicalSharedViews",
    "LayoutBudget",
    "PackedSharedViews",
    "SharedStorage",
    "SharedViews",
    "TMEM_ALLOCATION_COLUMNS",
    "TMEM_COLUMNS",
    "TMEM_RANGES",
    "TmemViews",
    "VectorSharedViews",
    "get_layout_budget",
    "make_shared_views",
    "make_tmem_views",
]
