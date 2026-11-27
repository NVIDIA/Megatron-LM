# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Unit tests for the resharded distributed Newton-Schulz of wide GTP matrices
(``megatron.core.optimizer.muon_ns_reshard``, ``--muon-distributed-reshard-wide``).

- Decision helpers and auto's cost model (``_select_tp_mode``): at the N4 Ultra dense shapes on
  a 64-rank GTP group with the GB200 profile, the flag flips exactly the wide matrices the
  resharded chain beats (attention proj, dense-MLP fc2, MTP eh_proj) to distributed and keeps
  the latency-bound fc1_latent duplicated. Without the flag the collective latency never enters
  a decision.
- Config wiring: the reflective builders hand the flag to the optimizer the registry builds.
- Multi-rank (torchrun, the world as the GTP group): the two all_to_all exchanges are exact
  inverses and produce the column blocks of the full matrix; the resharded chain on random wide
  [M, K] (incl. GTP pad_length > 0, padding spanning shards, K not divisible by the group)
  matches the whole-matrix Newton-Schulz of the duplicated path, tightly in fp32 ("highest"),
  to bf16 tolerance under "medium", with Muon's scale from the logical shape and exact-zero pad
  rows; both paddings are exact bit for bit (the update does not depend on the GTP dim-0
  padding, and the dim-1 zero columns equal explicit zero columns of the input); tall, square,
  TP-sharded and expert weights are bitwise unchanged by the flag; auto picks and runs the
  resharded chain; an expert weight under "distributed" takes it over the expert GTP group.

Launch: torchrun --nproc-per-node=4 -m pytest tests/unit_tests/optimizer/test_muon_ns_reshard.py
"""

import inspect

import pytest
import torch
import torch.distributed as dist

pytest.importorskip("emerging_optimizers", reason="requires emerging-optimizers")

from megatron.core.optimizer import OptimizerConfig
from megatron.core.optimizer import emerging_optimizers as eo_mod
from megatron.core.optimizer import muon_ns_reshard as mrs
from megatron.core.process_groups_config import ProcessGroupCollection
from tests.unit_tests.test_emerging_optimizers import _DispatchChunk
from tests.unit_tests.test_utilities import Utils

_SEED = 1234
GTP = 64
GB200 = eo_mod._PROFILES["GB200"]
NO_LATENCY = eo_mod.HardwareProfile(
    bf16_peak_tflops=GB200.bf16_peak_tflops,
    bw_intra_gbps=GB200.bw_intra_gbps,
    bw_inter_gbps=GB200.bw_inter_gbps,
)
# N4 Ultra dense (GTP) matrices, logical [out, in].
ULTRA = {
    "attention qkv": (25088, 10240),
    "attention proj": (10240, 24576),
    "MoE fc1_latent": (3072, 10240),
    "MoE fc2_latent": (10240, 3072),
    "MoE shared fc1/fc2": (10240, 10240),
    "dense MLP fc1": (40960, 10240),
    "dense MLP fc2": (10240, 40960),
    "MTP eh_proj": (10240, 20480),
    "GDP out_proj": (10240, 5120),
}
FLIPPED = {"attention proj", "dense MLP fc2", "MTP eh_proj"}


def _padded_rows(rows, group_size, align=32):
    """Rows of the GTP-padded matrix: every rank's shard a multiple of ``align`` (mxfp8)."""
    return -(-rows // (group_size * align)) * align * group_size


def _auto(shape, reshard, profile=GB200, steps=16, group_size=GTP, use_syrk=False):
    """auto's decision for a GTP-only weight, as _resolve_tp_mode calls the cost model."""
    rows, cols = shape
    reshard_rows = rows if reshard and rows < cols else None
    return eo_mod._select_tp_mode(
        _padded_rows(rows, group_size),
        cols,
        group_size,
        steps,
        use_syrk=use_syrk,
        elem_size=2,
        communication_crosses_domain=False,
        profile=profile,
        reshard_rows=reshard_rows,
    )


@pytest.fixture(scope="module")
def dist_world():
    """The world group (initialized on first use) for the multi-rank tests."""
    Utils.initialize_model_parallel()
    torch.manual_seed(_SEED)
    yield dist.group.WORLD
    Utils.destroy_model_parallel()


@pytest.fixture
def world(dist_world):
    if dist.get_world_size() < 2:
        pytest.skip("requires >= 2 ranks (torchrun --nproc-per-node=4)")
    prev = torch.get_float32_matmul_precision()
    yield dist_world
    torch.set_float32_matmul_precision(prev)


# ---------------------------------------------------------------------------
# Decision helpers and auto's cost model (no communication)
# ---------------------------------------------------------------------------


def test_reshard_helpers():
    assert mrs.local_cols(40960, 64) == 640
    assert mrs.local_cols(24576, 64) == 384
    assert mrs.local_cols(202, 4) == 51  # zero-padded to 204
    # fc2 at gtp64: the [160, 40960] fp32 shard minus the slab a rank keeps (~25.8 MB).
    assert mrs.reshard_exchange_bytes(160, 40960, 64) == 160 * 40960 * 4 * 63 // 64
    assert mrs.reshard_exchange_bytes(24, 202, 4) == 24 * 204 * 4 * 3 // 4


def test_auto_flips_exactly_the_wide_ultra_matrices():
    """GB200 profile at gtp64: proj, fc2 and eh_proj run duplicated without the flag and the
    resharded distributed chain with it; fc1_latent (3072 rows: 16 latency-bound Gram
    all-reduces) stays duplicated; tall and square matrices are not resharded at all and run
    distributed (on their dim-0 shard, Gram over the short dim 1) with and without the flag."""
    off = {name: _auto(shape, False) for name, shape in ULTRA.items()}
    on = {name: _auto(shape, True) for name, shape in ULTRA.items()}
    assert {name for name in ULTRA if on[name] != off[name]} == FLIPPED, (off, on)
    for name in FLIPPED:
        assert (off[name], on[name]) == ("duplicated", "distributed"), name
    assert off["MoE fc1_latent"] == on["MoE fc1_latent"] == "duplicated"
    tall_or_square = {name for name, (rows, cols) in ULTRA.items() if rows >= cols}
    assert tall_or_square == set(ULTRA) - FLIPPED - {"MoE fc1_latent"}
    for name in tall_or_square:
        assert off[name] == on[name] == "distributed", name


def test_reshard_pricing_is_latency_aware():
    """fc1_latent 3072 x 10240: with the GB200 latency (0.5 ms per collective) the 19
    collectives of the resharded chain lose to one all-gather; without latency it wins."""
    lat1 = ULTRA["MoE fc1_latent"]
    assert _auto(lat1, True) == "duplicated"
    assert _auto(lat1, True, profile=NO_LATENCY) == "distributed"
    # FLOP-only auto (hardware not in the registry): the resharded chain wins on FLOPs.
    fc2 = ULTRA["dense MLP fc2"]
    assert _auto(fc2, False, profile=None) == "duplicated"
    assert _auto(fc2, True, profile=None) == "distributed"


def test_flag_can_move_a_near_square_wide_weight_to_duplicated():
    """The flag's comparison is latency-aware, the flag-off one is not, so the flag can also
    move a weight from distributed to duplicated. 6144 x 6400 at gtp64 (GB200 profile, 16
    steps): without the flag the dim-0 chain beats duplicated on FLOPs and bytes; with it the
    19 collectives of the resharded chain cost more than duplicated's one all-gather. Without
    latency the resharded chain would win, so the latency is what decides. None of the Ultra
    shapes is in this regime (test_auto_flips_exactly_the_wide_ultra_matrices)."""
    shape = (6144, 6400)
    assert _padded_rows(shape[0], GTP) == shape[0] and shape[0] < shape[1]
    assert _auto(shape, False) == "distributed"
    assert _auto(shape, True) == "duplicated"
    assert _auto(shape, True, profile=NO_LATENCY) == "distributed"


def test_latency_only_enters_the_resharded_comparison():
    """Without reshard_rows no decision depends on collective_latency_s: every decision over
    a grid of shapes, group sizes, step counts, SYRK and element sizes is the same with no
    latency, the GB200 latency and an absurd one."""
    huge = eo_mod.HardwareProfile(
        bf16_peak_tflops=2500.0, bw_intra_gbps=900.0, bw_inter_gbps=100.0, collective_latency_s=1.0
    )
    shapes = list(ULTRA.values()) + [(1024, 4096), (4096, 1024), (640, 1024), (96, 256)]
    for m, n in shapes:
        for group_size in (2, 4, 8, 64):
            for steps in (5, 16):
                for use_syrk in (False, True):
                    for elem_size in (2, 4):
                        decisions = {
                            eo_mod._select_tp_mode(
                                _padded_rows(m, group_size),
                                n,
                                group_size,
                                steps,
                                use_syrk,
                                elem_size,
                                communication_crosses_domain=False,
                                profile=profile,
                            )
                            for profile in (NO_LATENCY, GB200, huge)
                        }
                        assert len(decisions) == 1, (m, n, group_size, steps, use_syrk)


# ---------------------------------------------------------------------------
# Config wiring
# ---------------------------------------------------------------------------


def test_builders_hand_the_flag_to_the_optimizer():
    """The reflective builders hand the flag to the TensorParallelMuon the registry builds
    under auto (the production-like configuration: auto with split-QKV on), and to
    adaptive_muon, whose class accepts it."""
    cfg = OptimizerConfig(
        optimizer='muon',
        use_layer_wise_distributed_optimizer=True,
        muon_tp_mode='auto',
        muon_split_qkv=True,
        muon_distributed_reshard_wide=True,
    )
    cfg.validate()
    group = {'params': [torch.nn.Parameter(torch.randn(8, 8))]}
    opt, _ = eo_mod._create_emerging_optimizer(cfg, [group], 'muon', [_DispatchChunk()], None)
    assert type(opt) is eo_mod.TensorParallelMuon
    assert opt.tp_mode == "auto" and opt.split_qkv is True
    assert opt.distributed_reshard_wide is True
    adaptive_cfg = OptimizerConfig(
        optimizer='adaptive_muon', muon_tp_mode='auto', muon_distributed_reshard_wide=True
    )
    adaptive = eo_mod._adaptive_muon_config_to_kwargs(adaptive_cfg, [_DispatchChunk()], None)
    assert adaptive["distributed_reshard_wide"] is True
    adaptive_init = inspect.signature(eo_mod.TensorParallelAdaptiveMuon.__init__)
    assert "distributed_reshard_wide" in adaptive_init.parameters


# ---------------------------------------------------------------------------
# Multi-rank: the exchanges
# ---------------------------------------------------------------------------


def _broadcast_randn(rows, cols):
    full = torch.randn(rows, cols, dtype=torch.float32, device="cuda")
    dist.broadcast(full, src=0)
    return full


@pytest.mark.parametrize("rows_per_rank, cols", [(3, 16), (5, 18), (2, 3)])
def test_exchanges_give_column_blocks_and_invert(world, rows_per_rank, cols):
    """rows_to_cols returns this rank's column block of the full (dim-1 zero-padded) matrix,
    over all rows; cols_to_rows brings back the row shard bit for bit (18 and 3 columns are not
    divisible by 4 ranks; with 3 columns some ranks hold only padding)."""
    size, rank = dist.get_world_size(), dist.get_rank()
    full = _broadcast_randn(size * rows_per_rank, cols)
    shard = full[rank * rows_per_rank : (rank + 1) * rows_per_rank].clone()
    block = mrs.rows_to_cols(shard, world)
    kc = mrs.local_cols(cols, size)
    padded = torch.nn.functional.pad(full, (0, kc * size - cols))
    assert block.shape == (size * rows_per_rank, kc)
    assert torch.equal(block, padded[:, rank * kc : (rank + 1) * kc])
    back = mrs.cols_to_rows(block, world, cols)
    assert back.shape == shard.shape and back.is_contiguous()
    assert torch.equal(back, shard)


# ---------------------------------------------------------------------------
# Multi-rank: the resharded chain
# ---------------------------------------------------------------------------


def _muon(
    tp_mode,
    reshard,
    steps=5,
    coefficient_type="quintic",
    tp_group=None,
    expert=False,
    scale_mode="spectral",
):
    placeholder = torch.nn.Parameter(torch.zeros(1, device="cuda"))
    groups = dict(gtp_remat=dist.group.WORLD, tp=tp_group)
    if expert:
        groups.update(expt_gtp_remat=dist.group.WORLD, expt_tp=tp_group)
    return eo_mod.TensorParallelMuon(
        params=[placeholder],
        lr=0.01,
        momentum=0.95,
        weight_decay=0.0,
        num_ns_steps=steps,
        coefficient_type=coefficient_type,
        scale_mode=scale_mode,
        pg_collection=ProcessGroupCollection(**groups),
        tp_mode=tp_mode,
        distributed_reshard_wide=reshard,
    )


def _gtp_shard(full, align):
    """This rank's dim-0 shard of ``full`` GTP-padded with zero rows to a multiple of
    ``align * world`` rows, tagged like a GTP param; returns (shard, pad_length)."""
    size, rank = dist.get_world_size(), dist.get_rank()
    pad_length = (-full.size(0)) % (align * size)
    padded = torch.nn.functional.pad(full, (0, 0, 0, pad_length))
    shard_rows = padded.size(0) // size
    local = padded[rank * shard_rows : (rank + 1) * shard_rows].clone()
    local.is_gtp_weight_remat = True
    local.pad_length = pad_length
    return local, pad_length


def _padded_momentum(logical_rows, cols, align):
    """Random [logical_rows, cols] momentum (same on every rank) and its GTP shard: returns
    (logical, pad_length, this rank's row shard)."""
    logical = _broadcast_randn(logical_rows, cols)
    local, pad_length = _gtp_shard(logical, align)
    return logical, pad_length, local


def _orth(opt, local, tp_group=None, partition_dim=None, qkv_split_shapes=None):
    return opt.scaled_orthogonalize_fn_with_gtp_remat(
        local, local.clone(), tp_group, partition_dim, qkv_split_shapes=qkv_split_shapes
    )


def _max_rel(a, b):
    """max |a - b| over all ranks, relative to max |b| over all ranks."""
    t = torch.tensor([(a - b).abs().max().item(), b.abs().max().item()], device="cuda")
    dist.all_reduce(t, op=dist.ReduceOp.MAX)
    return (t[0] / t[1]).item()


# (logical rows, cols, dim-0 alignment, matmul precision, tolerance): the padded rows are sharded
# over the world. fp32 on every layout; bf16 (production) on the layout with both paddings, since
# the bf16 update does not depend on either padding (test_resharded_padding_is_exact_bitwise).
_WIDE_CASES = [
    pytest.param(96, 256, 1, "highest", 5e-4, id="even"),
    pytest.param(96, 202, 1, "highest", 5e-4, id="cols-not-divisible"),
    pytest.param(90, 256, 8, "highest", 5e-4, id="pad"),
    pytest.param(40, 130, 16, "highest", 5e-4, id="pad-spans-shards"),
    pytest.param(40, 130, 16, "medium", 6e-2, id="pad-spans-shards-bf16"),
]


@pytest.mark.parametrize("logical_rows, cols, align, prec, tol", _WIDE_CASES)
def test_resharded_matches_whole_matrix_ns(world, logical_rows, cols, align, prec, tol):
    """The resharded chain on this rank's row shard equals the row shard of the whole-matrix
    Newton-Schulz (duplicated path: Gram over the short dim 0, scale from the logical shape),
    and this rank's pad rows are exactly zero. 16 polar_express steps, the production
    iteration."""
    size, rank = dist.get_world_size(), dist.get_rank()
    logical, pad_length, local = _padded_momentum(logical_rows, cols, align)
    if align > 1:
        assert pad_length > 0
    resh = _muon("distributed", True, 16, "polar_express")
    dup = _muon("duplicated", False, 16, "polar_express")
    torch.set_float32_matmul_precision(prec)
    out = _orth(resh, local)
    ref_full = dup.scaled_orthogonalize_fn(logical.clone(), None, partition_dim=None)
    ref_dup_path = _orth(dup, local)
    shard_rows = local.size(0)
    expected = torch.nn.functional.pad(ref_full, (0, 0, 0, pad_length))[
        rank * shard_rows : (rank + 1) * shard_rows
    ]
    assert out.shape == local.shape
    assert resh._reshard_logged == {(logical_rows, cols, size)}
    # The duplicated GTP path is the whole-matrix NS of the logical matrix (sanity check of
    # the reference: same GEMMs on the same values).
    assert _max_rel(ref_dup_path, expected) < tol / 10
    rel = _max_rel(out, expected)
    assert rel < tol, f"resharded vs whole-matrix NS: {rel:.3e} (tol {tol})"
    first_pad_row = logical_rows - rank * shard_rows
    if first_pad_row < shard_rows:
        assert torch.count_nonzero(out[max(0, first_pad_row) :]) == 0


def _all_rows(local):
    """The whole matrix whose dim-0 shards are every rank's ``local``."""
    full = torch.empty(dist.get_world_size() * local.size(0), local.size(1), device=local.device)
    dist.all_gather_into_tensor(full, local.contiguous())
    return full


def _nonzero_bits(t):
    """Number of elements whose bit pattern is not +0.0 (so -0.0 counts)."""
    return int(t.contiguous().view(torch.int32).ne(0).sum().item())


def test_resharded_padding_is_exact_bitwise(world):
    """Both paddings of the resharded chain are exact, bit for bit (bf16 Newton-Schulz, the
    production precision).

    GTP dim-0 padding: the pad rows are stripped after the first exchange, so the chain sees the
    same ``[rows, ceil(cols / G)]`` column blocks whatever the dim-0 layout is. A padded layout
    (here the padding covers at least one whole shard) gives bitwise the update of the unpadded
    layout of the same logical matrix, and pad rows that are exactly +0.0.

    Dim-1 padding (``cols`` not divisible by the group): the zero columns the chain appends give
    bitwise what explicit zero columns of the input give, those columns stay exactly +0.0 through
    every step, and the update has the logical width. ``scale_mode="shape_scaling"`` (factor 1
    for a wide matrix) keeps the two widths on the same Muon scale.
    """
    size = dist.get_world_size()
    rows, cols = 8 * size, 64 * size + 3  # 16-row alignment: 8 * size pad rows, >= one shard
    ext_cols = mrs.local_cols(cols, size) * size
    assert rows % size == 0 and cols % size != 0 and ext_cols > cols
    logical = _broadcast_randn(rows, cols)
    plain, no_pad = _gtp_shard(logical, 1)
    padded, pad_length = _gtp_shard(logical, 16)
    extended, _ = _gtp_shard(torch.nn.functional.pad(logical, (0, ext_cols - cols)), 1)
    assert no_pad == 0 and pad_length >= padded.size(0)
    opt = _muon("distributed", True, 16, "polar_express", scale_mode="shape_scaling")
    torch.set_float32_matmul_precision("medium")
    out_plain, out_padded = _orth(opt, plain), _orth(opt, padded)
    out_extended = _orth(opt, extended)
    assert opt._reshard_logged == {(rows, cols, size), (rows, ext_cols, size)}
    assert out_plain.shape == plain.shape and out_padded.shape == padded.shape
    assert out_extended.shape == extended.shape
    full_plain, full_padded, full_extended = map(_all_rows, (out_plain, out_padded, out_extended))
    assert torch.isfinite(full_plain).all() and full_plain.abs().max() > 0
    # Dim-0 padding: same update on the logical rows, +0.0 on the pad rows.
    assert torch.equal(full_padded[:rows], full_plain)
    assert _nonzero_bits(full_padded[rows:]) == 0
    # Dim-1 padding: same update on the logical columns, +0.0 on the appended ones.
    assert torch.equal(full_extended[:, :cols], full_plain)
    assert _nonzero_bits(full_extended[:, cols:]) == 0


def test_unflipped_weights_are_bitwise_unchanged(world):
    """The flag changes nothing but the wide GTP-only distributed chain: tall and square
    matrices under 'distributed', a wide matrix whose TP is a second sharding axis (two-step
    path), a wide expert matrix under 'auto' (duplicated) and a split qkv under 'auto'
    (duplicated) give bitwise the flag-off update."""
    torch.set_float32_matmul_precision("medium")
    for logical_rows, cols in ((256, 96), (96, 96)):
        _, _, local = _padded_momentum(logical_rows, cols, 1)
        resh = _muon("distributed", True)
        assert torch.equal(_orth(resh, local), _orth(_muon("distributed", False), local))
        assert resh._reshard_logged == set()

    # TP as a second sharding axis: the two-step path keeps its layout.
    _, _, wide = _padded_momentum(96, 256, 1)
    resh_tp = _muon("distributed", True, tp_group=world)
    plain_tp = _muon("distributed", False, tp_group=world)
    assert resh_tp._reshard_rows(wide, 0, dist.get_world_size(), world, 1) is None
    assert torch.equal(_orth(resh_tp, wide, world, 1), _orth(plain_tp, wide, world, 1))
    assert resh_tp._reshard_logged == set()
    # partition_dim=None (TP-replicated) is not a TP sharding axis.
    assert resh_tp._reshard_rows(wide, 0, dist.get_world_size(), world, None) == 96

    # Expert weights: auto's cost model covers dense weights only, they run duplicated.
    wide.expert_tp = True
    resh_expert = _muon("auto", True, expert=True)
    assert torch.equal(_orth(resh_expert, wide), _orth(_muon("auto", False, expert=True), wide))
    assert resh_expert._tp_mode_cache == {} and resh_expert._reshard_logged == set()

    # A split qkv under auto stays the duplicated split (q | k | v rows of a wide matrix).
    _, _, qkv = _padded_momentum(48, 256, 1)
    resh_auto = _muon("auto", True)
    split = [8, 4, 4]
    assert torch.equal(
        _orth(resh_auto, qkv, qkv_split_shapes=split),
        _orth(_muon("auto", False), qkv, qkv_split_shapes=split),
    )
    assert resh_auto._tp_mode_cache == {} and resh_auto._reshard_logged == set()


def test_auto_picks_and_runs_the_resharded_chain(world):
    """FLOP-only auto (no hardware profile): a wide 90 x 256 is duplicated without the flag
    and resolves to the resharded distributed chain with it."""
    size = dist.get_world_size()
    _, pad_length, local = _padded_momentum(90, 256, 8)
    assert pad_length > 0
    plain = _muon("auto", False, 16, "polar_express")
    auto = _muon("auto", True, 16, "polar_express")
    plain._hw_profile = None
    auto._hw_profile = None
    dup = _muon("duplicated", False, 16, "polar_express")
    torch.set_float32_matmul_precision("highest")
    assert torch.equal(_orth(plain, local), _orth(dup, local))
    out = _orth(auto, local)
    ref = _orth(dup, local)
    assert plain._tp_mode_cache == {(90 + pad_length, 256, size): "duplicated"}
    assert auto._tp_mode_cache == {(90 + pad_length, 256, size, 90): "distributed"}
    assert auto._reshard_logged == {(90, 256, size)}
    assert _max_rel(out, ref) < 5e-4


def test_expert_weight_under_distributed_takes_the_resharded_chain(world):
    """An expert weight (``expert_tp``, sharded over the expert GTP group) under
    tp_mode="distributed" -- what ``muon_tp_mode='distributed'`` gives every weight -- runs
    the resharded chain when wide (incl. GTP padding) and matches the whole-matrix
    Newton-Schulz (fp32); under "auto" it stays duplicated with or without the flag
    (test_unflipped_weights_are_bitwise_unchanged)."""
    size = dist.get_world_size()
    _, pad_length, local = _padded_momentum(90, 256, 8)
    assert pad_length > 0
    local.expert_tp = True
    resh = _muon("distributed", True, 16, "polar_express", expert=True)
    dup = _muon("duplicated", False, 16, "polar_express", expert=True)
    torch.set_float32_matmul_precision("highest")
    out, ref = _orth(resh, local), _orth(dup, local)
    assert resh._reshard_logged == {(90, 256, size)} and dup._reshard_logged == set()
    assert out.shape == local.shape
    assert _max_rel(out, ref) < 5e-4
