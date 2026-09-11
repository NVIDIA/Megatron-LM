# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Tests for LayerShardedMuon.

Invariant under test: ``LayerShardedMuon.step()`` equals the duplicated-mode step
(same momentum update, same full-matrix Newton-Schulz, same weight update); only the
communication pattern differs. 1-D tests treat all ranks as one GTP_remat group and run
at any world size; 2-D tests need exactly 4 ranks (TP 2 x GTP_remat 2).

Launch: torchrun --nproc-per-node=4 -m pytest tests/unit_tests/optimizer/test_layer_sharded_muon.py
"""

import os

import pytest
import torch
import torch.distributed as dist

pytest.importorskip("emerging_optimizers", reason="LayerShardedMuon requires emerging-optimizers")

from emerging_optimizers.orthogonalized_optimizers.muon_utils import newton_schulz

from megatron.core.optimizer.layer_sharded_muon import LayerShardedMuon
from megatron.core.utils import is_emerging_optimizers_min_version
from tests.unit_tests.test_utilities import Utils

# Batched (3-D) Newton-Schulz needs emerging-optimizers >= 0.3.0; the per-matrix path
# runs on any version with the newton_schulz API.
_HAVE_BATCHED_NS = is_emerging_optimizers_min_version("0.3.0")
_SEED = 42
_MUON_KW = dict(
    lr=1e-2,
    momentum=0.95,
    nesterov=True,
    weight_decay=0.0,
    coefficient_type="quintic",
    num_ns_steps=5,
    scale_mode="spectral",
    extra_scale_factor=1.0,
    fp32_matmul_prec="highest",
)


@pytest.fixture(scope="module", autouse=True)
def _torchrun_dist_init():
    Utils.initialize_model_parallel()
    cuda = os.environ.get("TEST_DEVICE", "cuda" if torch.cuda.is_available() else "cpu") == "cuda"
    if cuda:
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        torch.cuda.set_device(local_rank)
        torch.set_default_device(f"cuda:{local_rank}")
    torch.manual_seed(_SEED)
    # Reference newton_schulz calls run under the ambient precision; pin it to the
    # optimizer's so the comparisons are exact.
    prev_prec = torch.get_float32_matmul_precision()
    torch.set_float32_matmul_precision("highest")
    yield
    torch.set_float32_matmul_precision(prev_prec)
    if cuda:
        torch.set_default_device("cpu")
    Utils.destroy_model_parallel()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _world():
    world = dist.group.WORLD
    assert world is not None, "process group must be initialized before tests run"
    return world


def _require_multi_rank():
    if dist.get_world_size() < 2:
        pytest.skip("Requires >= 2 ranks (launch with torchrun --nproc-per-node=4)")


def _require_four_ranks(reason="Requires exactly 4 ranks"):
    if dist.get_world_size() != 4:
        pytest.skip(f"{reason} (launch with torchrun --nproc-per-node=4)")


def _require_batched_ns(ns_batch_size):
    if ns_batch_size > 1 and not _HAVE_BATCHED_NS:
        pytest.skip("requires emerging-optimizers >= 0.3.0 (batched Newton-Schulz)")


def _gtp_param(t):
    """Megatron tags GTP-row-sharded weights with is_gtp_weight_remat."""
    p = torch.nn.Parameter(t)
    p.is_gtp_weight_remat = True
    return p


def _muon(params, **kwargs):
    return LayerShardedMuon(params, **{**_MUON_KW, **kwargs})


def _reference_step(w, g, lr=1e-2, momentum=0.95, nesterov=True):
    """One duplicated-mode Muon step on a full matrix from zero momentum."""
    m = torch.zeros_like(g).lerp_(g, 1 - momentum)
    eff = g.lerp(m, momentum) if nesterov else m
    orth = newton_schulz(eff.float(), steps=5, coefficient_type="quintic")
    return w - lr * (max(w.shape) ** 0.5) * orth


def _padded_shard(clean, S, r, pad):
    """Pad dim 0 with ``pad`` zero rows, then take GTP_remat shard ``r`` of ``S``."""
    padded = torch.cat([clean, torch.zeros(pad, clean.shape[1], dtype=clean.dtype)])
    rows = padded.shape[0] // S
    return padded[r * rows : (r + 1) * rows].clone()


def _shard_2d(full, pd, t, g, gtp=True, T=2, G=2):
    """This rank's shard on the TP(T) x GTP_remat(G) grid.

    pd 0: TP splits dim 0, GTP_remat splits the TP-local rows. pd 1: TP splits dim 1,
    GTP_remat splits dim 0. pd None: GTP_remat splits dim 0, or whole when not GTP-sharded.
    """
    P, Q = full.shape
    if pd == 0:
        rows = P // (T * G)
        start = t * (P // T) + g * rows
        return full[start : start + rows, :].clone()
    if pd == 1:
        return full[g * (P // G) : (g + 1) * (P // G), t * (Q // T) : (t + 1) * (Q // T)].clone()
    if gtp:
        return full[g * (P // G) : (g + 1) * (P // G), :].clone()
    return full.clone()


class _PGStub:
    """Duck-typed ProcessGroupCollection: TensorParallelMuon only reads these fields."""

    def __init__(self, gtp_remat=None, expt_gtp_remat=None, tp=None, expt_tp=None):
        self.gtp_remat = gtp_remat
        self.expt_gtp_remat = expt_gtp_remat
        self.tp = tp
        self.expt_tp = expt_tp


_TP_GROUP = None
_GTP_REMAT_GROUP = None
_EGTP_GROUP = None


def _get_2d_groups():
    """TP(2) x GTP_remat(2) subgroups, created once. Rank convention: r = g * T + t."""
    global _TP_GROUP, _GTP_REMAT_GROUP
    if _TP_GROUP is None:
        _TP_GROUP, _ = dist.new_subgroups_by_enumeration([[0, 1], [2, 3]])
        _GTP_REMAT_GROUP, _ = dist.new_subgroups_by_enumeration([[0, 2], [1, 3]])
    return _TP_GROUP, _GTP_REMAT_GROUP


def _get_expert_group():
    """Expert GTP_remat(2) subgroups {0,1} and {2,3}: a different partition than the dense
    GTP_remat group, so routing expert params over the wrong group is detectable."""
    global _EGTP_GROUP
    if _EGTP_GROUP is None:
        _EGTP_GROUP, _ = dist.new_subgroups_by_enumeration([[0, 1], [2, 3]])
    return _EGTP_GROUP


def _dense_and_expert_params(t, g, ep_rank, egtp_rank, n_same_experts):
    """Two column-parallel dense params on TP2 x GTP2 plus expert params on EGTP(2):
    ``n_same_experts`` of shape (16, 8) and one of shape (8, 16), homes alternating."""
    torch.manual_seed(_SEED + 60)
    dense_specs = [((32, 16), (0, 1)), ((32, 16), (1, 0))]
    dense_w = [torch.randn(*s) for s, _ in dense_specs]
    dense_g = [torch.randn(*s) for s, _ in dense_specs]
    dense = []
    for w, gr in zip(dense_w, dense_g):
        p = _gtp_param(_shard_2d(w, 0, t, g))
        p.grad = _shard_2d(gr, 0, t, g)
        p.partition_dim = 0
        dense.append(p)

    torch.manual_seed(_SEED + 70 + ep_rank)  # each EP group holds different experts
    expert_shapes = [(16, 8)] * n_same_experts + [(8, 16)]
    expert_w = [torch.randn(*s) for s in expert_shapes]
    expert_g = [torch.randn(*s) for s in expert_shapes]
    expert = []
    for w, gr in zip(expert_w, expert_g):
        rows = w.shape[0] // 2
        p = _gtp_param(w[egtp_rank * rows : (egtp_rank + 1) * rows, :].clone())
        p.grad = gr[egtp_rank * rows : (egtp_rank + 1) * rows, :].clone()
        expert.append(p)
    homes = {id(p): h for p, (_, h) in zip(dense, dense_specs)}
    homes.update({id(p): (i % 2, 0) for i, p in enumerate(expert)})
    return dense, dense_w, dense_g, expert, expert_w, expert_g, homes


# ---------------------------------------------------------------------------
# 1-D domain (all ranks one GTP_remat group): bitwise parity with duplicated mode
# ---------------------------------------------------------------------------

_PARITY_CASES = {
    # rows per rank, Q, N params, momentum, nesterov, homes, pad rows
    "even_nesterov": dict(rows=8, Q=16, N=4, momentum=0.95, nesterov=True, homes="even"),
    "even_plain": dict(rows=4, Q=32, N=8, momentum=0.9, nesterov=False, homes="even"),
    "square_nesterov": dict(rows=16, Q=64, N=4, momentum=0.95, nesterov=True, homes="even"),
    "uneven_homes": dict(rows=4, Q=8, N=3, momentum=0.95, nesterov=True, homes="uneven"),
    "no_homes_fallback": dict(rows=16, Q=24, N=3, momentum=0.95, nesterov=False, homes=None),
    "gtp_padding": dict(rows=4, Q=24, N=1, momentum=0.95, nesterov=False, homes="even", pad=3),
}


@pytest.mark.parametrize("case", list(_PARITY_CASES), ids=list(_PARITY_CASES))
def test_step_matches_duplicated_mode(case):
    """Two steps of LayerShardedMuon equal all_gather + NS + reshard bitwise.

    Cases: even home assignment, uneven assignment (idle ranks), no homes at all (the
    TensorParallelMuon fallback), and GTP alignment padding (stripped before NS, scale on
    the true dims, restored before the reverse exchange).
    """
    c = _PARITY_CASES[case]
    _require_multi_rank()
    S, r = dist.get_world_size(), dist.get_rank()
    if c["homes"] == "uneven" and S < 4:
        pytest.skip("uneven assignment needs >= 4 ranks")
    pad, lr, momentum, nesterov = c.get("pad", 0), 1e-2, c["momentum"], c["nesterov"]

    torch.manual_seed(_SEED + 10)
    P = c["rows"] * S - pad  # clean rows; padded back to rows * S when pad > 0
    full_w = [torch.randn(P, c["Q"]) for _ in range(c["N"])]
    step_grads = [[torch.randn(P, c["Q"]) for _ in range(c["N"])] for _ in range(2)]

    def _shard(t):
        return _padded_shard(t, S, r, pad)

    params = [_gtp_param(_shard(w)) for w in full_w]
    for p in params:
        if pad:
            p.pad_length = pad
    extra = {"pg_collection": _PGStub(gtp_remat=_world())} if c["homes"] is None else {}
    opt = _muon(params, momentum=momentum, nesterov=nesterov, gtp_remat_group=_world(), **extra)
    if c["homes"] == "even":
        opt.set_param_ns_homes({id(p): (i % S, 0) for i, p in enumerate(params)})
    elif c["homes"] == "uneven":
        opt.set_param_ns_homes({id(p): (i, 0) for i, p in enumerate(params)})  # ranks >= N idle

    ref_w = [_shard(w) for w in full_w]
    ref_m = [torch.zeros_like(w) for w in ref_w]
    for step in range(2):
        for p, g in zip(params, step_grads[step]):
            p.grad = _shard(g)
        opt.step()
        for i in range(c["N"]):
            g_shard = _shard(step_grads[step][i])
            ref_m[i].lerp_(g_shard, 1 - momentum)
            eff = g_shard.lerp(ref_m[i], momentum) if nesterov else ref_m[i]
            gathered = [torch.zeros_like(eff) for _ in range(S)]
            dist.all_gather(gathered, eff.contiguous(), group=_world())
            full = torch.cat(gathered, dim=0)
            clean = full[:-pad] if pad else full
            orth = newton_schulz(clean.float(), steps=5, coefficient_type="quintic")
            orth = orth * (max(clean.shape) ** 0.5) * 1.0  # optimizer's multiply order
            restored = torch.nn.functional.pad(orth, (0, 0, 0, pad)) if pad else orth
            rows = restored.shape[0] // S
            ref_w[i].add_(restored[r * rows : (r + 1) * rows], alpha=-lr)
            assert torch.equal(params[i].data, ref_w[i]), (
                f"{case}: param {i} on rank {r} at step {step} differs, "
                f"max_diff={(params[i].data - ref_w[i]).abs().max().item():.2e}"
            )
    if pad and r == S - 1:
        assert torch.equal(params[0].data[-pad:], torch.zeros(pad, c["Q"])), "pad rows moved"


# ---------------------------------------------------------------------------
# 2-D domain (TP 2 x GTP_remat 2): two-stage exchange vs full-matrix reference
# ---------------------------------------------------------------------------


@pytest.mark.launch_on_gb200
@pytest.mark.parametrize("ns_batch", [1, 32], ids=["ns_batch1", "ns_batch32"])
def test_2d_domain_matches_full_matrix_reference(ns_batch):
    """Column-parallel, row-parallel, GTP-only and replicated params mixed in one group.

    Replicated params (sharded by neither axis, e.g. the MoE router) must skip both
    exchanges and stay bitwise equal across the domain.
    """
    _require_four_ranks("Requires exactly 4 ranks (TP=2 x GTP=2)")
    _require_batched_ns(ns_batch)
    r = dist.get_rank()
    t, g = r % 2, r // 2
    tp_group, gtp_remat_group = _get_2d_groups()
    assert dist.get_rank(tp_group) == t and dist.get_rank(gtp_remat_group) == g
    torch.manual_seed(_SEED + 40)

    # (shape, partition_dim, gtp_sharded, (g_home, t_home))
    specs = [
        ((32, 16), 0, True, (0, 1)),
        ((16, 32), 1, True, (1, 0)),
        ((8, 16), None, True, (1, 1)),
        ((32, 16), 0, True, (1, 0)),
        ((12, 20), None, False, (1, 0)),
        ((20, 12), None, False, (0, 1)),
        ((12, 20), None, False, (0, 0)),  # same shape as the first replicated: batched together
    ]
    full_w = [torch.randn(*s) for s, _, _, _ in specs]
    full_g = [torch.randn(*s) for s, _, _, _ in specs]
    params = []
    for (_, pd, gtp, _), w, gr in zip(specs, full_w, full_g):
        shard = _shard_2d(w, pd, t, g, gtp)
        p = _gtp_param(shard) if gtp else torch.nn.Parameter(shard)
        p.grad = _shard_2d(gr, pd, t, g, gtp)
        if pd is not None:
            p.partition_dim = pd
        params.append(p)

    opt = _muon(params, gtp_remat_group=gtp_remat_group, tp_group=tp_group, ns_batch_size=ns_batch)
    opt.set_param_ns_homes({id(p): s[3] for p, s in zip(params, specs)})
    opt.step()

    for i, ((_, pd, gtp, _), w, gr) in enumerate(zip(specs, full_w, full_g)):
        torch.testing.assert_close(
            params[i].data,
            _shard_2d(_reference_step(w, gr), pd, t, g, gtp),
            atol=1e-4,
            rtol=1e-4,
            msg=lambda m, i=i, pd=pd: f"param {i} (pd={pd}) on rank {r} (t={t}, g={g})\n\n{m}",
        )
    for i, (_, _, gtp, _) in enumerate(specs):
        if gtp:
            continue
        gathered = [torch.empty_like(params[i].data) for _ in range(dist.get_world_size())]
        dist.all_gather(gathered, params[i].data.contiguous())
        for other_rank, other in enumerate(gathered):
            assert torch.equal(
                other, gathered[0]
            ), f"replicated param {i} diverged on rank {other_rank}"


@pytest.mark.launch_on_gb200
@pytest.mark.parametrize("ns_batch", [1, 8], ids=["ns_batch1", "ns_batch8"])
def test_dense_and_expert_groups_use_their_own_domains(ns_batch):
    """Dense params route over (gtp_remat, tp) and expert params over (egtp, None) in one
    step; routing experts over the dense groups would silently corrupt them."""
    _require_four_ranks()
    _require_batched_ns(ns_batch)
    r = dist.get_rank()
    t, g = r % 2, r // 2
    ep_rank, egtp_rank = r // 2, r % 2
    tp_group, gtp_remat_group = _get_2d_groups()
    egtp_group = _get_expert_group()
    assert dist.get_rank(egtp_group) == egtp_rank

    dense, dense_w, dense_g, expert, expert_w, expert_g, homes = _dense_and_expert_params(
        t, g, ep_rank, egtp_rank, n_same_experts=9
    )
    opt = _muon(
        [{"params": dense}, {"params": expert}],
        gtp_remat_group=gtp_remat_group,
        tp_group=tp_group,
        ns_batch_size=ns_batch,
    )
    opt.set_group_process_groups({0: (gtp_remat_group, tp_group), 1: (egtp_group, None)})
    opt.set_param_ns_homes(homes)
    opt.step()

    for i, (w, gr) in enumerate(zip(dense_w, dense_g)):
        torch.testing.assert_close(
            dense[i].data,
            _shard_2d(_reference_step(w, gr), 0, t, g),
            atol=1e-4,
            rtol=1e-4,
            msg=lambda m, i=i: f"dense param {i} on rank {r} (t={t}, g={g})\n\n{m}",
        )
    for i, (w, gr) in enumerate(zip(expert_w, expert_g)):
        rows = w.shape[0] // 2
        torch.testing.assert_close(
            expert[i].data,
            _reference_step(w, gr)[egtp_rank * rows : (egtp_rank + 1) * rows, :],
            atol=1e-4,
            rtol=1e-4,
            msg=lambda m, i=i: (
                f"expert param {i} on rank {r} (ep={ep_rank}, egtp={egtp_rank})\n\n{m}"
            ),
        )


@pytest.mark.launch_on_gb200
def test_concurrent_groups_match_serial_bitwise():
    """Per-group CUDA streams only reorder work across groups, so any difference is a
    synchronization bug, not rounding."""
    cuda = os.environ.get("TEST_DEVICE", "cuda" if torch.cuda.is_available() else "cpu") == "cuda"
    if not cuda or dist.get_world_size() != 4:
        pytest.skip("Requires exactly 4 ranks on CUDA (streams are a no-op on CPU)")
    r = dist.get_rank()
    t, g = r % 2, r // 2
    ep_rank, egtp_rank = r // 2, r % 2
    tp_group, gtp_remat_group = _get_2d_groups()
    egtp_group = _get_expert_group()

    def _run(concurrent):
        dense, _, _, expert, _, _, homes = _dense_and_expert_params(
            t, g, ep_rank, egtp_rank, n_same_experts=2
        )
        opt = _muon(
            [{"params": dense}, {"params": expert}],
            gtp_remat_group=gtp_remat_group,
            tp_group=tp_group,
            concurrent_groups=concurrent,
        )
        opt.set_group_process_groups({0: (gtp_remat_group, tp_group), 1: (egtp_group, None)})
        opt.set_param_ns_homes(homes)
        for _ in range(2):  # the second step reads momentum the first one wrote
            opt.step()
        torch.cuda.synchronize()
        return [p.data.clone() for p in dense + expert]

    for i, (a, b) in enumerate(zip(_run(False), _run(True))):
        assert torch.equal(a, b), (
            f"param {i} differs between serial and concurrent groups on rank {r}: "
            f"max |diff| = {(a - b).abs().max().item():.3e}"
        )


@pytest.mark.launch_on_gb200
def test_degenerate_domain_group_falls_back_to_local_ns():
    """A group whose (GTP_remat x TP) domain is a single rank runs plain local NS."""
    _require_four_ranks()
    r = dist.get_rank()
    tp_group, gtp_remat_group = _get_2d_groups()

    torch.manual_seed(_SEED + 80 + r)  # distinct per rank: purely local math
    w, g = torch.randn(12, 8), torch.randn(12, 8)
    p = torch.nn.Parameter(w.clone())
    p.grad = g.clone()
    # A second group with a real domain keeps the layer-sharded path active.
    torch.manual_seed(_SEED + 90)
    other = _gtp_param(_shard_2d(torch.randn(32, 16), 0, r % 2, r // 2))
    other.grad = _shard_2d(torch.randn(32, 16), 0, r % 2, r // 2)
    other.partition_dim = 0

    opt = _muon(
        [{"params": [other]}, {"params": [p]}], gtp_remat_group=gtp_remat_group, tp_group=tp_group
    )
    opt.set_group_process_groups({0: (gtp_remat_group, tp_group), 1: (None, None)})
    opt.set_param_ns_homes({id(other): (0, 0)})
    opt.step()
    torch.testing.assert_close(p.data, _reference_step(w, g), atol=1e-4, rtol=1e-4)


# ---------------------------------------------------------------------------
# Batched Newton-Schulz (ns_batch_size): baddbmm chunks vs the per-matrix path
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "ns_batch_size,n_same,n_other,home",
    [
        (32, 10, 0, "single"),  # one chunk on one home
        (4, 10, 0, "single"),  # 4 + 4 + 2
        (3, 9, 0, "single"),  # exact multiple
        (4, 6, 5, "single"),  # two shape buckets
        (32, 6, 3, "spread"),  # homes round-robin, each batches its own few
    ],
    ids=["b32_10same", "b4_10same", "b3_9same", "b4_6same_5other", "b32_6same_3other_spread"],
)
def test_batched_matches_unbatched(ns_batch_size, n_same, n_other, home):
    """Batched NS equals the per-matrix path up to kernel-level rounding."""
    _require_multi_rank()
    _require_batched_ns(ns_batch_size)
    S, r = dist.get_world_size(), dist.get_rank()
    shapes = [(8 * S, 12)] * n_same + [(4 * S, 16)] * n_other

    torch.manual_seed(_SEED + 100)
    full_w = [torch.randn(*s) for s in shapes]
    full_g = [torch.randn(*s) for s in shapes]

    def _shard(full):
        n = full.size(0) // S
        return full[r * n : (r + 1) * n, :].clone()

    def _run(batch_size):
        params = []
        for w, g in zip(full_w, full_g):
            p = _gtp_param(_shard(w))
            p.grad = _shard(g)
            params.append(p)
        opt = _muon(params, gtp_remat_group=_world(), ns_batch_size=batch_size)
        opt.set_param_ns_homes(
            {id(p): ((i % S, 0) if home == "spread" else (0, 0)) for i, p in enumerate(params)}
        )
        opt.step()
        return [p.data.clone() for p in params]

    for i, (a, b) in enumerate(zip(_run(ns_batch_size), _run(1))):
        torch.testing.assert_close(
            a,
            b,
            atol=1e-5,
            rtol=1e-5,
            msg=lambda m, i=i: (
                f"batched vs unbatched mismatch for param {i} {shapes[i]} on rank {r}\n\n{m}"
            ),
        )


@pytest.mark.launch_on_gb200
def test_tp_replicated_ns_chunking_is_column_invariant():
    """nsb > 1 must not diverge the TP replicas of a GTP_remat-only param.

    Each TP column orthogonalizes such a param independently, so its batch chunking
    must not depend on the column's own TP-sharded params: here the canary shares a shape
    bucket with a TP-sharded param homed on column 0 only.
    """
    _require_four_ranks("Requires exactly 4 ranks (TP=2 x GTP_remat=2)")
    _require_batched_ns(8)
    r = dist.get_rank()
    t, g = r % 2, r // 2
    tp_group, gtp_remat_group = _get_2d_groups()
    torch.manual_seed(_SEED + 500)

    p_a = _gtp_param(_shard_2d(torch.randn(32, 16), None, t, g))
    p_a.grad = _shard_2d(torch.randn(32, 16), None, t, g)
    p_b = _gtp_param(_shard_2d(torch.randn(32, 16), 0, t, g))
    p_b.grad = _shard_2d(torch.randn(32, 16), 0, t, g)
    p_b.partition_dim = 0

    opt = _muon(
        [p_a, p_b],
        nesterov=False,
        gtp_remat_group=gtp_remat_group,
        tp_group=tp_group,
        ns_batch_size=8,
    )
    opt.set_param_ns_homes({id(p_a): (0, 0), id(p_b): (0, 0)})
    opt.step()

    gathered = [torch.empty_like(p_a.data) for _ in range(2)]
    dist.all_gather(gathered, p_a.data.contiguous(), group=tp_group)
    assert torch.equal(gathered[0], gathered[1]), (
        f"TP replicas diverged on rank {r} (t={t}, g={g}): "
        f"max_diff={(gathered[0] - gathered[1]).abs().max().item():.2e}"
    )


# ---------------------------------------------------------------------------
# GTP alignment padding on the 2-D grid
# ---------------------------------------------------------------------------


@pytest.mark.launch_on_gb200
@pytest.mark.parametrize("pd", [0, 1], ids=["col_parallel", "row_parallel"])
def test_padded_param_2d_two_stage_matches_reference(pd):
    """Padded param on TP2 x GTP2: the pad must be stripped at the stage-1 seam (after TP
    assembly it would sit inside every TP block), scale on the true dims, pad restored."""
    _require_four_ranks()
    tp_group, gtp_remat_group = _get_2d_groups()
    T, G = 2, 2
    r = dist.get_rank()
    t, g = r % T, r // T
    lr, momentum, PAD = 1e-2, 0.95, 3
    # Clean dims are chosen so the TP-local dim 0 is not divisible by G without padding.
    M_full, Q = (26, 24) if pd == 0 else (13, 32)
    M_local = M_full // T if pd == 0 else M_full

    torch.manual_seed(_SEED + 510)
    full_w = torch.randn(M_full, Q)
    step_grads = [torch.randn(M_full, Q) for _ in range(2)]

    def _tp_local(x):
        return (
            x[t * M_local : (t + 1) * M_local, :]
            if pd == 0
            else x[:, t * (Q // T) : (t + 1) * (Q // T)]
        )

    p = _gtp_param(_padded_shard(_tp_local(full_w), G, g, PAD))
    p.partition_dim = pd
    p.pad_length = PAD
    opt = _muon([p], nesterov=False, gtp_remat_group=gtp_remat_group, tp_group=tp_group)
    opt.set_param_ns_homes({id(p): (1, 1)})

    ref_shard = _padded_shard(_tp_local(full_w), G, g, PAD)
    ref_mom = torch.zeros_like(ref_shard)
    for step in range(2):
        g_shard = _padded_shard(_tp_local(step_grads[step]), G, g, PAD)
        p.grad = g_shard.clone()
        opt.step()

        ref_mom.lerp_(g_shard, 1 - momentum)
        g_parts = [torch.zeros_like(ref_mom) for _ in range(G)]
        dist.all_gather(g_parts, ref_mom.contiguous(), group=gtp_remat_group)
        tp_local_clean = torch.cat(g_parts, dim=0)[:-PAD]
        t_parts = [torch.zeros_like(tp_local_clean) for _ in range(T)]
        dist.all_gather(t_parts, tp_local_clean.contiguous(), group=tp_group)
        orth = newton_schulz(
            torch.cat(t_parts, dim=pd).float(), steps=5, coefficient_type="quintic"
        )
        orth = orth * (max(M_full, Q) ** 0.5) * 1.0
        restored = torch.nn.functional.pad(_tp_local(orth), (0, 0, 0, PAD))
        rows = restored.shape[0] // G
        ref_shard.add_(restored[g * rows : (g + 1) * rows], alpha=-lr)
        assert torch.equal(p.data, ref_shard), (
            f"padded 2-D parity mismatch (pd={pd}) on rank {r} step {step}: "
            f"max_diff={(p.data - ref_shard).abs().max().item():.2e}"
        )


# ---------------------------------------------------------------------------
# Guard rails, hooks, routing-plan cache
# ---------------------------------------------------------------------------


@pytest.mark.launch_on_gb200
def test_tp_sharded_without_gtp_marker_is_rejected():
    """A TP-sharded param without is_gtp_weight_remat is replicated across GTP_remat; the
    exchange would concatenate the copies as shards, so step() must raise before any
    collective."""
    _require_four_ranks("Requires exactly 4 ranks (TP=2 x GTP=2)")
    tp_group, gtp_remat_group = _get_2d_groups()
    p = torch.nn.Parameter(torch.randn(8, 6))
    p.partition_dim = 0
    opt = LayerShardedMuon([p], lr=0.1, gtp_remat_group=gtp_remat_group, tp_group=tp_group)
    opt.set_param_ns_homes({id(p): (0, 0)})
    p.grad = torch.randn_like(p)
    with pytest.raises(ValueError, match="not GTP-sharded"):
        opt.step()


def test_weight_update_hooks_called_on_all_paths():
    """pre/post_weight_update_fn_inplace fire once per updated param on the replicated,
    routed and degenerate paths."""
    _require_multi_rank()
    S = dist.get_world_size()
    torch.manual_seed(_SEED + 420)

    routed = [_gtp_param(torch.randn(4, 8)) for _ in range(2)]
    params = routed + [torch.nn.Parameter(torch.randn(6, 6))]
    opt = _muon(params, nesterov=False, gtp_remat_group=_world())
    opt.set_param_ns_homes({id(p): (i % S, 0) for i, p in enumerate(routed)})
    calls = {"pre": 0, "post": 0}
    opt.pre_weight_update_fn_inplace = lambda p, update: calls.__setitem__("pre", calls["pre"] + 1)
    opt.post_weight_update_fn_inplace = lambda p: calls.__setitem__("post", calls["post"] + 1)
    for p in params:
        p.grad = torch.randn_like(p)
    opt.step()
    assert calls == {"pre": 3, "post": 3}, f"hooks missed on the sharded paths: {calls}"

    p2 = torch.nn.Parameter(torch.randn(8, 8))
    opt2 = _muon([p2], nesterov=False, gtp_remat_group=None)
    calls2 = {"pre": 0, "post": 0}
    opt2.pre_weight_update_fn_inplace = lambda p, update: calls2.__setitem__(
        "pre", calls2["pre"] + 1
    )
    opt2.post_weight_update_fn_inplace = lambda p: calls2.__setitem__("post", calls2["post"] + 1)
    p2.grad = torch.randn_like(p2)
    opt2.step()
    assert calls2 == {"pre": 1, "post": 1}, f"hooks missed on the degenerate path: {calls2}"


@pytest.mark.launch_on_gb200
def test_exchange_plan_cache_bitwise_and_reused():
    """Routing plans are reused across steps and reproduce cold-start steps bitwise."""
    _require_four_ranks()
    tp_group, gtp_remat_group = _get_2d_groups()
    r = dist.get_rank()
    t, g = r % 2, r // 2
    torch.manual_seed(_SEED + 700)
    full_w = [torch.randn(16, 12), torch.randn(8, 16)]
    pdims = [0, 1]
    step_grads = [[torch.randn_like(w) for w in full_w] for _ in range(3)]

    def _build():
        params = []
        for w, pd in zip(full_w, pdims):
            p = _gtp_param(_shard_2d(w, pd, t, g))
            p.partition_dim = pd
            params.append(p)
        opt = _muon(params, nesterov=False, gtp_remat_group=gtp_remat_group, tp_group=tp_group)
        opt.set_param_ns_homes({id(params[0]): (0, 1), id(params[1]): (1, 0)})
        return params, opt

    params_a, opt_a = _build()  # warm: plans persist
    params_b, opt_b = _build()  # cold: plans cleared before every step
    plan_ids = None
    for step in range(3):
        for params, opt, cold in ((params_a, opt_a, False), (params_b, opt_b, True)):
            for i, p in enumerate(params):
                p.grad = _shard_2d(step_grads[step][i], pdims[i], t, g)
            if cold:
                opt._exchange_plans.clear()
            opt.step()
        for pa, pb in zip(params_a, params_b):
            assert torch.equal(pa.data, pb.data), f"warm/cold divergence at step {step}"
        current = {tag: id(sub) for tag, sub in opt_a._exchange_plans[0].items() if tag != 'key'}
        assert current, "no plans were cached"
        if plan_ids is None:
            plan_ids = current
        else:
            assert current == plan_ids, "plan dicts were rebuilt on a warm step"


def test_exchange_plan_rebuilds_when_param_set_changes():
    """Dropping a grad changes the routed param set: the plan must rebuild and the
    grad-less param must not move."""
    _require_multi_rank()
    S, r = dist.get_world_size(), dist.get_rank()
    torch.manual_seed(_SEED + 710)
    full_w = [torch.randn(4 * S, 8) for _ in range(2)]

    def _shard(x):
        return x[r * 4 : (r + 1) * 4].clone()

    params = [_gtp_param(_shard(w)) for w in full_w]
    opt = _muon([{'params': params}], nesterov=False, gtp_remat_group=_world())
    opt.set_param_ns_homes({id(p): (i % S, 0) for i, p in enumerate(params)})
    for p, w in zip(params, full_w):
        p.grad = _shard(torch.randn_like(w))
    opt.step()
    key_both = opt._exchange_plans[0]['key']

    params[1].grad = None
    params[0].grad = _shard(torch.randn_like(full_w[0]))
    before = params[1].data.clone()
    opt.step()
    assert opt._exchange_plans[0]['key'] != key_both, "stale plan reused"
    assert torch.equal(params[1].data, before), "grad-less param must not move"
