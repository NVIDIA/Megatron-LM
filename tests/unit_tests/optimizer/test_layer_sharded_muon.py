# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Tests for LayerShardedMuon and the fwd/NS/bwd layer-sharding pipeline.

Each test runs with world_size == 4, treating all 4 ranks as a single GTP group
(or a TP(2) x GTP(2) domain for the 2D tests).

Key correctness invariant:
  LayerShardedMuon.step()  ==  duplicated-mode NS step
  (same momentum update + same NS + same weight update, just a different
  communication pattern)

Launch with torchrun (the fixture initializes the torchrun-managed group):
  torchrun --nproc-per-node=4 -m pytest tests/unit_tests/optimizer/test_layer_sharded_muon.py            # CPU / gloo
  TEST_DEVICE=cuda torchrun --nproc-per-node=4 -m pytest tests/unit_tests/optimizer/test_layer_sharded_muon.py  # GPU / nccl
"""

import os

import pytest
import torch
import torch.distributed as dist

pytest.importorskip("emerging_optimizers", reason="LayerShardedMuon requires emerging-optimizers")

from emerging_optimizers.orthogonalized_optimizers.muon_utils import newton_schulz

from megatron.core.optimizer.layer_sharded_a2a import (
    layer_sharded_all_to_all_bwd,
    layer_sharded_all_to_all_fwd,
)
from megatron.core.optimizer.layer_sharded_muon import LayerShardedMuon, _check_eo_version

try:
    # LayerShardedMuon.__init__ raises on emerging-optimizers < 0.3.0 (no batched
    # 3-D Newton-Schulz); turn that into a module-level skip so CI containers with
    # an older wheel stay green instead of erroring at collection.
    _check_eo_version()
except ImportError as _e:
    pytest.skip(str(_e), allow_module_level=True)
from tests.unit_tests.test_utilities import Utils

_SEED = 42


@pytest.fixture(scope="module", autouse=True)
def _torchrun_dist_init():
    Utils.initialize_model_parallel()
    cuda = os.environ.get("TEST_DEVICE", "cpu") == "cuda"
    if cuda:
        # NCCL path: one GPU per rank; default device makes every torch.randn /
        # torch.zeros in the tests land on this rank's GPU without touching each test.
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        torch.cuda.set_device(local_rank)
        torch.set_default_device(f"cuda:{local_rank}")
    torch.manual_seed(_SEED)
    # Pin to "highest" (full fp32) so both the reference path and LayerShardedMuon.step()
    # use the same NS precision. With "medium", the optimizer's internal
    # fp32_matmul_precision("medium") context overrides the global, but the reference
    # newton_schulz call runs with whatever the ambient global is — causing ~1e-2 BF16
    # rounding differences. Using "highest" throughout avoids this.
    prev_prec = torch.get_float32_matmul_precision()
    torch.set_float32_matmul_precision("highest")
    yield
    torch.set_float32_matmul_precision(prev_prec)
    if cuda:
        torch.set_default_device("cpu")
    Utils.destroy_model_parallel()


def _world():
    # ``dist.group.WORLD`` is only valid after the fixture initialized the process
    # group, and LayerShardedMuon reads None as "no group / size 1" rather than
    # "the default group" — so always resolve it lazily, never at import time.
    world = dist.group.WORLD
    assert world is not None, "process group must be initialized before tests run"
    return world


def _require_multi_rank():
    if dist.get_world_size() < 2:
        pytest.skip("Requires >= 2 ranks (launch with torchrun --nproc-per-node=4)")


def _require_four_ranks(reason="Requires exactly 4 ranks"):
    if dist.get_world_size() != 4:
        pytest.skip(f"{reason} (launch with torchrun --nproc-per-node=4)")


def _gtp_param(t):
    # Megatron tags GTP-row-sharded weights with is_gtp_weight_remat; params
    # without it are whole on every rank and skip the exchanges.
    p = torch.nn.Parameter(t)
    p.is_gtp_weight_remat = True
    return p


# ---------------------------------------------------------------------------
# Subgroup helpers (created at most once per process)
# ---------------------------------------------------------------------------

_TP_GROUP = None
_GTP_GROUP = None
_EGTP_GROUP = None


def _get_2d_groups():
    """Create TP(2) x GTP(2) subgroups once. Convention: rank = g * T + t (tp innermost)."""
    global _TP_GROUP, _GTP_GROUP
    if _TP_GROUP is None:
        _TP_GROUP, _ = dist.new_subgroups_by_enumeration([[0, 1], [2, 3]])
        _GTP_GROUP, _ = dist.new_subgroups_by_enumeration([[0, 2], [1, 3]])
    return _TP_GROUP, _GTP_GROUP


def _get_expert_group():
    """Create the expert GTP(2) subgroup: {0,1} and {2,3}.

    Models the MoE layout where each EP group holds a *different* set of expert
    matrices, sharded along dim 0 across EGTP. Deliberately a different partition
    of the world than the dense GTP group, so using the wrong one is detectable.
    """
    global _EGTP_GROUP
    if _EGTP_GROUP is None:
        _EGTP_GROUP, _ = dist.new_subgroups_by_enumeration([[0, 1], [2, 3]])
    return _EGTP_GROUP


# ---------------------------------------------------------------------------
# Direct API parity: all_to_all fwd + NS + all_to_all bwd == all_gather + NS
# This tests the primitives together WITHOUT optimizer machinery.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "P,Q,N", [(32, 16, 4), (16, 32, 8), (64, 64, 4)], ids=["P32_Q16_N4", "P16_Q32_N8", "P64_Q64_N4"]
)
def test_layer_sharding_pipeline_matches_allgather_ns(P, Q, N):
    """fwd_all_to_all -> NS -> bwd_all_to_all == all_gather -> NS -> reshard."""
    _require_multi_rank()
    S = dist.get_world_size()
    r = dist.get_rank()
    if P % S != 0 or N % S != 0:
        pytest.skip(f"P or N not divisible by world_size={S}")
    P_shard = P // S

    torch.manual_seed(_SEED + 30)
    # Same full momentums on all ranks (same seed)
    full_momentums = [torch.randn(P, Q) for _ in range(N)]
    # Each rank holds its GTP row shard of each param
    local_shards = [m[r * P_shard : (r + 1) * P_shard, :].clone() for m in full_momentums]

    assignment = {i: i % S for i in range(N)}

    # --- Layer sharding path ---
    complete, my_indices = layer_sharded_all_to_all_fwd(
        local_shards, assignment, r, S, _world(), gtp_dim=0
    )
    ns_results = [newton_schulz(m.float(), steps=5, coefficient_type="quintic") for m in complete]
    update_shards = layer_sharded_all_to_all_bwd(
        ns_results, my_indices, local_shards, assignment, r, S, _world(), gtp_dim=0
    )

    # --- Reference path: all_gather + NS + reshard ---
    for i in range(N):
        ref_shards = [torch.zeros_like(local_shards[i]) for _ in range(S)]
        dist.all_gather(ref_shards, local_shards[i].contiguous(), group=_world())
        ref_full = torch.cat(ref_shards, dim=0)
        ref_orth = newton_schulz(ref_full.float(), steps=5, coefficient_type="quintic")
        ref_shard = ref_orth[r * P_shard : (r + 1) * P_shard, :]

        assert update_shards[i] is not None
        torch.testing.assert_close(
            update_shards[i],
            ref_shard,
            atol=1e-6,
            rtol=0,
            msg=lambda m, i=i, ref_shard=ref_shard: (
                f"NS pipeline mismatch for param {i} on rank {r}: "
                f"max_diff={(update_shards[i] - ref_shard).abs().max().item():.2e}\n\n{m}"
            ),
        )


# ---------------------------------------------------------------------------
# Parity test: LayerShardedMuon.step() == duplicated-mode NS
#
# Duplicated mode: each rank all-gathers momentum, runs NS on the full matrix,
# takes its shard.
# Layer sharding:  one all_to_all -> NS home gets full matrix -> NS locally ->
# backward all_to_all.
# Both must produce identical (P/S, Q) update shards on every rank.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "P,Q,N,momentum,nesterov",
    [(32, 16, 4, 0.95, True), (16, 32, 8, 0.9, False), (64, 64, 4, 0.95, True)],
    ids=["P32_Q16_N4_nesterov", "P16_Q32_N8_plain", "P64_Q64_N4_nesterov"],
)
def test_layer_sharded_matches_duplicated(P, Q, N, momentum, nesterov):
    """LayerShardedMuon step == all_gather + NS + reshard (duplicated mode)."""
    _require_multi_rank()
    S = dist.get_world_size()
    r = dist.get_rank()
    if P % S != 0:
        pytest.skip(f"P={P} not divisible by world_size={S}")
    if N % S != 0:
        pytest.skip(f"N={N} not divisible by world_size={S}")

    P_shard = P // S
    lr = 1e-2

    torch.manual_seed(_SEED + 10)

    # Full data (same on all ranks so we can check every rank's shard)
    full_weights = [torch.randn(P, Q) for _ in range(N)]
    full_grads = [torch.randn(P, Q) for _ in range(N)]
    full_momentums = [torch.zeros(P, Q) for _ in range(N)]  # start from zero

    # Each rank's local GTP shard
    def _shard(t):
        return t[r * P_shard : (r + 1) * P_shard, :].clone()

    local_weights = [_shard(w) for w in full_weights]
    local_grads = [_shard(g) for g in full_grads]
    local_momentums = [_shard(m) for m in full_momentums]

    # ---------------------------------------------------------------
    # Reference: duplicated mode
    #   For each param: all_gather momentum shards -> NS on full (P, Q) -> take my shard
    # ---------------------------------------------------------------
    ref_updated_weights = []
    for i in range(N):
        # Momentum update (local, per rank)
        m_shard = local_momentums[i].clone()
        g_shard = local_grads[i]
        m_shard.lerp_(g_shard, 1 - momentum)
        if nesterov:
            eff_grad = g_shard.lerp(m_shard, momentum)
        else:
            eff_grad = m_shard

        # All-gather to get full (P, Q) momentum across GTP group
        shards = [torch.zeros_like(eff_grad) for _ in range(S)]
        dist.all_gather(shards, eff_grad.contiguous(), group=_world())
        full_eff_grad = torch.cat(shards, dim=0)  # (P, Q)

        # NS on full matrix
        full_orth = newton_schulz(full_eff_grad.float(), steps=5, coefficient_type="quintic")
        # Scale: max(P, Q)^0.5 (spectral mode)
        scale = max(P, Q) ** 0.5
        full_orth = full_orth * scale

        # Take my shard and update weight
        my_orth_shard = full_orth[r * P_shard : (r + 1) * P_shard, :]
        w_updated = local_weights[i].clone()
        w_updated.add_(my_orth_shard, alpha=-lr)
        ref_updated_weights.append(w_updated)

    # ---------------------------------------------------------------
    # Test: LayerShardedMuon
    # ---------------------------------------------------------------
    # Create params and set grad
    params = []
    for i in range(N):
        p = _gtp_param(local_weights[i].clone())
        p.grad = local_grads[i].clone()
        params.append(p)

    optimizer = LayerShardedMuon(
        params,
        lr=lr,
        momentum=momentum,
        nesterov=nesterov,
        weight_decay=0.0,
        coefficient_type="quintic",
        num_ns_steps=5,
        scale_mode="spectral",
        extra_scale_factor=1.0,
        fp32_matmul_prec="highest",  # match the module fixture precision
        gtp_group=_world(),
    )

    # Set NS home assignment: param i -> GTP rank i % S
    assignment = {id(p): (i % S, 0) for i, p in enumerate(params)}
    optimizer.set_param_ns_homes(assignment)

    optimizer.step()

    # ---------------------------------------------------------------
    # Compare
    # ---------------------------------------------------------------
    for i, (p, ref_w) in enumerate(zip(params, ref_updated_weights)):
        torch.testing.assert_close(
            p.data,
            ref_w,
            atol=1e-4,
            rtol=1e-4,
            msg=lambda m, i=i, p=p, ref_w=ref_w: (
                f"Weight mismatch for param {i} on rank {r}: "
                f"max_diff={(p.data - ref_w).abs().max().item():.2e}\n\n{m}"
            ),
        )


def test_unequal_assignment_still_correct():
    """When params are assigned unevenly (some ranks get more), result is still correct.

    Uses N=3 params with S=4 ranks: rank 0 gets params 0,3; rank 1 gets param 1;
    rank 2 gets param 2; rank 3 gets nothing.
    """
    _require_multi_rank()
    S = dist.get_world_size()
    if S < 4:
        pytest.skip("Requires world_size >= 4 for uneven test")
    r = dist.get_rank()
    P, Q = 16, 8
    P_shard = P // S
    N = 3
    lr = 1e-2

    torch.manual_seed(_SEED + 20)

    full_weights = [torch.randn(P, Q) for _ in range(N)]
    full_grads = [torch.randn(P, Q) for _ in range(N)]

    def _shard(t):
        return t[r * P_shard : (r + 1) * P_shard, :].clone()

    local_weights = [_shard(w) for w in full_weights]
    local_grads = [_shard(g) for g in full_grads]

    # Custom assignment: 0->rank0, 1->rank1, 2->rank2 (rank3 gets nothing)
    ns_homes = {0: 0, 1: 1, 2: 2}

    # Reference via duplicated mode
    momentum = 0.95
    ref_updated = []
    for i in range(N):
        m_shard = torch.zeros_like(local_grads[i])
        m_shard.lerp_(local_grads[i], 1 - momentum)
        eff_grad = local_grads[i].lerp(m_shard, momentum)  # nesterov=True

        shards = [torch.zeros_like(eff_grad) for _ in range(S)]
        dist.all_gather(shards, eff_grad.contiguous(), group=_world())
        full_g = torch.cat(shards, dim=0)

        full_orth = newton_schulz(full_g.float(), steps=5, coefficient_type="quintic")
        scale = max(P, Q) ** 0.5
        full_orth = full_orth * scale

        my_shard = full_orth[r * P_shard : (r + 1) * P_shard, :]
        w_ref = local_weights[i].clone()
        w_ref.add_(my_shard, alpha=-lr)
        ref_updated.append(w_ref)

    # LayerShardedMuon
    params = []
    for i in range(N):
        p = _gtp_param(local_weights[i].clone())
        p.grad = local_grads[i].clone()
        params.append(p)

    optimizer = LayerShardedMuon(
        params,
        lr=lr,
        momentum=momentum,
        nesterov=True,
        weight_decay=0.0,
        coefficient_type="quintic",
        num_ns_steps=5,
        scale_mode="spectral",
        extra_scale_factor=1.0,
        fp32_matmul_prec="highest",
        gtp_group=_world(),
    )
    optimizer.set_param_ns_homes({id(params[i]): (ns_homes[i], 0) for i in range(N)})
    optimizer.step()

    for i, (p, ref_w) in enumerate(zip(params, ref_updated)):
        torch.testing.assert_close(
            p.data,
            ref_w,
            atol=1e-4,
            rtol=1e-4,
            msg=lambda m, i=i: f"Uneven assignment mismatch for param {i} on rank {r}\n\n{m}",
        )


# ---------------------------------------------------------------------------
# 2D parity test: GTP x TP domain (two-stage all_to_all) == full-matrix NS
#
# Covers column-parallel (partition_dim=0), row-parallel (partition_dim=1), and
# non-TP-sharded (partition_dim=None) params mixed in one param group, with
# heterogeneous shapes.
# ---------------------------------------------------------------------------


def test_mixed_partition_dims_match_full_matrix_reference():
    _require_four_ranks("Requires exactly 4 ranks (TP=2 x GTP=2)")
    T, G = 2, 2
    r = dist.get_rank()
    t_rank, g_rank = r % T, r // T
    tp_group, gtp_group = _get_2d_groups()
    assert dist.get_rank(tp_group) == t_rank
    assert dist.get_rank(gtp_group) == g_rank

    lr, momentum = 1e-2, 0.95
    torch.manual_seed(_SEED + 40)

    # (full_shape, partition_dim, (g_home, t_home))
    specs = [
        ((32, 16), 0, (0, 1)),  # column-parallel
        ((16, 32), 1, (1, 0)),  # row-parallel
        ((8, 16), None, (1, 1)),  # not TP-sharded (t_home ignored)
        ((32, 16), 0, (1, 0)),  # second col-parallel, different home
    ]

    full_weights = [torch.randn(*shape) for shape, _, _ in specs]
    full_grads = [torch.randn(*shape) for shape, _, _ in specs]

    def _shard(full, pd):
        P, Q = full.shape
        if pd == 0:
            rows = P // (T * G)
            start = t_rank * (P // T) + g_rank * rows
            return full[start : start + rows, :].clone()
        if pd == 1:
            return full[
                g_rank * (P // G) : (g_rank + 1) * (P // G),
                t_rank * (Q // T) : (t_rank + 1) * (Q // T),
            ].clone()
        return full[g_rank * (P // G) : (g_rank + 1) * (P // G), :].clone()

    params = []
    for (shape, pd, _), w_full, g_full in zip(specs, full_weights, full_grads):
        p = _gtp_param(_shard(w_full, pd))
        p.grad = _shard(g_full, pd)
        if pd is not None:
            p.partition_dim = pd
        params.append(p)

    optimizer = LayerShardedMuon(
        params,
        lr=lr,
        momentum=momentum,
        nesterov=True,
        weight_decay=0.0,
        coefficient_type="quintic",
        num_ns_steps=5,
        scale_mode="spectral",
        extra_scale_factor=1.0,
        fp32_matmul_prec="highest",
        gtp_group=gtp_group,
        tp_group=tp_group,
    )
    optimizer.set_param_ns_homes({id(p): spec[2] for p, spec in zip(params, specs)})
    optimizer.step()

    # Reference: full-matrix Muon math on the (identical-on-all-ranks) full tensors.
    for i, ((shape, pd, _), w_full, g_full) in enumerate(zip(specs, full_weights, full_grads)):
        P, Q = shape
        m_full = torch.zeros_like(g_full).lerp_(g_full, 1 - momentum)
        eff = g_full.lerp(m_full, momentum)  # nesterov
        orth = newton_schulz(eff.float(), steps=5, coefficient_type="quintic")
        w_new = w_full - lr * (max(P, Q) ** 0.5) * orth
        torch.testing.assert_close(
            params[i].data,
            _shard(w_new, pd),
            atol=1e-4,
            rtol=1e-4,
            msg=lambda m, i=i, pd=pd: (
                f"2D parity mismatch for param {i} (pd={pd}) on rank {r} "
                f"(t={t_rank}, g={g_rank})\n\n{m}"
            ),
        )


# ---------------------------------------------------------------------------
# Replicated params: sharded by neither GTP nor TP (MoE router / latent projections)
#
# TE leaves these whole on every rank of the domain, so they must skip both
# exchanges: routing one through stage 1 would concatenate G copies into a
# (G*P, Q) matrix and silently corrupt the update.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("ns_batch", [1, 32], ids=["ns_batch1", "ns_batch32"])
@pytest.mark.parametrize("fused", [False, True], ids=["two_stage", "fused"])
def test_replicated_mixed_with_sharded(fused, ns_batch):
    _require_four_ranks("Requires exactly 4 ranks (TP=2 x GTP=2)")
    T, G = 2, 2
    r = dist.get_rank()
    t_rank, g_rank = r % T, r // T
    tp_group, gtp_group = _get_2d_groups()
    lr, momentum = 1e-2, 0.95
    torch.manual_seed(_SEED + 60)

    # (full_shape, partition_dim, gtp_sharded, (g_home, t_home))
    specs = [
        ((32, 16), 0, True, (0, 1)),  # GTP + TP sharded
        ((12, 20), None, False, (1, 0)),  # replicated, router-like
        ((20, 12), None, False, (0, 1)),  # replicated, latent-proj-like
        # Same shape as the router-like one: with ns_batch > 1 the replicated
        # path must stack these two, which is where its launch cost goes.
        ((12, 20), None, False, (0, 0)),
        ((16, 32), 1, True, (1, 1)),  # row-parallel
    ]
    full_w = [torch.randn(*s) for s, _, _, _ in specs]
    full_g = [torch.randn(*s) for s, _, _, _ in specs]

    def _shard(full, pd, gtp):
        P, Q = full.shape
        if not gtp and pd is None:
            return full.clone()
        if pd == 0:
            rows = P // (T * G)
            start = t_rank * (P // T) + g_rank * rows
            return full[start : start + rows, :].clone()
        if pd == 1:
            return full[
                g_rank * (P // G) : (g_rank + 1) * (P // G),
                t_rank * (Q // T) : (t_rank + 1) * (Q // T),
            ].clone()
        return full[g_rank * (P // G) : (g_rank + 1) * (P // G), :].clone()

    params = []
    for (_, pd, gtp, _), w, g in zip(specs, full_w, full_g):
        shard = _shard(w, pd, gtp)
        p = _gtp_param(shard) if gtp else torch.nn.Parameter(shard)
        p.grad = _shard(g, pd, gtp)
        if pd is not None:
            p.partition_dim = pd
        params.append(p)

    optimizer = LayerShardedMuon(
        params,
        lr=lr,
        momentum=momentum,
        nesterov=True,
        weight_decay=0.0,
        coefficient_type="quintic",
        num_ns_steps=5,
        scale_mode="spectral",
        extra_scale_factor=1.0,
        fp32_matmul_prec="highest",
        gtp_group=gtp_group,
        tp_group=tp_group,
        fused_group=_world() if fused else None,
        ns_batch_size=ns_batch,
    )
    optimizer.set_param_ns_homes({id(p): s[3] for p, s in zip(params, specs)})
    optimizer.step()

    for i, ((shape, pd, gtp, _), w, g) in enumerate(zip(specs, full_w, full_g)):
        P, Q = shape
        m = torch.zeros_like(g).lerp_(g, 1 - momentum)
        eff = g.lerp(m, momentum)  # nesterov
        orth = newton_schulz(eff.float(), steps=5, coefficient_type="quintic")
        w_new = w - lr * (max(P, Q) ** 0.5) * orth
        torch.testing.assert_close(
            params[i].data,
            _shard(w_new, pd, gtp),
            atol=1e-4,
            rtol=1e-4,
            msg=lambda default, i=i, pd=pd, gtp=gtp: (
                f"param {i} (pd={pd}, gtp_sharded={gtp}) on rank {r} "
                f"(t={t_rank}, g={g_rank})\n\n{default}"
            ),
        )

    # Replicated params are updated independently on every rank; the domain must
    # still agree bitwise, or the replicas drift apart over training.
    for i, (_, _, gtp, _) in enumerate(specs):
        if gtp:
            continue
        gathered = [torch.empty_like(params[i].data) for _ in range(dist.get_world_size())]
        dist.all_gather(gathered, params[i].data.contiguous())
        for other_rank, other in enumerate(gathered):
            torch.testing.assert_close(
                other,
                gathered[0],
                atol=0,
                rtol=0,
                msg=lambda default, i=i, o=other_rank: (
                    f"replicated param {i} diverged on rank {o}\n\n{default}"
                ),
            )


# ---------------------------------------------------------------------------
# Per-param-group process groups: dense and expert params in one optimizer
#
# Regression test for the expert-domain bug: expert params are sharded over
# (expt_gtp_remat, expt_tp), a different partition of the world than the dense
# (gtp_remat, tp). Routing them through the dense groups silently produces the
# wrong result, so this exercises both domains inside a single step().
# ---------------------------------------------------------------------------


def test_dense_and_expert_groups_use_their_own_domains():
    _require_four_ranks()
    T, G = 2, 2
    r = dist.get_rank()
    t_rank, g_rank = r % T, r // T
    tp_group, gtp_group = _get_2d_groups()
    egtp_group = _get_expert_group()
    # Expert layout: EP group = which experts this rank holds, EGTP = row shard.
    ep_rank, egtp_rank = r // 2, r % 2
    assert dist.get_rank(egtp_group) == egtp_rank

    lr, momentum = 1e-2, 0.95

    # --- Dense params: TP(2) x GTP(2), column-parallel ---
    torch.manual_seed(_SEED + 60)
    dense_specs = [((32, 16), (0, 1)), ((32, 16), (1, 0))]
    dense_w = [torch.randn(*s) for s, _ in dense_specs]
    dense_g = [torch.randn(*s) for s, _ in dense_specs]

    def _dense_shard(full):
        P = full.size(0)
        rows = P // (T * G)
        start = t_rank * (P // T) + g_rank * rows
        return full[start : start + rows, :].clone()

    dense_params = []
    for w, g in zip(dense_w, dense_g):
        p = _gtp_param(_dense_shard(w))
        p.grad = _dense_shard(g)
        p.partition_dim = 0
        dense_params.append(p)

    # --- Expert params: EGTP(2) only, no TP. Each EP group has its own experts. ---
    torch.manual_seed(_SEED + 70 + ep_rank)
    expert_shapes = [(16, 8), (16, 8), (8, 16)]
    expert_w = [torch.randn(*s) for s in expert_shapes]
    expert_g = [torch.randn(*s) for s in expert_shapes]
    expert_homes = [0, 1, 1]

    def _expert_shard(full):
        rows = full.size(0) // G
        return full[egtp_rank * rows : (egtp_rank + 1) * rows, :].clone()

    expert_params = []
    for w, g in zip(expert_w, expert_g):
        p = _gtp_param(_expert_shard(w))
        p.grad = _expert_shard(g)
        expert_params.append(p)  # no partition_dim: not TP-sharded

    optimizer = LayerShardedMuon(
        [{"params": dense_params}, {"params": expert_params}],
        lr=lr,
        momentum=momentum,
        nesterov=True,
        weight_decay=0.0,
        coefficient_type="quintic",
        num_ns_steps=5,
        scale_mode="spectral",
        extra_scale_factor=1.0,
        fp32_matmul_prec="highest",
        gtp_group=gtp_group,
        tp_group=tp_group,
    )
    optimizer.set_group_process_groups({0: (gtp_group, tp_group), 1: (egtp_group, None)})
    homes = {id(p): h for p, (_, h) in zip(dense_params, dense_specs)}
    homes.update({id(p): (h, 0) for p, h in zip(expert_params, expert_homes)})
    optimizer.set_param_ns_homes(homes)

    optimizer.step()

    def _reference(w_full, g_full):
        m = torch.zeros_like(g_full).lerp_(g_full, 1 - momentum)
        eff = g_full.lerp(m, momentum)  # nesterov
        orth = newton_schulz(eff.float(), steps=5, coefficient_type="quintic")
        P, Q = w_full.shape
        return w_full - lr * (max(P, Q) ** 0.5) * orth

    for i, (w, g) in enumerate(zip(dense_w, dense_g)):
        torch.testing.assert_close(
            dense_params[i].data,
            _dense_shard(_reference(w, g)),
            atol=1e-4,
            rtol=1e-4,
            msg=lambda m, i=i: (
                f"dense param {i} mismatch on rank {r} (t={t_rank}, g={g_rank})\n\n{m}"
            ),
        )
    for i, (w, g) in enumerate(zip(expert_w, expert_g)):
        torch.testing.assert_close(
            expert_params[i].data,
            _expert_shard(_reference(w, g)),
            atol=1e-4,
            rtol=1e-4,
            msg=lambda m, i=i: (
                f"expert param {i} mismatch on rank {r} " f"(ep={ep_rank}, egtp={egtp_rank})\n\n{m}"
            ),
        )


def test_concurrent_groups_match_serial_bitwise():
    """Running the groups on separate streams must not change a single bit.

    concurrent_groups only reorders work ACROSS groups; the ops and their order
    WITHIN each group are untouched, so any difference is a synchronization bug
    (a group reading grads before backward retired, or the caller's stream
    observing params before a group finished), not floating point noise.
    """
    if os.environ.get("TEST_DEVICE", "cpu") != "cuda" or dist.get_world_size() != 4:
        pytest.skip("Requires exactly 4 ranks on CUDA (streams are a no-op on CPU)")
    T, G = 2, 2
    r = dist.get_rank()
    t_rank, g_rank = r % T, r // T
    tp_group, gtp_group = _get_2d_groups()
    egtp_group = _get_expert_group()
    ep_rank, egtp_rank = r // 2, r % 2
    lr, momentum = 1e-2, 0.95

    def _build():
        torch.manual_seed(_SEED + 60)
        dense_specs = [((32, 16), (0, 1)), ((32, 16), (1, 0))]
        dense_w = [torch.randn(*s) for s, _ in dense_specs]
        dense_g = [torch.randn(*s) for s, _ in dense_specs]

        def _dense_shard(full):
            rows = full.size(0) // (T * G)
            start = t_rank * (full.size(0) // T) + g_rank * rows
            return full[start : start + rows, :].clone()

        dense = []
        for w, g in zip(dense_w, dense_g):
            p = _gtp_param(_dense_shard(w))
            p.grad = _dense_shard(g)
            p.partition_dim = 0
            dense.append(p)

        torch.manual_seed(_SEED + 70 + ep_rank)
        expert_shapes = [(16, 8), (16, 8), (8, 16)]
        expert = []
        for s in expert_shapes:
            w, g = torch.randn(*s), torch.randn(*s)
            rows = s[0] // G
            p = _gtp_param(w[egtp_rank * rows : (egtp_rank + 1) * rows, :].clone())
            p.grad = g[egtp_rank * rows : (egtp_rank + 1) * rows, :].clone()
            expert.append(p)
        return dense, expert, dense_specs, [0, 1, 1]

    def _run(concurrent):
        dense, expert, dense_specs, expert_homes = _build()
        opt = LayerShardedMuon(
            [{"params": dense}, {"params": expert}],
            lr=lr,
            momentum=momentum,
            nesterov=True,
            weight_decay=0.0,
            coefficient_type="quintic",
            num_ns_steps=5,
            scale_mode="spectral",
            extra_scale_factor=1.0,
            fp32_matmul_prec="highest",
            gtp_group=gtp_group,
            tp_group=tp_group,
            ns_batch_size=1,
            concurrent_groups=concurrent,
        )
        opt.set_group_process_groups({0: (gtp_group, tp_group), 1: (egtp_group, None)})
        homes = {id(p): h for p, (_, h) in zip(dense, dense_specs)}
        homes.update({id(p): (h, 0) for p, h in zip(expert, expert_homes)})
        opt.set_param_ns_homes(homes)
        # Two steps: the second one starts from momentum the first wrote, so a
        # stream hazard on the momentum buffers would show up here.
        for _ in range(2):
            opt.step()
        torch.cuda.synchronize()
        return [p.data.clone() for p in dense + expert]

    serial = _run(False)
    concurrent = _run(True)
    for i, (a, b) in enumerate(zip(serial, concurrent)):
        assert torch.equal(a, b), (
            f"param {i} differs between serial and concurrent groups on rank {r}: "
            f"max |diff| = {(a - b).abs().max().item():.3e}"
        )


def test_degenerate_domain_group_falls_back_to_local_ns():
    """A group whose (GTP x TP) domain is a single rank runs plain local NS."""
    _require_four_ranks()
    r = dist.get_rank()
    tp_group, gtp_group = _get_2d_groups()
    lr, momentum = 1e-2, 0.95

    torch.manual_seed(_SEED + 80 + r)  # distinct per rank: purely local math
    w = torch.randn(12, 8)
    g = torch.randn(12, 8)
    p = torch.nn.Parameter(w.clone())
    p.grad = g.clone()

    # Another group with a real domain, so _param_ns_homes is non-empty and the
    # layer-sharded path is active for the optimizer as a whole.
    torch.manual_seed(_SEED + 90)
    other_full = torch.randn(32, 16)
    other_grad = torch.randn(32, 16)
    rows = 32 // 4
    start = (r % 2) * 16 + (r // 2) * rows
    other = _gtp_param(other_full[start : start + rows, :].clone())
    other.grad = other_grad[start : start + rows, :].clone()
    other.partition_dim = 0

    optimizer = LayerShardedMuon(
        [{"params": [other]}, {"params": [p]}],
        lr=lr,
        momentum=momentum,
        nesterov=True,
        weight_decay=0.0,
        coefficient_type="quintic",
        num_ns_steps=5,
        scale_mode="spectral",
        extra_scale_factor=1.0,
        fp32_matmul_prec="highest",
        gtp_group=gtp_group,
        tp_group=tp_group,
    )
    optimizer.set_group_process_groups({0: (gtp_group, tp_group), 1: (None, None)})
    optimizer.set_param_ns_homes({id(other): (0, 0)})
    optimizer.step()

    m = torch.zeros_like(g).lerp_(g, 1 - momentum)
    eff = g.lerp(m, momentum)
    orth = newton_schulz(eff.float(), steps=5, coefficient_type="quintic")
    expected = w - lr * (max(12, 8) ** 0.5) * orth
    torch.testing.assert_close(
        p.data,
        expected,
        atol=1e-4,
        rtol=1e-4,
        msg=lambda m: f"degenerate-domain local NS mismatch on rank {r}\n\n{m}",
    )


# ---------------------------------------------------------------------------
# Batched Newton-Schulz (ns_batch_size) matches the per-matrix path
#
# Models the MoE shape: many identically shaped expert weights on one NS home.
# Batching routes through baddbmm instead of addmm, so parity holds to
# kernel-level floating point rounding rather than bit-exactly.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "ns_batch_size,n_same,n_other,home",
    [
        # home="spread": homes round-robin, so each home batches its own few.
        # home="single": every param lands on home 0, so ns_batch_size actually
        # splits one home's list and the chunk-boundary logic is exercised.
        (32, 10, 0, "spread"),
        (32, 10, 0, "single"),
        (4, 10, 0, "single"),  # 4+4+2
        (3, 9, 0, "single"),  # 3+3+3 exact
        (4, 6, 5, "single"),  # mixed shapes
        (32, 6, 3, "spread"),
        (1, 6, 3, "single"),  # disabled
    ],
    ids=[
        "b32_10same_spread",
        "b32_10same_single",
        "b4_10same_single",
        "b3_9same_single",
        "b4_6same_5other_single",
        "b32_6same_3other_spread",
        "b1_6same_3other_single",
    ],
)
def test_batched_matches_unbatched(ns_batch_size, n_same, n_other, home):
    _require_multi_rank()
    S = dist.get_world_size()
    r = dist.get_rank()
    lr, momentum = 1e-2, 0.95
    P, Q = 8 * S, 12
    P2, Q2 = 4 * S, 16
    rows, rows2 = P // S, P2 // S

    torch.manual_seed(_SEED + 100)
    shapes = [(P, Q)] * n_same + [(P2, Q2)] * n_other
    full_w = [torch.randn(*s) for s in shapes]
    full_g = [torch.randn(*s) for s in shapes]

    def _shard(full):
        n = full.size(0) // S
        return full[r * n : (r + 1) * n, :].clone()

    def _build():
        ps = []
        for w, g in zip(full_w, full_g):
            p = _gtp_param(_shard(w))
            p.grad = _shard(g)
            ps.append(p)
        return ps

    homes = {i: ((i % S, 0) if home == "spread" else (0, 0)) for i in range(len(shapes))}

    def _run(batch_size):
        ps = _build()
        opt = LayerShardedMuon(
            ps,
            lr=lr,
            momentum=momentum,
            nesterov=True,
            weight_decay=0.0,
            coefficient_type="quintic",
            num_ns_steps=5,
            scale_mode="spectral",
            extra_scale_factor=1.0,
            fp32_matmul_prec="highest",
            gtp_group=_world(),
            ns_batch_size=batch_size,
        )
        opt.set_param_ns_homes({id(p): homes[i] for i, p in enumerate(ps)})
        opt.step()
        return [p.data.clone() for p in ps]

    got = _run(ns_batch_size)
    ref = _run(1)

    for i, (a, b) in enumerate(zip(got, ref)):
        torch.testing.assert_close(
            a,
            b,
            atol=1e-5,
            rtol=1e-5,
            msg=lambda m, i=i, a=a, b=b: (
                f"batched(ns_batch_size={ns_batch_size}) vs unbatched mismatch "
                f"for param {i} shape={shapes[i]} on rank {r}: "
                f"max_diff={(a - b).abs().max().item():.2e}\n\n{m}"
            ),
        )


def test_batched_expert_group_matches_full_matrix_reference():
    """Many same-shape expert weights on an EGTP(2) domain, batched, vs reference."""
    _require_four_ranks()
    G = 2
    r = dist.get_rank()
    egtp_group = _get_expert_group()
    ep_rank, egtp_rank = r // 2, r % 2
    lr, momentum = 1e-2, 0.95

    torch.manual_seed(_SEED + 110 + ep_rank)
    n_experts = 9
    shape = (16, 8)
    full_w = [torch.randn(*shape) for _ in range(n_experts)]
    full_g = [torch.randn(*shape) for _ in range(n_experts)]
    rows = shape[0] // G

    params = []
    for w, g in zip(full_w, full_g):
        p = _gtp_param(w[egtp_rank * rows : (egtp_rank + 1) * rows, :].clone())
        p.grad = g[egtp_rank * rows : (egtp_rank + 1) * rows, :].clone()
        params.append(p)

    optimizer = LayerShardedMuon(
        params,
        lr=lr,
        momentum=momentum,
        nesterov=True,
        weight_decay=0.0,
        coefficient_type="quintic",
        num_ns_steps=5,
        scale_mode="spectral",
        extra_scale_factor=1.0,
        fp32_matmul_prec="highest",
        gtp_group=egtp_group,
        ns_batch_size=8,  # 9 experts -> chunks of 8 + 1
    )
    optimizer.set_param_ns_homes({id(p): (i % G, 0) for i, p in enumerate(params)})
    optimizer.step()

    for i, (w, g) in enumerate(zip(full_w, full_g)):
        m = torch.zeros_like(g).lerp_(g, 1 - momentum)
        eff = g.lerp(m, momentum)
        orth = newton_schulz(eff.float(), steps=5, coefficient_type="quintic")
        w_new = w - lr * (max(*shape) ** 0.5) * orth
        torch.testing.assert_close(
            params[i].data,
            w_new[egtp_rank * rows : (egtp_rank + 1) * rows, :],
            atol=1e-4,
            rtol=1e-4,
            msg=lambda m, i=i: (
                f"batched expert param {i} mismatch on rank {r} "
                f"(ep={ep_rank}, egtp={egtp_rank})\n\n{m}"
            ),
        )


# ---------------------------------------------------------------------------
# Fused single all_to_all over the flattened (GTP x TP) domain
#
# The fused path moves the exact same shard blocks as the two-stage path and
# assembles them in the same order, so with ns_batch_size=1 the two paths must
# be BIT-IDENTICAL — any deviation is a routing bug, not rounding.
#
# Rank convention (matches _get_2d_groups): t = r % T, g = r // T, so the flat
# rank g*T + t equals the world rank and the fused group is WORLD itself.
# ---------------------------------------------------------------------------


def _build_mixed_params(T, G, t_rank, g_rank, seed_offset):
    # (full_shape, partition_dim, (g_home, t_home))
    specs = [
        ((32, 16), 0, (0, 1)),  # column-parallel
        ((16, 32), 1, (1, 0)),  # row-parallel
        ((8, 16), None, (1, 1)),  # not TP-sharded
        ((32, 16), 0, (1, 0)),  # second col-parallel, different home
        ((8, 16), None, (0, 0)),  # second non-TP, different home
    ]
    torch.manual_seed(_SEED + seed_offset)
    full_w = [torch.randn(*s) for s, _, _ in specs]
    full_g = [torch.randn(*s) for s, _, _ in specs]

    def _shard(full, pd):
        P, Q = full.shape
        if pd == 0:
            rows = P // (T * G)
            start = t_rank * (P // T) + g_rank * rows
            return full[start : start + rows, :].clone()
        if pd == 1:
            return full[
                g_rank * (P // G) : (g_rank + 1) * (P // G),
                t_rank * (Q // T) : (t_rank + 1) * (Q // T),
            ].clone()
        return full[g_rank * (P // G) : (g_rank + 1) * (P // G), :].clone()

    params = []
    for (shape, pd, _), w, g in zip(specs, full_w, full_g):
        p = _gtp_param(_shard(w, pd))
        p.grad = _shard(g, pd)
        if pd is not None:
            p.partition_dim = pd
        params.append(p)
    return specs, full_w, full_g, params, _shard


def test_fused_bitwise_matches_two_stage():
    _require_four_ranks("Requires exactly 4 ranks (TP=2 x GTP=2)")
    T, G = 2, 2
    r = dist.get_rank()
    t_rank, g_rank = r % T, r // T
    tp_group, gtp_group = _get_2d_groups()
    lr, momentum = 1e-2, 0.95

    results = {}
    for label, fused in (("two_stage", None), ("fused", _world())):
        specs, _, _, params, _ = _build_mixed_params(T, G, t_rank, g_rank, 200)
        opt = LayerShardedMuon(
            params,
            lr=lr,
            momentum=momentum,
            nesterov=True,
            weight_decay=0.0,
            coefficient_type="quintic",
            num_ns_steps=5,
            scale_mode="spectral",
            extra_scale_factor=1.0,
            fp32_matmul_prec="highest",
            gtp_group=gtp_group,
            tp_group=tp_group,
            fused_group=fused,
            ns_batch_size=1,  # per-matrix NS -> the paths must be bit-identical
        )
        opt.set_param_ns_homes({id(p): s[2] for p, s in zip(params, specs)})
        opt.step()
        results[label] = [p.data.clone() for p in params]

    for i, (a, b) in enumerate(zip(results["fused"], results["two_stage"])):
        assert torch.equal(a, b), (
            f"fused vs two-stage NOT bit-identical for param {i} on rank {r}: "
            f"max_diff={(a - b).abs().max().item():.3e}"
        )


def test_fused_matches_full_matrix_reference():
    _require_four_ranks("Requires exactly 4 ranks (TP=2 x GTP=2)")
    T, G = 2, 2
    r = dist.get_rank()
    t_rank, g_rank = r % T, r // T
    tp_group, gtp_group = _get_2d_groups()
    lr, momentum = 1e-2, 0.95

    specs, full_w, full_g, params, _shard = _build_mixed_params(T, G, t_rank, g_rank, 210)
    opt = LayerShardedMuon(
        params,
        lr=lr,
        momentum=momentum,
        nesterov=True,
        weight_decay=0.0,
        coefficient_type="quintic",
        num_ns_steps=5,
        scale_mode="spectral",
        extra_scale_factor=1.0,
        fp32_matmul_prec="highest",
        gtp_group=gtp_group,
        tp_group=tp_group,
        fused_group=_world(),
    )
    opt.set_param_ns_homes({id(p): s[2] for p, s in zip(params, specs)})
    opt.step()

    for i, ((shape, pd, _), w, g) in enumerate(zip(specs, full_w, full_g)):
        P, Q = shape
        m = torch.zeros_like(g).lerp_(g, 1 - momentum)
        eff = g.lerp(m, momentum)  # nesterov
        orth = newton_schulz(eff.float(), steps=5, coefficient_type="quintic")
        w_new = w - lr * (max(P, Q) ** 0.5) * orth
        torch.testing.assert_close(
            params[i].data,
            _shard(w_new, pd),
            atol=1e-4,
            rtol=1e-4,
            msg=lambda m, i=i, pd=pd: (
                f"fused parity mismatch for param {i} (pd={pd}) on rank {r}\n\n{m}"
            ),
        )


def test_fused_per_group_domains():
    """Fused dense domain + two-stage expert domain in one step()."""
    _require_four_ranks()
    T, G = 2, 2
    r = dist.get_rank()
    t_rank, g_rank = r % T, r // T
    tp_group, gtp_group = _get_2d_groups()
    egtp_group = _get_expert_group()
    ep_rank, egtp_rank = r // 2, r % 2
    lr, momentum = 1e-2, 0.95

    specs, full_w, full_g, dense_params, _shard = _build_mixed_params(T, G, t_rank, g_rank, 220)

    torch.manual_seed(_SEED + 230 + ep_rank)
    e_shape = (16, 8)
    e_w = [torch.randn(*e_shape) for _ in range(3)]
    e_g = [torch.randn(*e_shape) for _ in range(3)]
    rows = e_shape[0] // G
    expert_params = []
    for w, g in zip(e_w, e_g):
        p = _gtp_param(w[egtp_rank * rows : (egtp_rank + 1) * rows, :].clone())
        p.grad = g[egtp_rank * rows : (egtp_rank + 1) * rows, :].clone()
        expert_params.append(p)

    opt = LayerShardedMuon(
        [{"params": dense_params}, {"params": expert_params}],
        lr=lr,
        momentum=momentum,
        nesterov=True,
        weight_decay=0.0,
        coefficient_type="quintic",
        num_ns_steps=5,
        scale_mode="spectral",
        extra_scale_factor=1.0,
        fp32_matmul_prec="highest",
        gtp_group=gtp_group,
        tp_group=tp_group,
    )
    opt.set_group_process_groups({0: (gtp_group, tp_group, _world()), 1: (egtp_group, None)})
    homes = {id(p): s[2] for p, s in zip(dense_params, specs)}
    homes.update({id(p): (i % G, 0) for i, p in enumerate(expert_params)})
    opt.set_param_ns_homes(homes)
    opt.step()

    def _reference(w, g):
        m = torch.zeros_like(g).lerp_(g, 1 - momentum)
        eff = g.lerp(m, momentum)
        orth = newton_schulz(eff.float(), steps=5, coefficient_type="quintic")
        P, Q = w.shape
        return w - lr * (max(P, Q) ** 0.5) * orth

    for i, ((shape, pd, _), w, g) in enumerate(zip(specs, full_w, full_g)):
        torch.testing.assert_close(
            dense_params[i].data,
            _shard(_reference(w, g), pd),
            atol=1e-4,
            rtol=1e-4,
            msg=lambda m, i=i: f"fused dense param {i} mismatch on rank {r}\n\n{m}",
        )
    for i, (w, g) in enumerate(zip(e_w, e_g)):
        ref = _reference(w, g)
        torch.testing.assert_close(
            expert_params[i].data,
            ref[egtp_rank * rows : (egtp_rank + 1) * rows, :],
            atol=1e-4,
            rtol=1e-4,
            msg=lambda m, i=i: f"two-stage expert param {i} mismatch on rank {r}\n\n{m}",
        )


# ---------------------------------------------------------------------------
# SYRK kernel path: same math as the GEMM path, different kernel rounding
#
# SYRK computes only one triangle of the symmetric NS GEMMs (A = X X^T and
# B = bA + cA^2). It requires CUDA + Triton >= 3.4 and only takes effect under
# fp32_matmul_prec='medium', so this test is GPU-only. Parity is asserted
# relative to the UPDATE magnitude (not the weight), since bf16 kernel-level
# rounding is the expected difference scale.
# ---------------------------------------------------------------------------


def test_syrk_matches_gemm_relative_to_update():
    if not torch.cuda.is_available() or dist.get_world_size() < 2:
        pytest.skip("Requires CUDA and >= 2 ranks")
    S = dist.get_world_size()
    r = dist.get_rank()
    lr, momentum = 1e-2, 0.95
    P, Q = 64 * S, 384  # 8-aligned dims (SYRK/TMA requirement)
    N = S
    rows = P // S

    torch.manual_seed(_SEED + 300)
    full_w = [torch.randn(P, Q) for _ in range(N)]
    full_g = [torch.randn(P, Q) for _ in range(N)]

    def _shard(t):
        return t[r * rows : (r + 1) * rows, :].clone()

    def _run(use_syrk):
        ps = []
        for w, g in zip(full_w, full_g):
            p = _gtp_param(_shard(w))
            p.grad = _shard(g)
            ps.append(p)
        opt = LayerShardedMuon(
            ps,
            lr=lr,
            momentum=momentum,
            nesterov=True,
            weight_decay=0.0,
            coefficient_type="quintic",
            num_ns_steps=5,
            scale_mode="spectral",
            extra_scale_factor=1.0,
            fp32_matmul_prec="medium",  # SYRK only engages on the bf16 path
            gtp_group=_world(),
            ns_batch_size=1,
            use_syrk=use_syrk,
        )
        opt.set_param_ns_homes({id(p): (i % S, 0) for i, p in enumerate(ps)})
        opt.step()
        return [p.data.clone() for p in ps]

    w_ref = _run(False)
    w_syrk = _run(True)

    for i in range(N):
        w_init = _shard(full_w[i])
        update = (w_ref[i] - w_init).norm()
        diff = (w_syrk[i] - w_ref[i]).norm()
        rel = (diff / update).item()
        assert rel < 0.05, (
            f"SYRK vs GEMM divergence too large for param {i} on rank {r}: "
            f"|diff|/|update| = {rel:.3e}"
        )
