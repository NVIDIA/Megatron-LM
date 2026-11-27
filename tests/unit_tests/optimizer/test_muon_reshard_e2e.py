# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""End-to-end: ``--muon-distributed-reshard-wide`` under ``--muon-tp-mode auto``, through the
real training-loop path (``setup_model_and_optimizer``), modeled on
``test_layer_sharded_e2e_parity.py``.

Production-like: ``--muon-tp-mode auto`` on a TP1 x GTP4 grid (GTP the only sharding axis of
the dense weights) with split-QKV on (projection level, so the qkv weight stays the duplicated
split under auto); the routed experts sit on an EP2 x EGTP2 grid and run duplicated (auto's cost
model covers dense weights only). Every layer is MoE; its shared expert supplies the dense MLP
weights, among them the wide shared fc2 [256, 512].

- Hardware-priced auto (GB200 profile): at these toy shapes the resharded chain's collectives
  cost more than one all-gather, so no weight flips and the flag leaves the losses bitwise
  unchanged.
- FLOP-priced auto (no hardware profile): the wide shared fc2 flips to the resharded chain. In
  the first optimizer step every other update -- each dense matrix that does not flip and each
  routed-expert update -- is bitwise the flag-off update, and the flipped fc2 update matches
  the flag-off one to fp32 rounding (fp32 Newton-Schulz).

Skips at module level when GTP is unavailable (TransformerEngine < 2.19.0.dev0).

Launch with 4 ranks:
  torchrun --nproc-per-node=4 -m pytest tests/unit_tests/optimizer/test_muon_reshard_e2e.py
"""

import pytest
import torch

pytest.importorskip("emerging_optimizers", reason="requires emerging-optimizers")

from megatron.core.tensor_parallel.gtp_api import HAVE_GTP

if not HAVE_GTP:
    pytest.skip("GTP requires TransformerEngine >= 2.19", allow_module_level=True)

from megatron.core.optimizer import emerging_optimizers as eo_mod
from tests.unit_tests.optimizer.test_layer_sharded_e2e_parity import (
    _COMMON,
    _MOE,
    _RECIPE,
    _LayerShardedHarness,
)
from tests.unit_tests.test_utilities import Utils

# TP1 x GTP4 dense, the routed experts EP2 x EGTP2 (their GTP group is not the dense one);
# 4 independent sequences.
_AUTO = {
    **_COMMON,
    **_MOE,
    'global_batch_size': 4,
    'tensor_parallel_num_weight_shards': 4,
    'moe_shared_expert_intermediate_size': 512,
    'muon_split_qkv': True,
    'expert_tensor_parallel_size': 1,
    'expert_model_parallel_size': 2,
    'expert_tensor_parallel_num_weight_shards': 2,
    'muon_tp_mode': 'auto',
}


class _RecordingHarness(_LayerShardedHarness):
    """Also keeps the TensorParallelMuon instances, to read their auto decisions."""

    def _on_model_built(self, model_chunks, optimizer, args):
        super()._on_model_built(model_chunks, optimizer, args)
        found = []

        def visit(opt):
            if isinstance(opt, eo_mod.TensorParallelMuon):
                found.append(opt)
                return
            children = getattr(opt, 'chained_optimizers', None)
            if children:
                for child in children:
                    visit(child)
                return
            inner = vars(opt).get('optimizer')
            if inner is not None and inner is not opt:
                visit(inner)

        visit(optimizer)
        self.muon = found


def _record_updates(monkeypatch, records):
    """Record, in call order, every orthogonalized update: the weight family (routed expert or
    dense), the shard shape, whether it took the resharded chain, the update, and the
    Newton-Schulz input shard."""
    orthogonalize = eo_mod.TensorParallelMuon.scaled_orthogonalize_fn_with_gtp_remat
    resharded = eo_mod.TensorParallelMuon._orthogonalize_resharded
    state = {"resharded": 0}

    def record_orthogonalize(self, p, grad, *args, **kwargs):
        before = state["resharded"]
        ns_input = grad.detach().clone()
        out = orthogonalize(self, p, grad, *args, **kwargs)
        flipped = state["resharded"] > before
        family = "expert" if getattr(p, 'expert_tp', False) else "dense"
        records.append((family, tuple(p.shape), flipped, out.clone(), ns_input))
        return out

    def record_resharded(self, *args, **kwargs):
        state["resharded"] += 1
        return resharded(self, *args, **kwargs)

    monkeypatch.setattr(
        eo_mod.TensorParallelMuon, "scaled_orthogonalize_fn_with_gtp_remat", record_orthogonalize
    )
    monkeypatch.setattr(eo_mod.TensorParallelMuon, "_orthogonalize_resharded", record_resharded)


def _run(harness, monkeypatch, profile, reshard, num_steps, **extra):
    """One training run; returns (losses, records of every update in call order, auto's
    decisions)."""
    records = []
    _record_updates(monkeypatch, records)
    monkeypatch.setattr(eo_mod, "_hardware_profile", lambda: profile)
    kwargs = {**_AUTO, **extra, 'num_steps': num_steps}
    if reshard:
        kwargs['muon_distributed_reshard_wide'] = True
    losses = harness._run_test_helper(tp_size=1, recipe=_RECIPE, fp8_param_gather=False, **kwargs)
    monkeypatch.undo()
    assert harness.muon and harness.layer_sharded_instances == 0
    assert all(type(opt) is eo_mod.TensorParallelMuon for opt in harness.muon), harness.muon
    assert all(opt.distributed_reshard_wide is reshard for opt in harness.muon)
    decisions = {}
    for opt in harness.muon:
        decisions.update(opt._tp_mode_cache)
    return losses, records, decisions


def _all_ranks_max(*values):
    t = torch.tensor(values, dtype=torch.float64, device="cuda")
    torch.distributed.all_reduce(t, op=torch.distributed.ReduceOp.MAX)
    return t.tolist()


def _gather(local):
    """Every rank's ``local`` tensor (the losses are compared on every rank)."""
    gathered = [torch.empty_like(local) for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(gathered, local.contiguous())
    return gathered


@pytest.fixture
def harness():
    if Utils.world_size != 4:
        pytest.skip("Requires exactly 4 torchrun ranks for TP1 x GTP4")
    h = _RecordingHarness()
    h.setup_method(None)
    h.seq_length = 64
    h.micro_batch_size = 1
    yield h
    h.teardown_method(None)
    from megatron.core.tensor_parallel.generalized_tensor_parallelism import update_gtp_config

    update_gtp_config(pad_for_alignment=16, calculate_per_token_loss=False)


@pytest.mark.launch_on_gb200
def test_reshard_flag_without_flips_is_bitwise_inert(harness, monkeypatch):
    """GB200-priced auto: no weight of the toy model flips, so 3 training steps with the flag
    give bitwise the losses and auto decisions of the flag-off run."""
    gb200 = eo_mod._PROFILES["GB200"]
    loss_off, rec_off, auto_off = _run(harness, monkeypatch, gb200, False, 3)
    loss_on, rec_on, auto_on = _run(harness, monkeypatch, gb200, True, 3)
    for rank, pair in enumerate(_gather(torch.stack((loss_off, loss_on)).cuda())):
        assert torch.isfinite(pair).all(), f"rank {rank}: {pair.tolist()}"
        assert torch.equal(pair[0], pair[1]), f"rank {rank}: {pair.tolist()}"
    assert len(rec_off) == len(rec_on) and not any(r[2] for r in rec_on)
    mismatches = sum(not torch.equal(a[3], b[3]) for a, b in zip(rec_off, rec_on))
    assert _all_ranks_max(mismatches) == [0.0], "an update differs from the flag-off run"
    # Same decisions; with the flag the wide shared fc2 is priced (and keyed) with its
    # reshard rows.
    assert {k[:3]: v for k, v in auto_on.items()} == auto_off, (auto_off, auto_on)
    assert (256, 512, 4, 256) in auto_on, auto_on


@pytest.mark.launch_on_gb200
def test_reshard_flip_changes_only_the_flipped_updates(harness, monkeypatch):
    """FLOP-priced auto: the wide shared fc2 flips to the resharded chain. In the first optimizer
    step (identical inputs) the routed experts' updates and every non-flipped dense update are
    bitwise the flag-off ones, and the flipped fc2 update, the same orthogonalization in another
    layout, matches the flag-off update to fp32 rounding (fp32 Newton-Schulz).

    The losses after step 1 are only required to be finite. They are no measure of the chain: a
    rounding-size difference in one update already moves the later losses."""
    steps = 3
    precision = {'muon_fp32_matmul_prec': "highest"}
    loss_off, rec_off, auto_off = _run(harness, monkeypatch, None, False, steps, **precision)
    loss_on, rec_on, auto_on = _run(harness, monkeypatch, None, True, steps, **precision)
    assert len(rec_off) == len(rec_on) and len(rec_on) % steps == 0
    first = rec_on[: len(rec_on) // steps]
    flipped = [r for r in first if r[2]]
    assert not any(r[2] for r in rec_off)
    # Both layers' shared fc2 [256, 512] (a [64, 512] GTP4 shard) flip, nothing else does.
    assert [r[1] for r in flipped] == [(64, 512)] * 2, [r[1] for r in first]
    assert {k: v for k, v in auto_on.items() if v != auto_off.get(k[:3])} == {
        (256, 512, 4, 256): "distributed"
    }
    mismatches, worst_flipped, input_differs = 0, 0.0, False
    for on, off in zip(first, rec_off[: len(first)]):
        assert on[:2] == off[:2], (on[:2], off[:2])
        if on[2]:
            worst_flipped = max(worst_flipped, ((on[3] - off[3]).norm() / off[3].norm()).item())
            input_differs = input_differs or not torch.equal(on[4], off[4])
        elif not torch.equal(on[3], off[3]):
            mismatches += 1
    n_expert = sum(1 for r in first if r[0] == "expert")
    mismatches, worst_flipped, n_expert_max, input_differs = _all_ranks_max(
        mismatches, worst_flipped, n_expert, float(input_differs)
    )
    assert n_expert_max > 0, "no expert update was recorded"
    assert mismatches == 0, f"{int(mismatches)} non-flipped updates differ from the flag-off run"
    assert input_differs == 0, "the flipped weights' step-1 Newton-Schulz input differs"
    # fp32 Newton-Schulz in two layouts: summation-order differences only.
    assert worst_flipped < 1e-3, worst_flipped
    for rank, pair in enumerate(_gather(torch.stack((loss_off, loss_on)).cuda())):
        msg = f"rank {rank}: {pair.tolist()}"
        assert torch.isfinite(pair).all(), msg
        # Step 1's loss precedes every update. Later losses are not compared (see the docstring).
        assert torch.equal(pair[0][0], pair[1][0]), msg
