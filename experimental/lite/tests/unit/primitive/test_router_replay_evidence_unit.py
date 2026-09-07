# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""R3 replay must produce direct evidence that it substituted routing.

``batch.routed_experts is not None`` proves the rollout's routes arrived. It does
not prove any routing decision was overridden: a build whose replay hook silently
no-ops emits an identical log and identical metrics keys. The counters here are
the only direct evidence, and the VOID paths are what stop "no output" from being
readable as "it worked".
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn
from megatron.lite.primitive.modules.router_replay import (
    RouterReplay,
    RouterReplayAction,
)

pytestmark = pytest.mark.mlite


def _fresh(n: int = 2) -> list[RouterReplay]:
    RouterReplay.clear_global_router_replay_instances()
    RouterReplay.reset_replay_stats()
    return [RouterReplay() for _ in range(n)]


def test_counters_record_rows_and_substitutions():
    (r,) = _fresh(1)
    native = torch.tensor([[0, 1], [2, 3], [4, 5]])
    target = torch.tensor([[9, 1], [2, 8], [4, 5]])  # 2 of 6 entries differ
    RouterReplay.set_replay_data([target])
    RouterReplay.set_global_router_replay_action(RouterReplayAction.REPLAY_FORWARD)

    out = r.select_indices(native)
    assert torch.equal(out, target)

    s = RouterReplay.replay_stats()
    assert s["calls"] == 1
    assert s["rows"] == 6
    assert s["changed"] == 2


def test_counters_accumulate_across_layers():
    layers = _fresh(2)
    native = torch.tensor([[0, 1], [2, 3]])
    targets = [torch.tensor([[5, 1], [2, 3]]), torch.tensor([[0, 1], [7, 7]])]
    RouterReplay.set_replay_data(targets)
    RouterReplay.set_global_router_replay_action(RouterReplayAction.REPLAY_FORWARD)
    for layer, _t in zip(layers, targets, strict=True):
        layer.select_indices(native)
    s = RouterReplay.replay_stats()
    assert s["calls"] == 2
    assert s["rows"] == 8
    assert s["changed"] == 3  # 1 in the first layer, 2 in the second


def test_identical_routes_count_as_zero_changed_not_as_not_running():
    """The distinction the counters exist for: ran-but-changed-nothing != did-not-run."""
    (r,) = _fresh(1)
    native = torch.tensor([[0, 1], [2, 3]])
    RouterReplay.set_replay_data([native.clone()])
    RouterReplay.set_global_router_replay_action(RouterReplayAction.REPLAY_FORWARD)
    r.select_indices(native)
    s = RouterReplay.replay_stats()
    assert s["calls"] == 1 and s["rows"] == 4, "replay DID run"
    assert s["changed"] == 0, "and legitimately changed nothing"


def test_mask_partial_replay_is_counted():
    (r,) = _fresh(1)
    native = torch.tensor([[0, 1], [2, 3], [4, 5]])
    target = torch.tensor([[9, 9], [9, 9], [9, 9]])
    mask = torch.tensor([True, False, True])
    RouterReplay.set_replay_data([target], replay_mask=mask)
    RouterReplay.set_global_router_replay_action(RouterReplayAction.REPLAY_FORWARD)
    out = r.select_indices(native)
    assert torch.equal(out[1], native[1]), "masked-out row keeps native routing"
    s = RouterReplay.replay_stats()
    assert s["rows"] == 6
    assert s["changed"] == 4  # rows 0 and 2, two slots each


def test_record_mode_does_not_inflate_replay_counters():
    (r,) = _fresh(1)
    RouterReplay.set_global_router_replay_action(RouterReplayAction.RECORD)
    r.select_indices(torch.tensor([[0, 1]]))
    assert RouterReplay.replay_stats()["calls"] == 0, "recording is not replaying"


def test_reset_clears_counters():
    (r,) = _fresh(1)
    RouterReplay.set_replay_data([torch.tensor([[9, 9]])])
    RouterReplay.set_global_router_replay_action(RouterReplayAction.REPLAY_FORWARD)
    r.select_indices(torch.tensor([[0, 1]]))
    assert RouterReplay.replay_stats()["rows"] == 2
    RouterReplay.reset_replay_stats()
    assert RouterReplay.replay_stats() == {"calls": 0, "rows": 0, "changed": 0}


# ------------------------------------------------------- the VOID (liveness) paths
def _driver_with(stats_calls: int, stats_rows: int, num_routers: int = 4):
    from megatron.lite.runtime.backends.mlite.router_replay import RouterReplayDriver

    d = RouterReplayDriver.__new__(RouterReplayDriver)
    d._num_routers = num_routers
    d._emitted_evidence = False
    RouterReplay.clear_global_router_replay_instances()
    RouterReplay.reset_replay_stats()
    RouterReplay.replay_calls = stats_calls
    RouterReplay.replay_rows_total = stats_rows
    RouterReplay.replay_rows_changed = 0
    return d


def test_void_when_no_select_indices_call_happened():
    """A build where the hook is detached must not pass silently."""
    d = _driver_with(stats_calls=0, stats_rows=0)
    with pytest.raises(RuntimeError, match="R3_REPLAY_VOID"):
        d._emit_replay_evidence()


def test_void_when_zero_rows_seen():
    d = _driver_with(stats_calls=3, stats_rows=0)
    with pytest.raises(RuntimeError, match="R3_REPLAY_VOID"):
        d._emit_replay_evidence()


def test_evidence_line_is_emitted_once_and_is_greppable(capsys):
    d = _driver_with(stats_calls=48, stats_rows=98304)
    RouterReplay.replay_rows_changed = 1234
    d._emit_replay_evidence()
    d._emit_replay_evidence()  # second step must not re-emit
    out = capsys.readouterr().out
    assert out.count("R3_REPLAY_EVIDENCE") == 1
    assert "calls=48" in out and "rows=98304" in out and "changed=1234" in out
    assert "changed_frac=0.012553" in out
    assert "routers=4" in out


def test_zero_changed_emits_a_warning_rather_than_silence(capsys):
    d = _driver_with(stats_calls=48, stats_rows=98304)
    d._emit_replay_evidence()
    out = capsys.readouterr().out
    assert "R3_REPLAY_EVIDENCE" in out
    assert "R3_REPLAY_WARN changed=0" in out


def test_replay_liveness_check_does_not_mask_a_pre_router_forward_error():
    from types import SimpleNamespace

    from megatron.lite.runtime.backends.mlite.router_replay import RouterReplayDriver

    class Root(nn.Module):
        def __init__(self):
            super().__init__()
            self.router = nn.Module()
            self.router.router_replay = None

    root = Root()
    protocol = SimpleNamespace(
        pack_routed_experts=lambda _model, _batch, _routed: [
            torch.tensor([[0, 1]])
        ],
        pack_r3_replay_mask=lambda _model, _batch: None,
    )
    handle = SimpleNamespace(
        _model=root,
        _extras={"model_chunks": [root], "protocol": protocol},
    )
    driver = RouterReplayDriver(handle, "replay")
    driver.begin()
    stepped = driver.wrap(
        lambda _model, _batch: (_ for _ in ()).throw(ValueError("sentinel forward error"))
    )
    batch = SimpleNamespace(routed_experts=torch.tensor([[[0, 1]]]))

    try:
        with pytest.raises(ValueError, match="sentinel forward error"):
            stepped(root, batch)
    finally:
        driver.end()
