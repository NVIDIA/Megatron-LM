# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Model-agnostic MoE router replay for the MLite runtime."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch
import torch.distributed as dist

from megatron.lite.model import protocol_utils
from megatron.lite.primitive.modules.router_replay import (
    RouterReplay,
    RouterReplayAction,
    attach_router_replay,
    detach_router_replay,
)
from megatron.lite.primitive.parallel.thd import parallel_state_from_model


def _protocol_fn(protocol, name: str, fallback):
    fn = getattr(protocol, name, None) if protocol is not None else None
    return fn or fallback


class RouterReplayDriver:
    """Own the record/replay lifecycle for one ``forward_backward`` call."""

    def __init__(self, handle, action: str):
        if action not in ("record", "replay"):
            raise ValueError(
                f"router replay action must be 'record' or 'replay', got {action!r}."
            )
        self.handle = handle
        self.action = action
        self._chunks = handle._extras.get("model_chunks", [handle._model])
        self._protocol = handle._extras.get("protocol")
        self._num_routers = 0
        self._ps = None
        self._pp_offset = 0
        self._pp_total = 0

    def _replay_roots(self):
        selector = (
            getattr(self._protocol, "router_replay_roots", None)
            if self._protocol is not None
            else None
        )
        for chunk in self._chunks:
            roots = selector(chunk) if selector is not None else [chunk]
            yield from roots

    @classmethod
    def maybe_create(cls, handle, spec: Any) -> "RouterReplayDriver | None":
        if not spec:
            return None
        action = spec.get("action") if isinstance(spec, dict) else spec
        if action in (None, "disabled"):
            return None
        return cls(handle, action)

    def begin(self) -> None:
        RouterReplay.clear_global_router_replay_instances()
        self._num_routers = sum(
            attach_router_replay(root, reset=False) for root in self._replay_roots()
        )
        if self._num_routers == 0:
            raise RuntimeError("router replay requested but the model has no MoE routers.")
        self._ps = parallel_state_from_model(self._chunks[-1])
        self._compute_pp_layout()
        if self.action == "record":
            RouterReplay.set_global_router_replay_action(RouterReplayAction.RECORD)

    def _compute_pp_layout(self) -> None:
        ps = self._ps
        if ps is None or ps.pp_size <= 1:
            self._pp_total = self._num_routers
            return
        counts = [torch.zeros(1, dtype=torch.long, device="cuda") for _ in range(ps.pp_size)]
        dist.all_gather(
            counts,
            torch.tensor([self._num_routers], dtype=torch.long, device="cuda"),
            group=ps.pp_group,
        )
        values = [int(count.item()) for count in counts]
        self._pp_offset = sum(values[: ps.pp_rank])
        self._pp_total = sum(values)

    def wrap(self, forward_step: Callable) -> Callable:
        if self.action == "record":
            return self._wrap_record(forward_step)
        return self._wrap_replay(forward_step)

    def _wrap_record(self, forward_step: Callable) -> Callable:
        def stepped(model, batch):
            RouterReplay.set_global_router_replay_action(RouterReplayAction.RECORD)
            return forward_step(model, batch)

        return stepped

    def _wrap_replay(self, forward_step: Callable) -> Callable:
        def stepped(model, batch):
            routed = getattr(batch, "routed_experts", None)
            if routed is None:
                raise ValueError("R3 router replay requires batch.routed_experts.")
            routed = self._select_local_layers(routed)
            pack_routes = _protocol_fn(
                self._protocol, "pack_routed_experts", protocol_utils.pack_routed_experts
            )
            pack_mask = _protocol_fn(
                self._protocol, "pack_r3_replay_mask", protocol_utils.pack_r3_replay_mask
            )
            targets = pack_routes(model, batch, routed)
            replay_mask = pack_mask(model, batch)
            RouterReplay.set_replay_data(targets, replay_mask=replay_mask)
            RouterReplay.set_global_router_replay_action(RouterReplayAction.REPLAY_FORWARD)
            try:
                return forward_step(model, batch)
            finally:
                # Pipeline schedules may recompute checkpointed router forwards
                # after one or more newer micro-batches have run.  Those calls
                # must consume the saved per-microbatch FIFO, not the latest
                # forward target.
                RouterReplay.set_global_router_replay_action(
                    RouterReplayAction.REPLAY_BACKWARD
                )

        return stepped

    def _select_local_layers(self, routed):
        ps = self._ps
        if ps is None or ps.pp_size <= 1:
            expected = self._num_routers
            actual = (
                int(routed.values().size(1))
                if getattr(routed, "is_nested", False)
                else int(routed.size(-2))
            )
            if actual != expected:
                raise ValueError(
                    f"R3 route layer count mismatch: rollout={actual}, actor={expected}. "
                    "DS4 routing must include hash and learned MoE layers in model order."
                )
            return routed
        lo, hi = self._pp_offset, self._pp_offset + self._num_routers
        first_row = next(iter(routed.unbind(0)))
        if int(first_row.size(1)) != self._pp_total:
            raise ValueError(
                f"R3 route layer count mismatch: rollout={first_row.size(1)}, "
                f"actor={self._pp_total}. DS4 routing must include hash and learned "
                "MoE layers in model order."
            )
        rows = [row[:, lo:hi, :] for row in routed.unbind(0)]
        return torch.nested.as_nested_tensor(rows, layout=torch.jagged)

    def end(self) -> None:
        RouterReplay.clear_global_state()
        for root in self._replay_roots():
            detach_router_replay(root)
        RouterReplay.clear_global_router_replay_instances()


__all__ = ["RouterReplayDriver"]
