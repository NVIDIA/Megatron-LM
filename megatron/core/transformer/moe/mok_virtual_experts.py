# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""MoonEP physical placement and replica transport for MoK's training megakernel.

The router, optimizer, and checkpoint work with logical experts. Each forward owns
a plan, including checkpoint replay and repeated invocations of tied MTP experts.
Transfers and MoK use the caller's stream. Only replica payloads and planner
scratch are shared across layers, and the backward reloads its own plan's weights.
"""

from __future__ import annotations

from types import SimpleNamespace

import torch

from megatron.core.transformer.moe.virtual_expert_load_balancer import (
    MAX_VIRTUAL_EXPERT_WEIGHT_SMS,
    VirtualExpertPlan,
    VirtualExpertPlannerWorkspace,
    _VirtualExpertConfig,
    _VirtualExpertStorage,
    launch_virtual_expert_grad_reduce,
    launch_virtual_expert_planner,
    launch_virtual_expert_weight_prefetch,
)
from megatron.core.utils import nvtx_decorator


class _MoKReplicaPool(_VirtualExpertStorage):
    """Two physical weight orientations and a single FP32 gradient arena."""

    def __init__(self, group, layout):
        self.column = None
        self.slot_views = self.slot_grads = ()
        # MoK consumes raw slot tensors rather than TE Parameter wrappers.
        super().__init__(group, layout, templates=())
        try:
            if layout.mxfp8:
                import torch.distributed._symmetric_memory as symm_mem

                arena = symm_mem.empty(
                    self.weight_arena.numel(), dtype=torch.uint8, device=layout.device
                )
                self.column = SimpleNamespace(
                    weight_arena=arena,
                    weight_handle=symm_mem.rendezvous(arena, group),
                    weight_grid_barrier=self.weight_grid_barrier,
                    rank=self.rank,
                    world_size=self.world_size,
                    member_numels=self.member_numels,
                    num_local_experts=self.num_local_experts,
                    num_sms=self.num_sms,
                )
                if self.column.weight_handle.signal_pad_size < 4 * self.world_size:
                    raise RuntimeError("MoK virtual experts need one signal word per EP rank.")
                arena.zero_()
            self.slot_views = self._weight_views()
            self.slot_grads = tuple(
                section.view(layout.num_local_experts, *shape)
                for section, shape in zip(
                    self.grad_arena.split(self._grad_sections), layout.member_shapes
                )
            )
        except Exception:
            self.destroy()
            raise

    def _weight_views(self):
        count = self.num_local_experts
        rows = self.weight_arena.split(self._weight_sections)
        cols = self.column.weight_arena.split(self._weight_sections) if self.column else None
        result = []
        for i, (m, k) in enumerate(self.config.member_shapes):
            payload = rows[2 * i].view(count, m, k)
            if self.column is None:
                result.append(payload)
                continue
            # Preserve MoK's tcgen05 scale layout byte for byte in both directions.
            # Native TE's column payload has [M,K] storage and uses a transpose flag.
            result.append(
                (
                    payload.view(torch.float8_e4m3fn),
                    rows[2 * i + 1].view(count * (m // 128), k // 128, 32, 16),
                    cols[2 * i].view(torch.float8_e4m3fn).view(count, m, k),
                    cols[2 * i + 1].view(count * (k // 128), m // 128, 32, 16),
                    True,
                )
            )
        return tuple(result)

    def destroy(self):
        super().destroy()
        self.column = None
        self.slot_views = self.slot_grads = ()


class MoKVirtualExperts:
    """Forward-local MoonEP plans over a stream-ordered pool of physical slots."""

    _planners = {}
    _pools = {}

    def __init__(self, config, ep_group, num_local_experts):
        self.config = config
        self.group = ep_group
        self.num_local_experts = num_local_experts
        self.clear()

    def clear(self):
        """Drop per-module state when owner storage moves, retaining the shared arenas."""
        self.pool = None
        self._tables = {}

    @torch.no_grad()
    @nvtx_decorator(message="mok_virtual_expert_planner")
    def plan(self, top_experts):
        """Keep logical route order, including the MoK -1 empty-route sentinel."""
        if VirtualExpertPlannerWorkspace is None:
            raise RuntimeError(
                "MoK virtual experts require Triton >= 3.5 and CUDA symmetric memory."
            )
        key = (self.group, top_experts.device, self.config.num_moe_experts)
        workspace = self._planners.get(key)
        if workspace is None:
            workspace = VirtualExpertPlannerWorkspace(
                num_experts=self.config.num_moe_experts,
                device=top_experts.device,
                group=self.group,
                symm_backend="CUDA",
            )
            self._planners[key] = workspace
        return VirtualExpertPlan(
            *launch_virtual_expert_planner(
                top_experts.contiguous(), workspace, alignment=256, masked_routes=True
            )
        )

    def _open_pool(self, weights):
        fc1, fc2 = weights
        mxfp8 = isinstance(fc1[0], tuple)
        first = fc1[0][0] if mxfp8 else fc1[0]
        shape = tuple(tuple((fc[0][0] if mxfp8 else fc[0]).shape) for fc in (fc1, fc2))
        if any(len(s) != 2 or any(v % 128 for v in s) for s in shape):
            raise ValueError(
                f"MoK virtual experts require 128-aligned 2D expert weights, got {shape}"
            )
        sms = self.config.moe_flex_dispatcher_num_sms or 32
        layout = _VirtualExpertConfig(
            self.group.group_name,
            first.device,
            self.group.size(),
            self.num_local_experts,
            shape,
            mxfp8,
            torch.float32,
            min(sms, MAX_VIRTUAL_EXPERT_WEIGHT_SMS),
            symm_backend="CUDA",
        )
        if layout not in self._pools:
            self._pools[layout] = _MoKReplicaPool(self.group, layout)
        return self._pools[layout]

    def _ptr_tables(self, label, groups):
        """Build transport tables once per layer, refreshing if owner pointers change."""
        addresses = tuple(tuple(t.data_ptr() for t in tensors) for tensors in groups)
        cached = self._tables.get(label)
        if cached is None or cached[0] != addresses:
            tables = tuple(
                torch.tensor(row, dtype=torch.int64, device=self.pool.config.device)
                for row in addresses
            )
            self._tables[label] = (addresses, tables)
        return self._tables[label][1]

    @torch.no_grad()
    @nvtx_decorator(message="mok_virtual_expert_weight_push")
    def prepare_weights(self, plan, weights, *, backward=False):
        """Publish MoK-format rowwise weights and, for MXFP8 backward, columnwise weights."""
        if self.pool is None:
            self.pool = self._open_pool(weights)
        pool = self.pool
        if not pool.config.mxfp8:
            launch_virtual_expert_weight_prefetch(
                pool,
                sources=self._ptr_tables("weights", weights),
                experts_to_copy=plan.experts_to_copy,
                wait_for_consumers=True,
            )
            return pool.slot_views
        directions = [(0, pool)]
        if backward:
            directions.append((2, pool.column))
        for component, arena in directions:
            data = [[member[component] for member in fc] for fc in weights]
            scales = [[member[component + 1] for member in fc] for fc in weights]
            launch_virtual_expert_weight_prefetch(
                arena,
                sources=self._ptr_tables(component, data),
                scale_sources=self._ptr_tables(component + 1, scales),
                experts_to_copy=plan.experts_to_copy,
                wait_for_consumers=True,
            )
        return pool.slot_views

    def gradient_slots(self):
        """Return [local_ghosts,M,K] MoK wgrad buffers, initialized to zero."""
        return self.pool.slot_grads

    @torch.no_grad()
    @nvtx_decorator(message="mok_virtual_expert_grad_reduce")
    def reduce_gradients(self, plan, owner_grads):
        """Sum partials into the owner's FP32 accumulation before DDP hooks are eligible."""
        if any(g.dtype != torch.float32 or not g.is_contiguous() for fc in owner_grads for g in fc):
            raise ValueError("MoK virtual experts require contiguous FP32 owner main gradients.")
        launch_virtual_expert_grad_reduce(
            self.pool,
            native_grads=self._ptr_tables("grads", owner_grads),
            experts_to_copy=plan.experts_to_copy,
        )
        # The reduction's exit rendezvous orders every peer's reads before the clear.
        self.pool.grad_arena.zero_()

    @classmethod
    def finalize(cls):
        """Release shared windows while their owning EP process group is alive."""
        for pool in cls._pools.values():
            pool.destroy()
        for workspace in cls._planners.values():
            workspace.destroy()
        cls._pools.clear()
        cls._planners.clear()
