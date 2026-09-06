# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.


"""Virtual-expert load balancing, virtual-expert planning, and runtime weight movement.
The load-balancing algorithm implemented here is taken from MoonEP
(https://github.com/moonshotAI/moonep): perfectly balanced expert parallelism through
redundant experts planned online from the router output.
"""

import functools
import gc
import math
import weakref
from collections.abc import Callable
from dataclasses import dataclass
from types import SimpleNamespace
from typing import TYPE_CHECKING

import torch
import torch.distributed as dist

from megatron.core.fp8_utils import is_mxfp8tensor

try:
    from megatron.core.transformer.moe.virtual_expert_triton import (
        MAX_VIRTUAL_EXPERT_WEIGHT_SMS,
        PLANNER_PROGRAMS,
        launch_virtual_expert_grad_reduce,
        launch_virtual_expert_planner,
        launch_virtual_expert_weight_prefetch,
    )

    _TRITON_AVAILABLE = True
except ImportError:
    MAX_VIRTUAL_EXPERT_WEIGHT_SMS = 32
    PLANNER_PROGRAMS = 128
    launch_virtual_expert_grad_reduce = None
    launch_virtual_expert_planner = None
    launch_virtual_expert_weight_prefetch = None
    _TRITON_AVAILABLE = False
from megatron.core.utils import nvtx_decorator

if TYPE_CHECKING:
    from megatron.core.transformer.transformer_config import TransformerConfig

# Push directions. MXFP8 forward GEMMs read the rowwise components, backward the columnwise.
FORWARD, BACKWARD = 0, 1
# Index of a projection's native wgrad pointer table, after the two push directions.
_GRAD = 2
_MXFP8_COMPONENTS = (
    "_rowwise_data",
    "_rowwise_scale_inv",
    "_columnwise_data",
    "_columnwise_scale_inv",
)


# --------------------------------------------------------------------------------------
# Planning
# --------------------------------------------------------------------------------------


@dataclass(slots=True)
class VirtualExpertPlan:
    """``virtual_experts``: int16 ``[num_tokens, router_topk]`` runtime ids (HybridEP's dense
    top-k routing); ``experts_to_copy``: int32 ``[ep_size, num_local_experts]`` semantic ids per
    virtual-expert slot, ``-1`` if unused.

    Plain data only. The backward hooks capture the plan inside autograd contexts, so a plan
    holding a differentiable tensor (the runtime probabilities) would close a reference cycle
    through the graph that Python's collector cannot see, leaking every layer's graph.
    """

    virtual_experts: torch.Tensor
    experts_to_copy: torch.Tensor


def _scratch_layout(num_experts: int, ep_size: int) -> tuple[dict, int]:
    """Fields of the planner's int32 scratch arena as ``name -> (offset, shape)`` plus its total
    size; mirrors ``_planner_fields`` in the Triton module."""
    block_ep = 1 << (ep_size - 1).bit_length()
    # Each flag word gets its own 128-byte line (the kernel's _FLAG_STRIDE).
    fields = (
        ("placement_grid_sync", (1,)),
        ("_pad0", (31,)),
        ("grid_sync", (1,)),
        ("_pad1", (31,)),
        ("sequence", (1,)),  # exchange launch counter the flags carry
        ("_pad2", (31,)),
        ("balance", (ep_size,)),  # native load minus rank capacity
        ("allocation", (num_experts, ep_size)),  # routes of each expert per destination
        ("destination_boundaries", (num_experts, block_ep)),  # segment ends, local ordinals
        ("virtual_expert_slots", (num_experts, ep_size)),  # slot holding an expert on a rank
        ("program_histogram", (PLANNER_PROGRAMS, num_experts)),
        ("running_counts", (PLANNER_PROGRAMS, num_experts)),
        ("tokens_per_expert", (num_experts,)),  # this rank's histogram
    )
    layout, offset = {}, 0
    for name, shape in fields:
        layout[name] = (offset, shape)
        offset += math.prod(shape)
    return layout, offset


@dataclass(slots=True)
class VirtualExpertPlannerWorkspace:
    """Planner scratch for one expert layout and EP group; every plan overwrites it.

    ``gathered_counts`` is this rank's NCCL symmetric-memory window: the planner kernel
    publishes the local histogram into every peer's window and reads the peers' rows from its
    own, so planning needs no collective. ``scratch`` is one int32 arena (see
    :func:`_scratch_layout`); :meth:`field` views its parts.
    """

    num_experts: int
    ep_size: int
    rank: int
    gathered_counts: torch.Tensor  # [ep_size, num_experts] window
    histogram_handle: object  # symmetric-memory handle of gathered_counts
    scratch: torch.Tensor

    @property
    def num_local_experts(self) -> int:
        return self.num_experts // self.ep_size

    def field(self, name: str) -> torch.Tensor:
        """View one field of the scratch arena."""
        offset, shape = _scratch_layout(self.num_experts, self.ep_size)[0][name]
        return self.scratch[offset : offset + math.prod(shape)].view(shape)

    def destroy(self) -> None:
        """Drop the symmetric window while its process group is still alive."""
        if self.histogram_handle is not None:
            torch.cuda.synchronize(self.gathered_counts.device)
        self.histogram_handle = self.gathered_counts = None

    @classmethod
    def allocate(cls, *, num_experts: int, device: torch.device, group: dist.ProcessGroup):
        """Allocate the scratch for one expert layout on ``group``, with the symmetric window."""
        import torch.distributed._symmetric_memory as symm_mem

        ep_size = dist.get_world_size(group=group)
        # The window needs the group's communicator (created by a first collective).
        dist.all_reduce(torch.zeros(1, device=device), group=group)
        if symm_mem.get_backend(device) != "NCCL":
            symm_mem.set_backend("NCCL")
        window = symm_mem.empty(ep_size * num_experts, dtype=torch.int32, device=device)
        handle = symm_mem.rendezvous(window, group)
        if handle.signal_pad_size < ep_size * 4 * 4:
            raise RuntimeError(
                "Virtual-expert planner needs one signal word per EP rank; the symmetric "
                f"memory signal pad holds {handle.signal_pad_size} bytes for {ep_size} ranks."
            )
        return cls.local(
            num_experts,
            ep_size,
            device,
            rank=dist.get_rank(group=group),
            gathered_counts=window.view(ep_size, num_experts),
            histogram_handle=handle,
        )

    @classmethod
    def local(
        cls, num_experts, ep_size, device, *, rank=0, gathered_counts=None, histogram_handle=None
    ):
        """The scratch without a window (process-local tests): ``gathered_counts`` is plain
        memory the caller fills with every rank's histogram."""
        if gathered_counts is None:
            gathered_counts = torch.empty((ep_size, num_experts), dtype=torch.int32, device=device)
        _, size = _scratch_layout(num_experts, ep_size)
        return cls(
            num_experts=num_experts,
            ep_size=ep_size,
            rank=rank,
            gathered_counts=gathered_counts,
            histogram_handle=histogram_handle,
            scratch=torch.zeros(size, dtype=torch.int32, device=device),
        )


_planner_workspaces: dict = {}


def get_planner_workspace(*, num_experts: int, device: torch.device, group: dist.ProcessGroup):
    """Return the process-wide planner scratch for one expert layout and process group;
    planning is stream-ordered, so every layer of a device shares it."""
    key = (num_experts, device.index, group.group_name)
    workspace = _planner_workspaces.get(key)
    if workspace is None:
        workspace = _planner_workspaces[key] = VirtualExpertPlannerWorkspace.allocate(
            num_experts=num_experts, device=device, group=group
        )
    return workspace


class _PlanRoutes(torch.autograd.Function):
    """One planner launch. The dense runtime probabilities it writes carry the gradient back to
    the router's ``[num_tokens, topk]`` probabilities through a gather at the runtime ids."""

    @staticmethod
    def forward(ctx, probs, top_indices, workspace, exchange):
        virtual_experts, runtime_probs, experts_to_copy = launch_virtual_expert_planner(
            top_indices, probs, workspace, exchange=exchange
        )
        ctx.save_for_backward(virtual_experts)
        ctx.probs_dtype = probs.dtype
        ctx.mark_non_differentiable(virtual_experts, experts_to_copy)
        # Autograd would otherwise zero-fill gradients for the two non-differentiable outputs.
        ctx.set_materialize_grads(False)
        return virtual_experts, runtime_probs, experts_to_copy

    @staticmethod
    def backward(ctx, grad_virtual_experts, grad_runtime_probs, grad_experts_to_copy):
        if grad_runtime_probs is None:
            return None, None, None, None
        (virtual_experts,) = ctx.saved_tensors
        grad_probs = grad_runtime_probs.gather(1, virtual_experts.long()).to(ctx.probs_dtype)
        return grad_probs, None, None, None


def plan_virtual_expert_routes(
    top_indices: torch.Tensor,
    probs: torch.Tensor,
    workspace: VirtualExpertPlannerWorkspace,
    *,
    exchange: bool = True,
) -> tuple[VirtualExpertPlan, torch.Tensor]:
    """Plan deterministic virtual-expert placement for one EP group and map this rank's routes.

    ``top_indices`` / ``probs`` are the router's ``[num_tokens, topk]`` expert ids and
    probabilities; every rank must route the same number of tokens. The histograms are the only
    cross-rank input and the planner kernel exchanges them itself, so every rank computes the
    same placement. ``exchange=False`` (process-local tests) plans from a pre-filled window.
    Returns the plan and the dense float32 ``[num_tokens, 2 * num_experts]`` runtime
    probabilities HybridEP consumes, which carry the gradient back to ``probs``.
    """
    if top_indices.shape != probs.shape or top_indices.dtype not in (torch.int32, torch.int64):
        raise ValueError("Virtual-expert planner takes matching [num_tokens, topk] ids and probs.")
    top_indices, probs = top_indices.contiguous(), probs.contiguous()
    virtual_experts, runtime_probs, experts_to_copy = _PlanRoutes.apply(
        probs, top_indices, workspace, exchange
    )
    return VirtualExpertPlan(virtual_experts, experts_to_copy), runtime_probs


# --------------------------------------------------------------------------------------
# Weight bridge
# --------------------------------------------------------------------------------------


def _wrap_mxfp8(template, shape, views, device) -> tuple[torch.Tensor, ...]:
    """Wrap raw ``(rowwise, rowwise_scale, columnwise, columnwise_scale)`` views as
    MXFP8 tensors carrying ``template``'s quantization metadata."""
    from transformer_engine.pytorch.tensor.mxfp8_tensor import MXFP8Tensor

    return tuple(
        MXFP8Tensor(
            shape=shape,
            dtype=template.dtype,
            rowwise_data=rowwise,
            rowwise_scale_inv=rowwise_scale,
            columnwise_data=columnwise,
            columnwise_scale_inv=columnwise_scale,
            fp8_dtype=template._fp8_dtype,
            quantizer=template._quantizer,
            with_gemm_swizzled_scales=template._with_gemm_swizzled_scales,
            requires_grad=False,
            device=device,
        )
        for rowwise, rowwise_scale, columnwise, columnwise_scale in views
    )


class _VirtualExpertWeightWorkspace:
    """Symmetric arenas and streams shared by every virtual-expert MoE layer of one EP group.

    The weight arena holds ``fc1 data, fc1 scales, fc2 data, fc2 scales`` with ``L``
    members per section (the scale sections are empty for BF16; MXFP8 keeps one because
    only one GEMM orientation is live at a time). The gradient arena holds ``fc1, fc2``.
    One layer's virtual experts are live at a time: the planner's histogram all-gather orders
    every push after the expert GEMMs that read the previous contents, and the
    reduction's exit rendezvous orders every slot rewrite after the owners' reads.
    """

    def __init__(self, group, device, config: tuple) -> None:
        import torch.distributed._symmetric_memory as symm_mem

        self.config = config
        world_size, num_local_experts, member_shapes, mxfp8, grad_dtype, num_sms = config
        self.num_local_experts = num_local_experts
        self.member_shapes = member_shapes
        self.member_numels = tuple(math.prod(shape) for shape in member_shapes)
        self.mxfp8 = mxfp8
        self.grad_dtype = grad_dtype
        self.num_sms = num_sms
        # One E8M0 scale byte per 32 MXFP8 weight bytes, unpadded (the config requires
        # 128-aligned projections so TE's padded scale layout has this exact size).
        self.scale_numels = tuple(numel // 32 if mxfp8 else 0 for numel in self.member_numels)
        arena_numel = num_local_experts * sum(self.member_numels)
        try:
            # NCCL window registration needs the group's device communicator, which NCCL creates
            # on the first collective (a barrier does not guarantee it); run one here, before
            # training or graph capture. The backend choice is process-global.
            dist.all_reduce(torch.zeros(1, device=device), group=group)
            if symm_mem.get_backend(device) != "NCCL":
                symm_mem.set_backend("NCCL")
            self.weight_arena = symm_mem.empty(
                arena_numel + num_local_experts * sum(self.scale_numels),
                dtype=torch.uint8 if mxfp8 else torch.bfloat16,
                device=device,
            )
            self.weight_handle = symm_mem.rendezvous(self.weight_arena, group)
            self.grad_arena = symm_mem.empty(arena_numel, dtype=grad_dtype, device=device)
            self.grad_handle = symm_mem.rendezvous(self.grad_arena, group)
        except RuntimeError as exc:
            raise RuntimeError(
                "Virtual-expert weights could not allocate NCCL symmetric memory for the EP group; "
                "the EP group must lie within one NVLink domain."
            ) from exc
        # The reduction reaches every peer's gradient slots through one TMA descriptor whose
        # outermost stride is the distance between consecutive peers' windows, so the
        # allocator's uniform mapping is a hard requirement: verify it once, here.
        bases = self.grad_handle.buffer_ptrs
        stride = bases[1] - bases[0] if len(bases) > 1 else 0
        if any(base != bases[0] + peer * stride for peer, base in enumerate(bases)):
            raise RuntimeError(
                "NCCL symmetric memory did not map the peers' gradient windows at a uniform "
                f"stride: {bases}."
            )
        self.weight_arena.zero_()
        self.grad_arena.zero_()
        self.weight_grid_barrier = torch.zeros(1, dtype=torch.int32, device=device)
        self.grad_grid_barrier = torch.zeros(1, dtype=torch.int32, device=device)
        # Two candidate weight streams: a CUDA-graph capture stream comes from the same
        # pool and may alias one of them.
        self.weight_streams = (torch.cuda.Stream(device=device), torch.cuda.Stream(device=device))
        self.grad_stream = torch.cuda.Stream(device=device)
        # Full native wgrad staging per projection, allocated on first use: TE's GEMM overwrites
        # it, the reduction adds the virtual-expert partials, autograd hands it to the optimizer.
        # GTP projections write per-layer GTP scratch instead and never allocate it.
        self._native_grads: dict[int, torch.Tensor] = {}

    def weight_stream(self, current_stream: torch.cuda.Stream) -> torch.cuda.Stream:
        """Return a weight stream distinct from ``current_stream``."""
        return next(s for s in self.weight_streams if s.cuda_stream != current_stream.cuda_stream)

    def slot_views(self, projection: int) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Return the ``[L, *shape]`` weight slots of one projection and, for MXFP8, their
        ``[L, numel // 32]`` scale bytes."""
        count, shape = self.num_local_experts, self.member_shapes[projection]
        numel, scale_numel = self.member_numels[projection], self.scale_numels[projection]
        offset = count * sum(self.member_numels[:projection]) + count * sum(
            self.scale_numels[:projection]
        )
        data = self.weight_arena.narrow(0, offset, count * numel).view(count, *shape)
        if not self.mxfp8:
            return data, None
        scales = self.weight_arena.narrow(0, offset + count * numel, count * scale_numel)
        return data, scales.view(count, scale_numel)

    def native_staging(self, projection: int) -> torch.Tensor:
        """Return the ``[L, *shape]`` native wgrad staging of one projection."""
        staging = self._native_grads.get(projection)
        if staging is None:
            staging = self._native_grads[projection] = torch.empty(
                (self.num_local_experts, *self.member_shapes[projection]),
                dtype=self.grad_dtype,
                device=self.grad_arena.device,
            )
        return staging

    def grad_slots(self, projection: int) -> torch.Tensor:
        """Return the ``[L, *shape]`` virtual-expert gradient slots of one projection."""
        count, numel = self.num_local_experts, self.member_numels[projection]
        offset = count * sum(self.member_numels[:projection])
        return self.grad_arena.narrow(0, offset, count * numel).view(
            count, *self.member_shapes[projection]
        )

    def destroy(self) -> None:
        """Drop the NCCL window registrations while the process group is still alive."""
        if self.weight_arena is not None:
            torch.cuda.synchronize(self.weight_arena.device)
        self.weight_handle = self.grad_handle = self.weight_arena = self.grad_arena = None


_workspaces: dict = {}
_bridges = weakref.WeakSet()


def _get_workspace(group, device, config: tuple) -> _VirtualExpertWeightWorkspace:
    # Group names are unique for the life of the process, so re-created groups never alias.
    key = (group.group_name, device.index)
    workspace = _workspaces.get(key)
    if workspace is None:
        workspace = _workspaces[key] = _VirtualExpertWeightWorkspace(group, device, config)
    elif workspace.config != config:
        raise ValueError(
            "All virtual-expert MoE layers on an EP group must share one weight shape and launch "
            f"configuration; expected {workspace.config}, got {config}."
        )
    return workspace


def finalize_virtual_expert_weight_bridges() -> None:
    """Release every virtual-expert arena; idempotent.

    A normal exit needs no call. Callers that destroy their process groups while a model is
    still alive (tests, an orderly shutdown) call it first so the NCCL windows are deregistered
    while their communicator exists.
    """
    try:
        for bridge in list(_bridges):
            bridge.release()
        for workspace in _workspaces.values():
            workspace.destroy()
        for workspace in _planner_workspaces.values():
            workspace.destroy()
    except Exception:  # a teardown release must never raise
        pass
    _workspaces.clear()
    _planner_workspaces.clear()
    # The runtime parameters and their TE ops sit in reference cycles; free the arenas now.
    gc.collect()


def _drop_grad(parameter: torch.nn.Parameter) -> None:
    """TE returns a dummy leaf grad once the fused wgrad is in ``main_grad``; drop it."""
    parameter.grad = None


class _VirtualExpertProjection:
    """One projection's optimizer parameters, runtime parameters and pointer tables.

    The ``2L`` runtime parameters are the natives followed by the virtual-expert slots. Their
    ``main_grad`` is the native staging (GTP: the layer's GTP wgrad scratch, bound per backward
    by :meth:`bind_wgrad_scratch`) or the slot's gradient arena member and carries
    ``overwrite_main_grad``, so TE's wgrad GEMM rewrites every member on each backward
    and the slots never need clearing (a planned slot always receives tokens).
    """

    def __init__(self, name, parameters, workspace: _VirtualExpertWeightWorkspace, index: int):
        self.name = name
        self.parameters = parameters
        self.index = index
        self.mxfp8 = workspace.mxfp8
        self.member_shape = workspace.member_shapes[index]
        self.device = parameters[0].device
        self.gtp_leader = (
            parameters[0] if getattr(parameters[0], "is_gtp_weight_remat", False) else None
        )
        self.grad_dtype = workspace.grad_dtype
        self.virtual_grad = workspace.grad_slots(index)
        if self.gtp_leader is None:
            self.native_grad = workspace.native_staging(index)
            native_grads = tuple(self.native_grad)
            grad_bases = [grad.data_ptr() for grad in self.native_grad]
        else:
            # GTP: every backward binds the layer's GTP wgrad scratch (bind_wgrad_scratch), so
            # nothing is staged and nothing is copied on the way to the reduce-scatter. TE's
            # forward only checks that a fused-accumulation weight has *a* main_grad; an empty
            # placeholder satisfies it and fails loudly if a backward ever ran unbound.
            self.native_grad = None
            placeholder = torch.empty(0, dtype=self.grad_dtype, device=self.device)
            native_grads = (placeholder,) * len(parameters)
            grad_bases = [0] * len(parameters)
        self.wgrad_scratch: list[torch.Tensor] | None = None
        data, scales = workspace.slot_views(index)
        if self.mxfp8:
            template = parameters[0]
            if self.gtp_leader is not None:
                rowwise, columnwise = (
                    template._gtp_gather_quantizer.get_scale_shape(self.member_shape, columnwise=c)
                    for c in (False, True)
                )
            else:
                rowwise, columnwise = (
                    template._rowwise_scale_inv.shape,
                    template._columnwise_scale_inv.shape,
                )
            views = tuple(
                (data[i], scales[i].view(rowwise), data[i], scales[i].view(columnwise))
                for i in range(len(parameters))
            )
            self.virtual_weights = _wrap_mxfp8(template, self.member_shape, views, self.device)
            # GTP gathers hold one orientation; alias them into shells that always carry
            # both, so TE sees complete tensors while the live orientation stays exact.
            natives = _wrap_mxfp8(template, self.member_shape, views, self.device)
        else:
            self.virtual_weights = tuple(data)
            natives = (torch.empty(0, dtype=torch.bfloat16, device=self.device),) * len(data)
        sources = natives if self.gtp_leader is not None else parameters
        self.runtime_parameters = tuple(
            torch.nn.Parameter(weight) for weight in (*sources, *self.virtual_weights)
        )
        for parameter, grad in zip(self.runtime_parameters, (*native_grads, *self.virtual_grad)):
            parameter.main_grad = grad
            parameter.grad_added_to_main_grad = True
            parameter.overwrite_main_grad = True
            parameter.register_post_accumulate_grad_hook(_drop_grad)
        # Device pointer tables the kernels read, with pinned mirrors and copy-completion
        # events: per push direction the ``[components, L]`` weight table (data, then MXFP8
        # scales), and the ``[1, L]`` native wgrad table of the reduction.
        rows = (2 if self.mxfp8 else 1, 2 if self.mxfp8 else 1, 1)
        self.tables = tuple(
            torch.empty((r, len(parameters)), dtype=torch.int64, device=self.device) for r in rows
        )
        self.host_tables = tuple(
            torch.empty((r, len(parameters)), dtype=torch.int64, pin_memory=True) for r in rows
        )
        self.copied = [torch.cuda.Event() for _ in rows]
        self.bound = [None, None]
        self.native_grad_bases = self.tables[_GRAD][0]
        self._upload(_GRAD, [grad_bases])

    def _upload(self, table: int, rows) -> None:
        """Refresh device pointer table ``table`` through its pinned mirror, which may only be
        rewritten once the previous copy has landed."""
        self.copied[table].synchronize()
        self.host_tables[table].copy_(torch.tensor(rows, dtype=torch.int64))
        self.tables[table].copy_(self.host_tables[table], non_blocking=True)
        self.copied[table].record(torch.cuda.current_stream(self.device))

    def _components(self, direction: int) -> tuple[str, ...]:
        return _MXFP8_COMPONENTS[2 * direction : 2 * direction + 2] if self.mxfp8 else ("data",)

    def prepare(self, direction: int) -> None:
        """Materialize the source weights of ``direction`` and bind them for the push."""
        if self.gtp_leader is None:
            sources = self.parameters
        else:
            gathered = (
                self.gtp_leader.materialize_group_for_backward()
                if direction == BACKWARD
                else self.gtp_leader.materialize_group_for_forward()
            )
            sources = tuple(gathered) if isinstance(gathered, (list, tuple)) else (gathered,)
        components = self._components(direction)
        pointers = tuple(
            tuple(getattr(source, name).data_ptr() for name in components) for source in sources
        )
        if pointers == self.bound[direction]:
            return
        # A rebind is the exception (GTP gathers land in stable buffers); validate
        # the storage and refresh the table and the runtime parameters it describes.
        numel = math.prod(self.member_shape)
        expected = (
            ((numel, torch.uint8), (numel // 32, torch.uint8))
            if self.mxfp8
            else ((numel, torch.bfloat16),)
        )
        for source in sources:
            for name, (count, dtype) in zip(components, expected):
                storage = getattr(source, name)
                if (
                    not storage.is_contiguous()
                    or storage.numel() != count
                    or storage.dtype != dtype
                ):
                    raise ValueError(
                        f"{self.name} virtual-expert source {name} must be contiguous {dtype} with "
                        f"{count} elements, got {storage.dtype} {tuple(storage.shape)}."
                    )
        for parameter, source in zip(self.runtime_parameters, sources):
            if self.mxfp8:
                for name in components:
                    setattr(parameter, name, getattr(source, name))
            else:
                parameter.data = source
        self._upload(direction, list(zip(*pointers)))
        self.bound[direction] = pointers

    def bind_wgrad_scratch(self) -> None:
        """Point the natives' ``main_grad`` and the reduction's pointer table at this layer's GTP
        wgrad scratch, before the backward that fills them.

        TE's wgrad GEMM then writes the native experts' gradients straight into the buffers the
        reduce-scatter sends and the reduction adds the virtual-expert partials there, so
        :meth:`take_wgrads` hands them over without a copy. The scratch comes through the same
        protocol TE uses (``get_wgrad_tensor``); GTP recycles it once its reduce-scatter is done.
        """
        if self.gtp_leader is None:
            return
        if self.wgrad_scratch is not None:
            raise RuntimeError(f"{self.name} virtual-expert wgrad scratch is already bound.")
        scratch = [weight.get_wgrad_tensor() for weight in self.gtp_leader._weights]
        numel = math.prod(self.member_shape)
        if len(scratch) != len(self.parameters) or any(
            grad.dtype != self.grad_dtype
            or grad.numel() != numel
            or not grad.is_contiguous()
            or grad.data_ptr() % 16
            for grad in scratch
        ):
            raise ValueError(
                f"{self.name} GTP wgrad scratch must be {len(self.parameters)} contiguous "
                f"16-byte aligned {self.grad_dtype} buffers of {numel} elements."
            )
        for parameter, grad in zip(self.runtime_parameters, scratch):
            parameter.main_grad = grad
        self._upload(_GRAD, [[grad.data_ptr() for grad in scratch]])
        self.wgrad_scratch = scratch

    def take_wgrads(self) -> tuple[torch.Tensor, ...]:
        """Return the buffers holding this backward's full native wgrads: the staging, or the
        bound GTP scratch, which passes to the caller."""
        if self.gtp_leader is None:
            return tuple(self.native_grad)
        if self.wgrad_scratch is None:
            raise RuntimeError(
                f"{self.name} has no GTP wgrad scratch bound; the backward weight push must run "
                "before the expert backward."
            )
        scratch, self.wgrad_scratch = self.wgrad_scratch, None
        return tuple(scratch)


class VirtualExpertWeightBridge:
    """Asynchronous virtual-expert weight push and gradient reduction for one MoE layer."""

    def __init__(
        self,
        *,
        experts: torch.nn.Module,
        group: dist.ProcessGroup,
        num_local_experts: int,
        grad_dtype: torch.dtype = torch.float32,
        num_sms: int | None = None,
    ) -> None:
        self.group = group
        self.rank = dist.get_rank(group=group)
        self.world_size = dist.get_world_size(group=group)
        self.num_local_experts = num_local_experts
        self.num_runtime_experts = 2 * num_local_experts
        self._experts_ref = weakref.ref(experts)
        self.last_plan = None
        self._prefetch_plan = None
        self._completed_plan = None
        self._backward_plan = None
        self._reduced: set[int] = set()

        linears = (experts.linear_fc1, experts.linear_fc2)
        parameters = tuple(
            tuple(linear.get_parameter(f"weight{i}") for i in range(num_local_experts))
            for linear in linears
        )
        member_shapes = tuple(
            (int(linear.out_features), int(linear.in_features)) for linear in linears
        )
        self.device = parameters[0][0].device
        config = (
            self.world_size,
            num_local_experts,
            member_shapes,
            is_mxfp8tensor(parameters[0][0]),
            grad_dtype,
            min(32 if num_sms is None else int(num_sms), MAX_VIRTUAL_EXPERT_WEIGHT_SMS),
        )
        self.workspace = _get_workspace(group, self.device, config)
        self.projections = [
            _VirtualExpertProjection(f"FC{i + 1}", parameters[i], self.workspace, i)
            for i in range(2)
        ]
        # CUDA events are created lazily on first record; materialize them before
        # training or graph capture.
        self.prefetch_done = torch.cuda.Event()
        self.grad_reduce_done = (torch.cuda.Event(), torch.cuda.Event())
        for event in (self.prefetch_done, *self.grad_reduce_done):
            event.record(torch.cuda.current_stream(self.device))
        _bridges.add(self)

    @property
    def runtime_fc1_weights(self) -> tuple[torch.nn.Parameter, ...]:
        """Native-then-virtual-expert FC1 runtime parameters."""
        return self.projections[0].runtime_parameters

    @property
    def runtime_fc2_weights(self) -> tuple[torch.nn.Parameter, ...]:
        """Native-then-virtual-expert FC2 runtime parameters."""
        return self.projections[1].runtime_parameters

    @property
    def source_parameters(self) -> tuple[torch.nn.Parameter, ...]:
        """The optimizer-owned FC1 then FC2 parameters."""
        return tuple(parameter for p in self.projections for parameter in p.parameters)

    @torch.no_grad()
    @nvtx_decorator(message="virtual_expert_weight_push_start")
    def start_prefetch(self, plan: VirtualExpertPlan, direction: int = FORWARD) -> None:
        """Enqueue the owner push of the plan's FC1/FC2 weights on the weight stream."""
        if self._prefetch_plan is not None:
            raise RuntimeError("Virtual-expert weight prefetch is already outstanding.")
        if direction == FORWARD:
            # DDP/FSDP parameter hooks (all-gathers) must run before the push reads them.
            self._experts_ref().prepare_fused_impl_parameters()
        # Expert backward computes FC2 before FC1; keep GTP's linked gathers in that order.
        for projection in self.projections[:: -1 if direction == BACKWARD else 1]:
            projection.prepare(direction)
            if direction == BACKWARD:
                projection.bind_wgrad_scratch()
        workspace = self.workspace
        current_stream = torch.cuda.current_stream(self.device)
        weight_stream = workspace.weight_stream(current_stream)
        weight_stream.wait_stream(current_stream)
        tables = tuple(projection.tables[direction] for projection in self.projections)
        with torch.cuda.stream(weight_stream):
            launch_virtual_expert_weight_prefetch(
                sources=tuple(table[0] for table in tables),
                scale_sources=tuple(table[1] for table in tables) if workspace.mxfp8 else None,
                arena=workspace.weight_arena,
                peer_bases=workspace.weight_handle.buffer_ptrs_dev,
                signal_bases=workspace.weight_handle.signal_pad_ptrs_dev,
                experts_to_copy=plan.experts_to_copy,
                grid_barrier=workspace.weight_grid_barrier,
                rank=self.rank,
                world_size=self.world_size,
                num_local_experts=self.num_local_experts,
                member_numels=workspace.member_numels,
                num_sms=workspace.num_sms,
            )
            self.prefetch_done.record(weight_stream)
        self._prefetch_plan = plan

    @torch.no_grad()
    @nvtx_decorator(message="virtual_expert_weight_push_wait")
    def wait_prefetch(self, plan: VirtualExpertPlan) -> None:
        """Make the current stream wait for the push of ``plan``."""
        if self._prefetch_plan is None:
            # Waiting again for the resident plan is a no-op; anything else never started.
            if plan is None or plan is not self._completed_plan:
                raise RuntimeError("Virtual-expert weights require a started prefetch before use.")
        elif self._prefetch_plan is not plan:
            raise RuntimeError("Virtual-expert weight prefetch plan changed while outstanding.")
        torch.cuda.current_stream(self.device).wait_event(self.prefetch_done)
        self._completed_plan, self._prefetch_plan = plan, None

    def wait_prefetch_for_backward(self, plan: VirtualExpertPlan) -> None:
        """Wait for the backward push and remember the plan the expert backward reduces."""
        self.wait_prefetch(plan)
        self._backward_plan = plan

    @torch.no_grad()
    @nvtx_decorator(message="virtual_expert_grad_reduce_start")
    def start_grad_reduce(self, projection: int) -> None:
        """Enqueue the virtual-expert-gradient reduction of one projection (0 = FC1, 1 = FC2)."""
        if self._backward_plan is None:
            raise RuntimeError("Virtual-expert gradient reduction needs the backward plan.")
        if projection in self._reduced:
            raise RuntimeError(
                f"Virtual-expert gradient reduction of FC{projection + 1} started twice."
            )
        for p in self.projections:
            if p.gtp_leader is not None and p.wgrad_scratch is None:
                raise RuntimeError(f"{p.name} has no GTP wgrad scratch bound for this backward.")
        workspace = self.workspace
        workspace.grad_stream.wait_stream(torch.cuda.current_stream(self.device))
        with torch.cuda.stream(workspace.grad_stream):
            launch_virtual_expert_grad_reduce(
                arena=workspace.grad_arena,
                native_grads=tuple(p.native_grad_bases for p in self.projections),
                peer_bases=workspace.grad_handle.buffer_ptrs_dev,
                signal_bases=workspace.grad_handle.signal_pad_ptrs_dev,
                experts_to_copy=self._backward_plan.experts_to_copy,
                grid_barrier=workspace.grad_grid_barrier,
                rank=self.rank,
                world_size=self.world_size,
                num_local_experts=self.num_local_experts,
                member_numels=workspace.member_numels,
                num_sms=workspace.num_sms,
                projections=(projection,),
            )
            self.grad_reduce_done[projection].record(workspace.grad_stream)
        self._reduced.add(projection)

    def start_fc2_grad_reduce(self) -> None:
        """Start FC2's reduction from the expert backward, right behind FC2's wgrad GEMM."""
        self.start_grad_reduce(1)

    def start_pending_grad_reduces(self, plan: VirtualExpertPlan) -> None:
        """Start every reduction not yet started, FC2 first, once dispatch backward is done.

        FC2 normally starts from the FC2 op's wgrad store; FC1 starts here and hides behind
        the latent, shared-expert and router backward.
        """
        if plan is not self._backward_plan:
            raise RuntimeError("Virtual-expert gradient reduction is outstanding for another plan.")
        for projection in (1, 0):
            if projection not in self._reduced:
                self.start_grad_reduce(projection)

    @torch.no_grad()
    @nvtx_decorator(message="virtual_expert_grad_reduce_wait")
    def wait_grad_reduce(self, plan: VirtualExpertPlan) -> tuple[torch.Tensor | None, ...]:
        """Finish both reductions and return one full wgrad per source parameter.

        GTP parameters reduce-scatter the scratch the GEMM and the reduction wrote (FC2 first,
        as their linked reduce-scatter chain expects) and return what their protocol returns.
        """
        if plan is not self._backward_plan or self._reduced != {0, 1}:
            raise RuntimeError(
                "Virtual-expert gradient reduction of both projections must be started."
            )
        current_stream = torch.cuda.current_stream(self.device)
        for event in self.grad_reduce_done:
            current_stream.wait_event(event)
        self._backward_plan = None
        self._reduced.clear()
        grads = [projection.take_wgrads() for projection in self.projections]
        for index in (1, 0):
            leader = self.projections[index].gtp_leader
            if leader is not None:
                # GTP owns and recycles this scratch once its asynchronous reduce-scatter has
                # read it, so the next layer's GEMM never races it.
                reduced = leader.wgrad_reduce_scatter(list(grads[index]))
                grads[index] = tuple(reduced) if isinstance(reduced, (list, tuple)) else (reduced,)
        return tuple(grad for projection in grads for grad in projection)

    def release(self) -> None:
        """Drop the runtime parameters so the arenas they alias can be freed."""
        experts = self._experts_ref()
        if experts is not None:
            experts._fused_ops = experts._virtual_expert_weight_bridge = None
        for projection in self.projections:
            projection.wgrad_scratch = None
            for parameter in projection.runtime_parameters:
                parameter.main_grad = None
        self.projections.clear()
        self.workspace = None
        _bridges.discard(self)


class _VirtualExpertBackwardHook(torch.autograd.Function):
    """Run ``hook()`` when the gradient passes this point; the gradient itself is unchanged."""

    @staticmethod
    def forward(ctx, tensor, hook):
        ctx.hook = hook
        return tensor

    @staticmethod
    def backward(ctx, grad):
        ctx.hook()
        return grad, None


class _VirtualExpertWaitGradReduce(torch.autograd.Function):
    """Finish the virtual-expert reductions and hand the wgrads to the source parameters.

    Applied to the router input. Its backward needs the router's input gradient, hence the
    dispatch backward; the FC1 reduction start sits on the dispatch input, is created later in
    the forward and so runs first under autograd's ordering, and ``wait_grad_reduce`` raises
    if it did not. ``context.plan`` is filled in by the dispatcher once routing has produced
    the plan.
    """

    @staticmethod
    def forward(ctx, hidden_states, *args):
        ctx.bridge, ctx.context = args[-2:]
        return hidden_states

    @staticmethod
    def backward(ctx, grad_hidden_states):
        from transformer_engine.pytorch.module.base import get_dummy_wgrad

        grads = []
        wgrads = ctx.bridge.wait_grad_reduce(ctx.context.plan)
        for parameter, wgrad in zip(ctx.bridge.source_parameters, wgrads):
            if wgrad is None or getattr(parameter, "is_gtp_weight_remat", False):
                grads.append(wgrad)  # GTP already reduce-scattered into the shard
            elif getattr(parameter, "main_grad", None) is None:
                grads.append(wgrad.clone())  # no fused accumulation: autograd owns a copy
            else:
                # Accumulate in main_grad's dtype and return a dummy so AccumulateGrad
                # still fires DDP's grad-ready hook without adding the dummy again.
                parameter.main_grad.add_(wgrad)
                parameter.grad_added_to_main_grad = True
                grads.append(
                    get_dummy_wgrad(
                        list(parameter.shape),
                        parameter.dtype,
                        zero=getattr(parameter, "zero_out_wgrad", False),
                    )
                )
        return (grad_hidden_states, *grads, None, None)


class VirtualExpertLoadBalancer:
    """Implements the load balancing scheme by MoonEP's (https://github.com/moonshotAI/moonep)
    This class is expected to be a mixin for moe.token_dispatcher.MoEFlexTokenDispatcher.

    Every EP rank receives exactly `tokens x topk` routes however skewed the routing is.
    Redundant ("virtual") experts per rank are prefetched into local slots before expert compute
    and whose gradients are reduced back to the owning rank in backward. A rank duplicates experts
    from a single overloaded home rank, so ``num_local_experts`` slots always suffice. The token
    transport is handled by the underlying token dispatcher while the virtual expert materialization
    and grad reduction are handled by Triton kernels in virtual_expert_triton.py.

    """

    def initialize_virtual_expert_load_balancer(
        self,
        *,
        group: torch.distributed.ProcessGroup,
        num_local_experts: int,
        router_topk: int,
        num_experts: int,
        config: "TransformerConfig",
    ) -> None:
        """Initialize virtual-expert state without initializing the transport parent."""
        if not _TRITON_AVAILABLE:
            raise ImportError("--moe-virtual-expert-load-balance requires Triton.")
        world_size = torch.distributed.get_world_size(group=group)
        if num_experts != world_size * num_local_experts:
            raise ValueError(
                "Virtual-expert load balancing requires an even expert distribution: "
                f"num_experts={num_experts}, world_size={world_size}, "
                f"num_local_experts={num_local_experts}."
            )
        self.group = group
        self.config = config
        self.router_topk = router_topk
        self.semantic_num_experts = num_experts
        self.num_owned_experts = num_local_experts
        self.routes: tuple[torch.Tensor, torch.Tensor] | None = None
        self.runtime_probs: torch.Tensor | None = None
        # Placement scratch shared by every layer of this device and group (planning is
        # stream-ordered); resolved at the first plan, when the group's communicator exists.
        self._planner_workspace: VirtualExpertPlannerWorkspace | None = None
        self._experts_ref = None
        self._bridge = None
        self._plan = None
        self._context = None

    def bind_experts(self, experts: torch.nn.Module) -> None:
        """Remember the expert module; runtime slots are bound after main-grad initialization."""
        self._experts_ref = weakref.ref(experts)

    def _ensure_experts_bound(self) -> VirtualExpertWeightBridge:
        """Bind runtime slots using the optimizer-owned main-gradient dtype."""
        if self._bridge is not None:
            return self._bridge
        experts = self._experts_ref() if self._experts_ref is not None else None
        if experts is None:
            raise RuntimeError("Virtual-expert load balancer has no bound expert module.")

        source_parameters = tuple(
            linear.get_parameter(f"weight{index}")
            for linear in (experts.linear_fc1, experts.linear_fc2)
            for index in range(self.num_owned_experts)
        )
        main_grads = tuple(getattr(parameter, "main_grad", None) for parameter in source_parameters)
        initialized_main_grads = tuple(grad for grad in main_grads if grad is not None)
        if initialized_main_grads and len(initialized_main_grads) != len(main_grads):
            raise RuntimeError(
                "Virtual-expert source main gradients must be initialized for every expert weight."
            )
        grad_dtypes = {grad.dtype for grad in initialized_main_grads}
        if len(grad_dtypes) > 1:
            raise RuntimeError(
                f"Virtual-expert source main gradients must have one dtype, got {grad_dtypes}."
            )
        # A no-DDP inference caller has no main gradients. Preserve the historical FP32 arena
        # default in that case; training derives the dtype from DDP's initialized buffers.
        grad_dtype = next(iter(grad_dtypes), torch.float32)
        self._bridge = VirtualExpertWeightBridge(
            experts=experts,
            group=self.group,
            num_local_experts=self.num_owned_experts,
            grad_dtype=grad_dtype,
            num_sms=self.config.moe_flex_dispatcher_num_sms,
        )
        experts.set_virtual_expert_weight_bridge(self._bridge)
        return self._bridge

    def wrap_layer_input(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Attach the virtual-expert-gradient completion hook to the whole MoE layer input."""
        bridge = self._ensure_experts_bound()
        if self._context is not None:
            raise RuntimeError("Virtual-expert layer input wrapped twice without a combine.")
        self._context = SimpleNamespace(plan=None)
        return _VirtualExpertWaitGradReduce.apply(
            hidden_states, *bridge.source_parameters, bridge, self._context
        )

    def setup_virtual_expert_metadata(self, top_indices: torch.Tensor, probs: torch.Tensor) -> None:
        """Keep the router's ``[num_tokens, topk]`` expert ids and probabilities for the planner."""
        self.routes = (top_indices, probs)
        self.num_local_tokens = int(top_indices.shape[0])
        self.token_probs = probs

    def plan_dispatch(self) -> None:
        """Plan routes and begin the weight push before shared-expert compute."""
        if self._bridge is None or self._context is None or self._plan is not None:
            raise RuntimeError(
                "Virtual-expert planning needs bound experts, a wrapped layer input and a "
                "combined previous dispatch."
            )
        top_indices, probs = self.routes
        self.routes = None
        if self._planner_workspace is None:
            self._planner_workspace = get_planner_workspace(
                num_experts=self.semantic_num_experts, device=probs.device, group=self.group
            )
        plan, self.runtime_probs = plan_virtual_expert_routes(
            top_indices, probs, self._planner_workspace
        )
        self._plan = self._context.plan = self._bridge.last_plan = plan
        self._bridge.start_prefetch(plan)

    def prepare_virtual_expert_dispatch(
        self, hidden_states: torch.Tensor, *, num_runtime_experts: int, alignment: int
    ) -> tuple[torch.Tensor, VirtualExpertPlan, int]:
        """Attach dispatch-backward work and calculate the transport capacity."""
        plan = self._plan
        if plan is None:
            raise RuntimeError("Virtual-expert dispatch requires plan_dispatch to run first.")
        num_permuted_tokens = self._get_rank_capacity(
            num_tokens=self.num_local_tokens,
            router_topk=self.router_topk,
            capacity_factor=self.config.moe_expert_rank_capacity_factor,
            num_runtime_experts=num_runtime_experts,
            alignment=alignment,
        )
        hidden_states = _VirtualExpertBackwardHook.apply(
            hidden_states, functools.partial(self._bridge.start_pending_grad_reduces, plan)
        )
        return hidden_states, plan, num_permuted_tokens

    def prepare_virtual_expert_combine(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Attach the backward weight-push wait before transport combine."""
        if self._plan is None:
            raise RuntimeError("Virtual-expert combine requires a matching dispatch plan.")
        return _VirtualExpertBackwardHook.apply(
            hidden_states, functools.partial(self._bridge.wait_prefetch_for_backward, self._plan)
        )

    def finalize_output(self, output: torch.Tensor) -> torch.Tensor:
        """Start the backward weight push from the MoE layer output."""
        plan, self._plan, self._context = self._plan, None, None
        if plan is None:
            raise RuntimeError("Virtual-expert output finalization requires a combined plan.")
        return _VirtualExpertBackwardHook.apply(
            output, functools.partial(self._bridge.start_prefetch, plan, BACKWARD)
        )

    @staticmethod
    def _get_rank_capacity(
        *,
        num_tokens: int,
        router_topk: int,
        capacity_factor: float,
        num_runtime_experts: int,
        alignment: int,
    ) -> int:
        """Return a static, dropless route capacity for one transport rank: every rank receives
        exactly its own route count, plus HybridEP's per-runtime-expert segment padding."""
        num_routes = num_tokens * router_topk
        padding = num_runtime_experts * max(alignment - 1, 0)
        capacity = max(int(num_routes * capacity_factor), num_routes + padding)
        return capacity + (-capacity % alignment if alignment > 1 else 0)
