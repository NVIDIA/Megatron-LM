# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.


"""Virtual-expert load balancing, virtual-expert planning, and runtime weight movement.
The load-balancing algorithm implemented here is taken from MoonEP
(https://github.com/moonshotAI/moonep): perfectly balanced expert parallelism through
redundant experts planned online from the router output.
"""

import gc
import math
import weakref
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import torch
import torch.distributed as dist

from megatron.core.fp8_utils import is_mxfp8tensor
from megatron.core.transformer.moe.moe_utils import get_align_size_for_quantization

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
from megatron.core.utils import ensure_params_ready, nvtx_decorator

if TYPE_CHECKING:
    from megatron.core.transformer.transformer_config import TransformerConfig

# Push directions. MXFP8 forward GEMMs read the rowwise components, backward the columnwise.
FORWARD, BACKWARD = 0, 1
# Index of a FC layer's native wgrad pointer table, after the two push directions.
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
    size; mirrors the field offsets in ``_plan_virtual_expert_routes_kernel``."""
    block_ep = 1 << (ep_size - 1).bit_length()
    # Each flag word gets its own 128-byte line (the kernel's _FLAG_STRIDE).
    fields = (
        ("placement_grid_sync", (1,)),
        ("_pad0", (31,)),
        ("grid_sync", (1,)),
        ("_pad1", (31,)),
        ("balance", (ep_size,)),  # native load minus rank capacity
        ("allocation", (num_experts, ep_size)),  # routes of each expert per destination
        ("destination_boundaries", (num_experts, block_ep)),  # segment ends, local ordinals
        ("virtual_expert_slots", (num_experts, ep_size)),  # slot holding an expert on a rank
        ("program_histogram", (PLANNER_PROGRAMS, num_experts)),
        ("running_counts", (PLANNER_PROGRAMS, num_experts)),
        ("tokens_per_expert", (num_experts,)),  # atomic histogram target, zeroed after use
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
        """Experts owned by each EP rank."""
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
        if handle.signal_pad_size < ep_size * 4:
            raise RuntimeError(
                "Virtual-expert planner needs one signal word per EP rank; the symmetric "
                f"memory signal pad holds {handle.signal_pad_size} bytes for {ep_size} ranks."
            )
        _, size = _scratch_layout(num_experts, ep_size)
        return cls(
            num_experts=num_experts,
            ep_size=ep_size,
            rank=dist.get_rank(group=group),
            gathered_counts=window.view(ep_size, num_experts),
            histogram_handle=handle,
            scratch=torch.zeros(size, dtype=torch.int32, device=device),
        )


class _PlanRoutes(torch.autograd.Function):
    """One planner launch. The dense runtime probabilities it writes carry the gradient back to
    the router's ``[num_tokens, topk]`` probabilities through a gather at the runtime ids."""

    @staticmethod
    def forward(ctx, probs, top_indices, workspace):
        virtual_experts, runtime_probs, experts_to_copy = launch_virtual_expert_planner(
            top_indices, probs, workspace
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
            return None, None, None
        (virtual_experts,) = ctx.saved_tensors
        grad_probs = grad_runtime_probs.gather(1, virtual_experts.long()).to(ctx.probs_dtype)
        return grad_probs, None, None


def plan_virtual_expert_routes(
    top_indices: torch.Tensor, probs: torch.Tensor, workspace: VirtualExpertPlannerWorkspace
) -> tuple[VirtualExpertPlan, torch.Tensor]:
    """Plan deterministic virtual-expert placement for one EP group and map this rank's routes.

    ``top_indices`` / ``probs`` are the router's ``[num_tokens, topk]`` expert ids and
    probabilities; every rank must route the same number of tokens. The histograms are the only
    cross-rank input and the planner kernel exchanges them itself, so every rank computes the
    same placement. Returns the plan and the dense float32 ``[num_tokens, 2 * num_experts]`` runtime
    probabilities HybridEP consumes, which carry the gradient back to ``probs``.
    """
    if top_indices.shape != probs.shape or top_indices.dtype not in (torch.int32, torch.int64):
        raise ValueError("Virtual-expert planner takes matching [num_tokens, topk] ids and probs.")
    top_indices, probs = top_indices.contiguous(), probs.contiguous()
    virtual_experts, runtime_probs, experts_to_copy = _PlanRoutes.apply(
        probs, top_indices, workspace
    )
    return VirtualExpertPlan(virtual_experts, experts_to_copy), runtime_probs


# --------------------------------------------------------------------------------------
# Workspace: arenas, planner scratch and streams shared by every layer of the process
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


class _VirtualExpertWorkspace:
    """Planner scratch, symmetric arenas and streams shared by every virtual-expert MoE layer.

    A training process has one device, one EP group and one expert layout, so it owns one
    workspace, ``VirtualExpertLoadBalancer.workspace``, allocated when the first layer binds
    its runtime weights and released by :func:`finalize_virtual_experts`. The weight arena
    holds ``fc1 data, fc1 scales, fc2 data, fc2 scales`` with ``L`` members per section (the
    scale sections are empty for BF16; MXFP8 keeps one because only one GEMM orientation is
    live at a time). The gradient arena holds ``fc1, fc2``. One layer's virtual experts are
    live at a time: the planner's histogram
    exchange orders every push after the expert GEMMs that read the previous contents, and the
    reduction's exit rendezvous orders every slot rewrite after the owners' reads.
    """

    def __init__(self, group, device, config: tuple) -> None:
        import torch.distributed._symmetric_memory as symm_mem

        self.group_name = group.group_name
        self.device = device
        self.config = config
        world_size, num_local_experts, member_shapes, mxfp8, grad_dtype, num_sms = config
        self.rank = dist.get_rank(group=group)
        self.world_size = world_size
        self.num_local_experts = num_local_experts
        self.member_shapes = member_shapes
        self.member_numels = tuple(math.prod(shape) for shape in member_shapes)
        self.mxfp8 = mxfp8
        self.grad_dtype = grad_dtype
        self.num_sms = num_sms
        # One E8M0 scale byte per 32 MXFP8 weight bytes, unpadded (the config requires
        # 128-aligned FC layers so TE's padded scale layout has this exact size).
        self.scale_numels = tuple(numel // 32 if mxfp8 else 0 for numel in self.member_numels)
        arena_numel = num_local_experts * sum(self.member_numels)
        # The planner's allocation runs the first collective on the group, which creates the
        # device communicator the NCCL window registrations below need.
        self.planner = VirtualExpertPlannerWorkspace.allocate(
            num_experts=world_size * num_local_experts, device=device, group=group
        )
        try:
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
        # Full native wgrad staging per FC layer, allocated on first use: TE's GEMM overwrites
        # it, the reduction adds the virtual-expert partials, autograd hands it to the optimizer.
        # GTP FC layers write per-backward GTP scratch instead and never allocate it.
        self._native_grads: dict[int, torch.Tensor] = {}

    def weight_stream(self, current_stream: torch.cuda.Stream) -> torch.cuda.Stream:
        """Return a weight stream distinct from ``current_stream``."""
        return next(s for s in self.weight_streams if s.cuda_stream != current_stream.cuda_stream)

    def slot_views(self, fc_layer: int) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Return the ``[L, *shape]`` weight slots of one FC layer and, for MXFP8, their
        ``[L, numel // 32]`` scale bytes."""
        count, shape = self.num_local_experts, self.member_shapes[fc_layer]
        numel, scale_numel = self.member_numels[fc_layer], self.scale_numels[fc_layer]
        offset = count * sum(self.member_numels[:fc_layer]) + count * sum(
            self.scale_numels[:fc_layer]
        )
        data = self.weight_arena.narrow(0, offset, count * numel).view(count, *shape)
        if not self.mxfp8:
            return data, None
        scales = self.weight_arena.narrow(0, offset + count * numel, count * scale_numel)
        return data, scales.view(count, scale_numel)

    def native_staging(self, fc_layer: int) -> torch.Tensor:
        """Return the ``[L, *shape]`` native wgrad staging of one FC layer."""
        staging = self._native_grads.get(fc_layer)
        if staging is None:
            staging = self._native_grads[fc_layer] = torch.empty(
                (self.num_local_experts, *self.member_shapes[fc_layer]),
                dtype=self.grad_dtype,
                device=self.grad_arena.device,
            )
        return staging

    def grad_slots(self, fc_layer: int) -> torch.Tensor:
        """Return the ``[L, *shape]`` virtual-expert gradient slots of one FC layer."""
        count, numel = self.num_local_experts, self.member_numels[fc_layer]
        offset = count * sum(self.member_numels[:fc_layer])
        return self.grad_arena.narrow(0, offset, count * numel).view(
            count, *self.member_shapes[fc_layer]
        )

    def destroy(self) -> None:
        """Drop the NCCL window registrations while the process group is still alive."""
        if self.weight_arena is not None:
            torch.cuda.synchronize(self.device)
        self.planner.destroy()
        self.weight_handle = self.grad_handle = self.weight_arena = self.grad_arena = None


_load_balancers = weakref.WeakSet()


def finalize_virtual_experts() -> None:
    """Release every virtual-expert layer's runtime parameters and the shared arenas; idempotent.

    A normal exit needs no call. Callers that destroy their process groups while a model is
    still alive (tests, an orderly shutdown) call it first so the NCCL windows are deregistered
    while their communicator exists.
    """
    try:
        for load_balancer in list(_load_balancers):
            load_balancer.release()
        if VirtualExpertLoadBalancer.workspace is not None:
            VirtualExpertLoadBalancer.workspace.destroy()
    except Exception:  # a teardown release must never raise
        pass
    VirtualExpertLoadBalancer.workspace = None
    # The runtime parameters and their TE ops sit in reference cycles; free the arenas now.
    gc.collect()


# --------------------------------------------------------------------------------------
# Runtime parameters and pointer tables
# --------------------------------------------------------------------------------------


def _drop_grad(parameter: torch.nn.Parameter) -> None:
    """TE returns a dummy leaf grad once the fused wgrad is in ``main_grad``; drop it."""
    parameter.grad = None


class _VirtualExpertFCLayer:
    """One FC layer's optimizer parameters, runtime parameters and pointer tables.

    The ``2L`` runtime parameters are the natives followed by the virtual-expert slots. Their
    ``main_grad`` is the native staging (GTP: the layer's GTP wgrad scratch, acquired per
    backward by :meth:`acquire_wgrad_scratch`) or the slot's gradient arena member and carries
    ``overwrite_main_grad``, so TE's wgrad GEMM rewrites every member on each backward and the
    slots never need clearing (a planned slot always receives tokens).

    The kernels read the members' base addresses from device pointer tables (one row per
    component, one column per local expert), so their launch arguments stay fixed. Plain
    parameters and their staging never move: their tables are written once, here. GTP gathers
    land in per-ticket buffers that stay put: bound on first use, then one canary (the first
    expert's pointers) is checked per use and a change rebinds everything. GTP wgrad scratch
    comes from a pool: bound on every backward. Under GTP the push only *peeks* at the gathered
    weights (:meth:`prepare`); GTP consumes them where TE would, right before the expert GEMMs
    (:meth:`consume`), so virtual experts leave its gather and prefetch schedule alone.
    """

    def __init__(self, name, parameters, workspace: _VirtualExpertWorkspace, index: int):
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
        else:
            # GTP: every backward acquires the layer's GTP wgrad scratch, so nothing is staged
            # and nothing is copied on the way to the reduce-scatter. TE's forward only checks
            # that a fused-accumulation weight has *a* main_grad; this empty placeholder
            # satisfies it and fails loudly if a backward ever ran without acquired scratch.
            self.native_grad = None
            self.placeholder = torch.empty(0, dtype=self.grad_dtype, device=self.device)
            native_grads = (self.placeholder,) * len(parameters)
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
        # What each table describes: the canary per push direction, every base of the reduction.
        self.bound = [None, None, None]
        self.native_grad_bases = self.tables[_GRAD][0]
        if self.gtp_leader is None:
            for direction in (FORWARD, BACKWARD):
                self._bind(direction, parameters)
            self._upload(_GRAD, [[grad.data_ptr() for grad in self.native_grad]])

    def _upload(self, table: int, rows) -> None:
        """Write device pointer table ``table`` through its pinned mirror.

        A pageable host-to-device copy would block the host until the stream drains, so the rows
        go through a pinned mirror with an asynchronous copy, and the mirror may only be
        rewritten once the previous copy has landed (long done in practice: query, do not block).
        """
        if not self.copied[table].query():
            self.copied[table].synchronize()
        self.host_tables[table].copy_(torch.tensor(rows, dtype=torch.int64))
        self.tables[table].copy_(self.host_tables[table], non_blocking=True)
        self.copied[table].record(torch.cuda.current_stream(self.device))

    def _components(self, direction: int) -> tuple[str, ...]:
        return _MXFP8_COMPONENTS[2 * direction : 2 * direction + 2] if self.mxfp8 else ("data",)

    def _canary(self, direction: int, sources) -> tuple:
        """The first source's component pointers for ``direction``: the buffers of one gather
        or one parameter set only ever move together, so one expert stands for all."""
        return tuple(getattr(sources[0], name).data_ptr() for name in self._components(direction))

    def _gathered(self, direction: int, peek: bool) -> tuple:
        """Peek at or consume the GTP group's gathered weights of ``direction``."""
        leader = self.gtp_leader
        if direction == BACKWARD:
            gathered = (
                leader.peek_group_for_backward()
                if peek
                else leader.materialize_group_for_backward()
            )
        else:
            gathered = (
                leader.peek_group_for_forward() if peek else leader.materialize_group_for_forward()
            )
        return tuple(gathered) if isinstance(gathered, (list, tuple)) else (gathered,)

    def _bind(self, direction: int, sources) -> None:
        """Validate the sources of ``direction``, point the runtime natives at their storage and
        write the push table."""
        components = self._components(direction)
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
        self._upload(
            direction,
            [[getattr(source, name).data_ptr() for source in sources] for name in components],
        )
        self.bound[direction] = self._canary(direction, sources)

    def prepare(self, direction: int) -> None:
        """Make the push table of ``direction`` current.

        Plain parameters are read once their DDP publication has landed. GTP weights are read
        from their gathered buffers without consuming them in GTP's chain (a peek, see
        ``GTPShardedParam._peek_gathered_weight``); :meth:`consume` follows at the GEMM.
        """
        if self.gtp_leader is None:
            # The push reads the parameters ahead of the expert module's pre-forward hook, where
            # DDP (overlap_param_gather) would otherwise finish publishing them.
            ensure_params_ready(self.parameters)
            sources = self.parameters
        else:
            sources = self._gathered(direction, peek=True)
        if self._canary(direction, sources) != self.bound[direction]:
            self._bind(direction, sources)

    def consume(self, direction: int) -> None:
        """Consume the GTP weights of ``direction`` in GTP's prefetch chain.

        This is GTP's real consume, the one TE would issue right before the expert GEMMs: it
        waits for this group's gather and issues the chain's next prefetch. The push has already
        read the buffers it returns (:meth:`prepare` peeked at the same gather), so they must be
        the bound ones; anything else means the push copied other bytes than the GEMM reads.
        """
        if self.gtp_leader is None:
            return
        sources = self._gathered(direction, peek=False)
        if self._canary(direction, sources) != self.bound[direction]:
            raise RuntimeError(
                f"{self.name}: GTP consumed the expert weights from other buffers than the ones "
                "the virtual-expert push read; the push must peek at the gather the GEMM consumes."
            )

    def acquire_wgrad_scratch(self) -> tuple[torch.Tensor, ...] | None:
        """GTP: point the natives' ``main_grad`` and the reduction's table at GTP wgrad scratch
        for the backward about to run and return it (``None`` for plain parameters, whose staging
        is fixed). TE's wgrad GEMM writes the natives' gradients straight into the buffers the
        reduce-scatter sends and the reduction adds the virtual-expert partials there, so nothing
        is copied. The scratch comes through the protocol TE's modules use (``get_wgrad_tensor``);
        GTP recycles it once its reduce-scatter has read it, hence one acquire per backward. Its
        LIFO pool usually hands back the same buffers, so the table is rewritten only on change.
        """
        if self.gtp_leader is None:
            return None
        scratch = tuple(weight.get_wgrad_tensor() for weight in self.gtp_leader._weights)
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
        bases = [grad.data_ptr() for grad in scratch]
        if bases != self.bound[_GRAD]:
            self._upload(_GRAD, [bases])
            self.bound[_GRAD] = bases
        return scratch

    def release_wgrad_scratch(self) -> None:
        """GTP: return the natives' ``main_grad`` to the placeholder once the scratch has been
        handed to GTP, so a backward without a fresh acquire fails instead of writing into
        recycled scratch."""
        if self.gtp_leader is None:
            return
        for parameter in self.runtime_parameters[: len(self.parameters)]:
            parameter.main_grad = self.placeholder


# --------------------------------------------------------------------------------------
# Load balancer
# --------------------------------------------------------------------------------------


@dataclass(slots=True)
class _PassTemporaries:
    """What one forward/backward pass of the layer holds in flight, from its layer input to the
    end of its backward. The manager owns the pass during its forward and again during its
    backward, taking it back at the layer-output hook; a repeated (MTP) layer has several forwards
    outstanding, so that hook carries the pass. Only the plan and the flags outlive the forward:
    the tensors are consumed within it, so an autograd context holding this object closes no
    cycle through the graph.
    """

    plan: VirtualExpertPlan | None = None
    # The dense runtime probabilities HybridEP consumes, from the plan to dispatch.
    runtime_probs: torch.Tensor | None = None
    # The weight push of this pass: in flight, or already waited (a repeated wait is a no-op).
    push_in_flight: bool = False
    push_done: bool = False
    # The backward: each FC layer's acquired GTP scratch (None for plain parameters) and the
    # FC layers whose reduction was launched.
    scratch: tuple | None = None
    started: set = field(default_factory=set)


def _weak_hook(method, *args):
    """Bind ``method(*args)`` for an autograd hook without keeping its owner alive.

    The owner is the dispatcher's manager, which holds differentiable tensors (the dispatched
    probabilities, the routing state) across a forward; a strong reference from an autograd
    context would close a reference cycle through the graph that Python's collector cannot see,
    leaking every layer's graph. The model keeps the owner alive as long as the hooks can fire.
    """
    bound = weakref.WeakMethod(method)

    def hook():
        return bound()(*args)

    return hook


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
    the forward and so runs first under autograd's ordering, and ``_finish_grad_reduce`` raises
    if it did not. ``load_balancer`` is a weak reference (see :func:`_weak_hook`); the pass it
    finishes is the one the layer-output hook handed the manager.
    """

    @staticmethod
    def forward(ctx, hidden_states, *args):
        ctx.load_balancer = args[-1]
        return hidden_states

    @staticmethod
    def backward(ctx, grad_hidden_states):
        from transformer_engine.pytorch.module.base import get_dummy_wgrad

        load_balancer = ctx.load_balancer()
        grads = []
        wgrads = load_balancer._finish_grad_reduce()
        for parameter, wgrad in zip(load_balancer.source_parameters, wgrads):
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
        return (grad_hidden_states, *grads, None)


class VirtualExpertLoadBalancer:
    """MoonEP's load balancing (https://github.com/moonshotAI/moonep) as a mixin for
    ``moe.token_dispatcher.MoEFlexTokenDispatcher``'s HybridEP manager.

    Every EP rank receives exactly ``tokens x topk`` routes however skewed the routing is:
    redundant ("virtual") experts are pushed into local slots before expert compute and their
    gradients are reduced back to the owning rank in backward. A rank duplicates experts from a
    single overloaded home rank, so ``num_local_experts`` slots always suffice. The token
    transport is the underlying dispatcher's; the planner, the weight push and the gradient
    reduction are the Triton kernels in ``virtual_expert_triton.py``, driven from here. The push
    starts at the planner (forward) and at the layer output (backward); GTP's weights are only
    peeked at and consumed right before the expert GEMMs, where TE consumes them, and their
    reduce-scatter is GTP's own protocol call, issued once the reductions have completed
    (:meth:`_finish_grad_reduce`).
    """

    # The process-wide workspace (planner scratch, arenas, streams): a process has one device,
    # one EP group and one expert layout, so the first layer to bind allocates it for all.
    workspace: _VirtualExpertWorkspace | None = None

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
        self.ep_size = torch.distributed.get_world_size(group=group)
        self.ep_rank = torch.distributed.get_rank(group=group)
        if num_experts != self.ep_size * num_local_experts:
            raise ValueError(
                "Virtual-expert load balancing requires an even expert distribution: "
                f"num_experts={num_experts}, world_size={self.ep_size}, "
                f"num_local_experts={num_local_experts}."
            )
        self.group = group
        self.config = config
        self.router_topk = router_topk
        self.semantic_num_experts = num_experts
        self.num_owned_experts = num_local_experts
        # HybridEP pads every runtime expert's segment to the quantization block.
        self._alignment = get_align_size_for_quantization(config)
        # The layer's token count and this rank's transport capacity for it, sized at the first
        # forward; the count is fixed in practice.
        self.num_tokens: int | None = None
        self.rank_capacity: int | None = None
        self._experts_ref = None
        # The pass this manager currently owns: from the layer input to the layer output in the
        # forward, and from the backward push wait to the wgrad hand-off in the backward.
        self._temporaries: _PassTemporaries | None = None
        self.device = torch.device("cuda", torch.cuda.current_device())
        # CUDA events are created lazily on first record; materialize them before training or
        # graph capture.
        self.prefetch_done = torch.cuda.Event()
        self.grad_reduce_done = torch.cuda.Event()
        for event in (self.prefetch_done, self.grad_reduce_done):
            event.record(torch.cuda.current_stream(self.device))
        # The runtime parameters are bound at the first forward, once DDP has built the main
        # gradients whose dtype the arenas take.
        self.fc_layers: list[_VirtualExpertFCLayer] = []

    # ---- binding -----------------------------------------------------------------------------

    def bind_experts(self, experts: torch.nn.Module) -> None:
        """Remember the expert module; runtime slots are bound after main-grad initialization."""
        self._experts_ref = weakref.ref(experts)

    def _runtime_init(self, hidden_states: torch.Tensor) -> None:
        """The start of a layer forward: check that the previous pass unwound completely, size
        this rank's transport capacity for the layer's token count (fixed in practice, re-sized
        only if it changes) and, once per process, build the runtime parameters over the shared
        arenas. A no-op after the first forward but for the checks."""
        if self._temporaries is not None:
            raise RuntimeError(
                "Virtual-expert layer input wrapped while the previous pass is still in flight: "
                "its forward was not finalized or its backward did not finish."
            )
        num_tokens = hidden_states.numel() // hidden_states.shape[-1]
        if num_tokens != self.num_tokens:
            self.num_tokens = num_tokens
            self.rank_capacity = self._get_rank_capacity(
                num_tokens=num_tokens,
                router_topk=self.router_topk,
                capacity_factor=self.config.moe_expert_rank_capacity_factor,
                num_runtime_experts=self.num_runtime_experts,
                alignment=self._alignment,
            )

        if self.fc_layers:
            return

        experts = self._experts_ref() if self._experts_ref is not None else None
        if experts is None:
            raise RuntimeError("Virtual-expert load balancer has no bound expert module.")

        linears = (experts.linear_fc1, experts.linear_fc2)
        parameters = tuple(
            tuple(linear.get_parameter(f"weight{i}") for i in range(self.num_owned_experts))
            for linear in linears
        )
        main_grads = tuple(
            getattr(parameter, "main_grad", None) for group in parameters for parameter in group
        )
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
        num_sms = self.config.moe_flex_dispatcher_num_sms
        config = (
            self.ep_size,
            self.num_owned_experts,
            tuple((int(linear.out_features), int(linear.in_features)) for linear in linears),
            is_mxfp8tensor(parameters[0][0]),
            grad_dtype,
            min(32 if num_sms is None else int(num_sms), MAX_VIRTUAL_EXPERT_WEIGHT_SMS),
        )
        workspace = VirtualExpertLoadBalancer.workspace
        if VirtualExpertLoadBalancer.workspace is None:
            workspace = _VirtualExpertWorkspace(self.group, self.device, config)
            VirtualExpertLoadBalancer.workspace = workspace
        elif (workspace.group_name, workspace.device, workspace.config) != (
            self.group.group_name,
            self.device,
            config,
        ):
            raise ValueError(
                "All virtual-expert MoE layers of a process must share one EP group, device and "
                f"weight configuration; have {workspace.config} on {workspace.group_name}, got "
                f"{config} on {self.group.group_name}. Callers that rebuild their process groups "
                "must call finalize_virtual_experts() first."
            )
        self.fc_layers = [
            _VirtualExpertFCLayer(f"FC{i + 1}", parameters[i], self.workspace, i) for i in range(2)
        ]
        experts.bind_virtual_experts(self)
        _load_balancers.add(self)

    def release(self) -> None:
        """Drop the runtime parameters so the arenas they alias can be freed."""
        experts = self._experts_ref() if self._experts_ref is not None else None
        if experts is not None:
            experts._fused_ops = experts._virtual_experts = None
        for fc_layer in self.fc_layers:
            for parameter in fc_layer.runtime_parameters:
                parameter.main_grad = None
        self.fc_layers = []
        self._temporaries = None
        _load_balancers.discard(self)

    @property
    def num_runtime_experts(self) -> int:
        """Natives plus virtual-expert slots per rank."""
        return 2 * self.num_owned_experts

    def runtime_weights(self, fc_layer: int) -> tuple[torch.nn.Parameter, ...]:
        """Native-then-virtual-expert runtime parameters of one FC layer (0 = FC1, 1 = FC2)."""
        return self.fc_layers[fc_layer].runtime_parameters

    @property
    def source_parameters(self) -> tuple[torch.nn.Parameter, ...]:
        """The optimizer-owned FC1 then FC2 parameters."""
        return tuple(parameter for p in self.fc_layers for parameter in p.parameters)

    def _fc_layers(self, direction: int) -> list:
        """The FC layers in GEMM order: the expert backward computes FC2 before FC1, and
        GTP's linked gathers are consumed in that order."""
        return self.fc_layers[:: -1 if direction == BACKWARD else 1]

    # ---- forward: dispatcher hooks -----------------------------------------------------------

    def wrap_layer_input(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Open a pass and attach its gradient completion hook to the whole MoE layer input."""
        self._runtime_init(hidden_states)
        self._temporaries = _PassTemporaries()
        return _VirtualExpertWaitGradReduce.apply(
            hidden_states, *self.source_parameters, weakref.ref(self)
        )

    # Called from the dispatcher's jit-fused preprocessing: the planner must launch eagerly
    # (Inductor re-emits user Triton kernels and drops their device asserts), so dynamo breaks
    # the graph around this call instead of tracing into it.
    @torch.compiler.disable
    @nvtx_decorator(message="virtual_expert_plan")
    def plan_dispatch(self, top_indices: torch.Tensor, probs: torch.Tensor) -> None:
        """Plan the router's ``[num_tokens, topk]`` routes and begin the weight push, before
        shared-expert compute."""
        self._temporaries.plan, self._temporaries.runtime_probs = plan_virtual_expert_routes(
            top_indices, probs, self.workspace.planner
        )
        self._start_weight_push(FORWARD)

    def prepare_virtual_expert_dispatch(
        self, hidden_states: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Attach dispatch-backward work and return what the transport dispatches: the hidden
        states, the int16 ``[num_tokens, topk]`` runtime expert ids and the dense runtime
        probabilities, which carry the gradient."""
        hidden_states = _VirtualExpertBackwardHook.apply(
            hidden_states, _weak_hook(self._start_pending_grad_reduces)
        )
        runtime_probs, self._temporaries.runtime_probs = self._temporaries.runtime_probs, None
        return hidden_states, self._temporaries.plan.virtual_experts, runtime_probs

    def prepare_virtual_expert_combine(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Attach the backward weight-push wait before transport combine."""
        return _VirtualExpertBackwardHook.apply(
            hidden_states, _weak_hook(self._prepare_expert_backward)
        )

    def finalize_output(self, output: torch.Tensor) -> torch.Tensor:
        """Close the forward; the layer output's backward hook hands the pass back for its
        backward and starts the backward weight push."""
        temporaries, self._temporaries = self._temporaries, None
        return _VirtualExpertBackwardHook.apply(
            output, _weak_hook(self._start_backward, temporaries)
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

    # ---- weight push -------------------------------------------------------------------------

    @torch.no_grad()
    @nvtx_decorator(message="virtual_expert_weight_push_start")
    def _start_weight_push(self, direction: int) -> None:
        """Enqueue the owner push of the pass's FC1/FC2 weights on the weight stream.

        Only reads the weights: GTP consumes them, and issues its next prefetch, at the expert
        GEMMs, exactly where it would without virtual experts.
        """
        temporaries = self._temporaries
        if temporaries.push_in_flight:
            raise RuntimeError("Virtual-expert weight prefetch is already outstanding.")
        for fc_layer in self._fc_layers(direction):
            fc_layer.prepare(direction)
        workspace = self.workspace
        current_stream = torch.cuda.current_stream(self.device)
        weight_stream = workspace.weight_stream(current_stream)
        weight_stream.wait_stream(current_stream)
        tables = tuple(fc_layer.tables[direction] for fc_layer in self.fc_layers)
        with torch.cuda.stream(weight_stream):
            launch_virtual_expert_weight_prefetch(
                workspace,
                sources=tuple(table[0] for table in tables),
                scale_sources=tuple(table[1] for table in tables) if workspace.mxfp8 else None,
                experts_to_copy=temporaries.plan.experts_to_copy,
            )
            self.prefetch_done.record(weight_stream)
        temporaries.push_in_flight, temporaries.push_done = True, False

    @torch.no_grad()
    @nvtx_decorator(message="virtual_expert_weight_push_wait")
    def _wait_weight_push(self) -> None:
        """Make the current stream wait for the pass's push; waiting again is a no-op."""
        temporaries = self._temporaries
        if temporaries.push_in_flight:
            torch.cuda.current_stream(self.device).wait_event(self.prefetch_done)
            temporaries.push_in_flight, temporaries.push_done = False, True
        elif not temporaries.push_done:
            raise RuntimeError("Virtual-expert weights require a started prefetch before use.")

    @torch.no_grad()
    def prepare_expert_forward(self) -> None:
        """Right before the expert forward GEMMs: wait for the forward push and consume the
        weights in GTP's prefetch chain (a no-op for FC layers without GTP)."""
        self._wait_weight_push()
        for fc_layer in self._fc_layers(FORWARD):
            fc_layer.consume(FORWARD)

    @torch.no_grad()
    def _start_backward(self, temporaries: _PassTemporaries) -> None:
        """At the layer output's backward: take the pass back and start its backward push."""
        if self._temporaries is not None:
            raise RuntimeError("Virtual-expert backward started while another pass is outstanding.")
        self._temporaries = temporaries
        self._start_weight_push(BACKWARD)

    @torch.no_grad()
    def _prepare_expert_backward(self) -> None:
        """Right before the expert backward: wait for the backward push, consume the weights in
        GTP's chain and acquire the natives' wgrad scratch, FC2 first."""
        self._wait_weight_push()
        scratch = [None, None]
        for fc_layer in self._fc_layers(BACKWARD):
            fc_layer.consume(BACKWARD)
            scratch[fc_layer.index] = fc_layer.acquire_wgrad_scratch()
        self._temporaries.scratch = tuple(scratch)

    # ---- gradient reduction ------------------------------------------------------------------

    @torch.no_grad()
    @nvtx_decorator(message="virtual_expert_grad_reduce_start")
    def _start_grad_reduce(self, fc_layer: int) -> None:
        """Enqueue the virtual-expert-gradient reduction of one FC layer (0 = FC1, 1 = FC2)
        on the reduction stream, behind everything the compute stream has issued."""
        temporaries = self._temporaries
        if temporaries is None or temporaries.scratch is None:
            raise RuntimeError("Virtual-expert gradient reduction needs the backward pass.")
        if fc_layer in temporaries.started:
            raise RuntimeError(
                f"Virtual-expert gradient reduction of FC{fc_layer + 1} started twice."
            )
        workspace = self.workspace
        workspace.grad_stream.wait_stream(torch.cuda.current_stream(self.device))
        with torch.cuda.stream(workspace.grad_stream):
            launch_virtual_expert_grad_reduce(
                workspace,
                native_grads=tuple(p.native_grad_bases for p in self.fc_layers),
                experts_to_copy=temporaries.plan.experts_to_copy,
                fc_layers=(fc_layer,),
            )
            # Both reductions run in order on the reduction stream, FC2 first, so the event
            # recorded after the last launch covers both.
            self.grad_reduce_done.record(workspace.grad_stream)
        temporaries.started.add(fc_layer)

    def start_fc2_grad_reduce(self) -> None:
        """Start FC2's reduction from the expert backward, right behind FC2's wgrad GEMM."""
        self._start_grad_reduce(1)

    def _start_pending_grad_reduces(self) -> None:
        """Start every reduction not yet started, FC2 first, once dispatch backward is done.

        FC2 normally starts from the FC2 op's wgrad store; FC1 starts here and hides behind
        the latent, shared-expert and router backward.
        """
        for fc_layer in (1, 0):
            if fc_layer not in self._temporaries.started:
                self._start_grad_reduce(fc_layer)

    @torch.no_grad()
    @nvtx_decorator(message="virtual_expert_grad_reduce_wait")
    def _finish_grad_reduce(self) -> tuple[torch.Tensor | None, ...]:
        """Close the pass: wait for both reductions and return one wgrad per source parameter,
        the full staging for plain parameters and what GTP's protocol returns for GTP parameters.

        GTP parameters reduce-scatter their scratch here, FC2 first as their linked chain
        expects, through the protocol call TE issues right after a wgrad GEMM
        (``finalize_group_grads``). It comes later than TE's because the natives' gradient is
        complete only after the cross-rank reduction, and it stays on the compute stream because
        GTP's finalize also adds the shard gradients into ``main_grad`` and fires DDP's grad-ready
        hooks, which DDP orders against the compute stream (a bucket shared with compute-stream
        parameters scales and copies the grad buffer from whichever hook completes it). This is
        the earliest compute-stream point after both reductions: FC2's started at its wgrad GEMM
        and FC1's at the dispatch backward, so the waits are normally free.
        """
        temporaries, self._temporaries = self._temporaries, None
        if temporaries is None or temporaries.started != {0, 1}:
            raise RuntimeError(
                "Virtual-expert gradient reduction of both fc_layers must be started."
            )
        torch.cuda.current_stream(self.device).wait_event(self.grad_reduce_done)
        grads = [None, None]
        for fc_layer in self._fc_layers(BACKWARD):
            scratch = temporaries.scratch[fc_layer.index]
            if scratch is None:
                grads[fc_layer.index] = tuple(fc_layer.native_grad)
            else:
                reduced = fc_layer.gtp_leader.finalize_group_grads(list(scratch))
                grads[fc_layer.index] = (
                    tuple(reduced) if isinstance(reduced, (list, tuple)) else (reduced,)
                )
                fc_layer.release_wgrad_scratch()
        return tuple(grad for fc_layer in grads for grad in fc_layer)
