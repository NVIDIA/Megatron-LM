# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.


"""Virtual-expert load balancing, virtual-expert planning, and runtime weight movement.
The load-balancing algorithm implemented here is taken from MoonEP
(https://github.com/moonshotAI/moonep): perfectly balanced expert parallelism through
redundant experts planned online from the router output.
"""

from __future__ import annotations

import gc
import math
import weakref
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum
from functools import partial
from typing import TYPE_CHECKING, Literal, NamedTuple

import torch
import torch.distributed as dist

from megatron.core.fp8_utils import is_mxfp8tensor
from megatron.core.transformer.moe.moe_utils import get_align_size_for_quantization

try:
    from megatron.core.transformer.moe.virtual_expert_triton import (
        MAX_VIRTUAL_EXPERT_WEIGHT_SMS,
        VirtualExpertPlannerWorkspace,
        launch_virtual_expert_grad_reduce,
        launch_virtual_expert_planner,
        launch_virtual_expert_weight_prefetch,
    )

    _TRITON_AVAILABLE = True
except ImportError:
    _TRITON_AVAILABLE = False
    MAX_VIRTUAL_EXPERT_WEIGHT_SMS = VirtualExpertPlannerWorkspace = None
    launch_virtual_expert_planner = launch_virtual_expert_weight_prefetch = None
    launch_virtual_expert_grad_reduce = None
from megatron.core.utils import ensure_params_ready, nvtx_decorator

if TYPE_CHECKING:
    from megatron.core.transformer.transformer_config import TransformerConfig


class WeightDirection(Enum):
    """Weight orientation used by the push, GTP consume and pointer table."""

    FORWARD = "forward"
    BACKWARD = "backward"


_MXFP8_COMPONENTS = (
    "_rowwise_data",
    "_rowwise_scale_inv",
    "_columnwise_data",
    "_columnwise_scale_inv",
)


def _is_gtp(parameter) -> bool:
    """Whether ``parameter`` is a GTP-sharded expert weight, gathered per forward."""
    return getattr(parameter, "is_gtp_weight_remat", False)


# ---- Planning --------------------------------------------------------------------------------


@dataclass(slots=True)
class VirtualExpertPlan:
    """One forward's routes and in-flight state, retained until its backward finishes.

    ``virtual_experts`` holds int16 [tokens, topk] runtime ids; ``experts_to_copy`` holds
    int32 [EP, local_experts] semantic ids (-1 for unused slots). Repeated MTP forwards
    retain separate plans through their layer-output hooks. ``ready`` joins the planner stream."""

    virtual_experts: torch.Tensor
    experts_to_copy: torch.Tensor
    # The plan's weight push was started and not yet waited for (one wait per push).
    push_in_flight: bool = False
    # The FC layers whose reduction was launched in the backward.
    started: set = field(default_factory=set)
    ready: torch.cuda.Event | None = None


def plan_virtual_expert_routes(top_indices: torch.Tensor, workspace) -> VirtualExpertPlan:
    """Plan deterministic virtual-expert placement for one EP group and map this rank's routes.

    ``top_indices`` are the router's ``[num_tokens, topk]`` expert ids; every rank must route the
    same number of tokens. The histograms are the only cross-rank input and the planner kernel
    exchanges them itself, so every rank computes the same placement.
    """
    if top_indices.dim() != 2 or top_indices.dtype not in (torch.int32, torch.int64):
        raise ValueError("Virtual-expert planner takes int32/int64 [num_tokens, topk] expert ids.")
    return VirtualExpertPlan(*launch_virtual_expert_planner(top_indices.contiguous(), workspace))


# ---- Virtual-expert slots: arenas and runtime parameters shared by every layer ---------------


@dataclass(frozen=True, slots=True)
class _VirtualExpertConfig:
    """The expert storage layout used to share slots between compatible layers."""

    group_name: str
    device: torch.device
    ep_size: int
    num_local_experts: int
    member_shapes: tuple  # (out_features, in_features) of FC1 and FC2
    mxfp8: bool
    grad_dtype: torch.dtype
    num_sms: int
    gtp: tuple[bool, ...] = (False, False)


class _VirtualExpertStorage:
    """Transport arenas and slots shared by layers with the same expert storage layout.

    MXFP8 main layers and BF16 MTP layers retain separate arenas. A layout's storage is reused
    across forwards; its NCCL registrations are released before the process group is destroyed.
    """

    def __init__(self, group, config: _VirtualExpertConfig, templates) -> None:
        self.config = config
        self.weight_arena = self.grad_arena = self.weight_handle = self.grad_handle = None
        self.slot_weights = ()
        try:
            self._allocate(group, config, templates)
        except Exception:
            self.destroy()
            raise

    def _allocate(self, group, config: _VirtualExpertConfig, templates) -> None:
        import torch.distributed._symmetric_memory as symm_mem

        device, count, mxfp8 = config.device, config.num_local_experts, config.mxfp8
        # What the kernels read.
        self.rank = dist.get_rank(group=group)
        self.world_size = config.ep_size
        self.num_local_experts = count
        self.member_numels = tuple(math.prod(shape) for shape in config.member_shapes)
        self.num_sms = config.num_sms
        # One E8M0 scale byte per 32 MXFP8 weight bytes, unpadded (the config requires 128-aligned
        # FC layers so TE's padded scale layout has this exact size).
        scale_numels = tuple(numel // 32 if mxfp8 else 0 for numel in self.member_numels)
        self._weight_sections = [
            count * n for pair in zip(self.member_numels, scale_numels) for n in pair
        ]
        self._grad_sections = [count * numel for numel in self.member_numels]
        try:
            if symm_mem.get_backend(device) != "NCCL":
                symm_mem.set_backend("NCCL")
            self.weight_arena = symm_mem.empty(
                sum(self._weight_sections),
                dtype=torch.uint8 if mxfp8 else torch.bfloat16,
                device=device,
            )
            self.weight_handle = symm_mem.rendezvous(self.weight_arena, group)
            self.grad_arena = symm_mem.empty(
                sum(self._grad_sections), dtype=config.grad_dtype, device=device
            )
            self.grad_handle = symm_mem.rendezvous(self.grad_arena, group)
        except RuntimeError as exc:
            raise RuntimeError(
                "Virtual-expert weights could not allocate NCCL symmetric memory for the EP group; "
                "the EP group must lie within one NVLink domain."
            ) from exc
        # The reduction reaches every peer's gradient slots through one TMA descriptor whose
        # outermost stride is the distance between consecutive peers' windows, so the allocator's
        # uniform mapping is a hard requirement: verify it once, here.
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
        self.slot_weights = tuple(
            self._slot_parameters(i, template) for i, template in enumerate(templates)
        )

    @staticmethod
    def _wrap_mxfp8(template, shape, views, device) -> tuple[torch.Tensor, ...]:
        """Wrap ``(rowwise, rowwise_scale, columnwise, columnwise_scale)`` views as MXFP8 tensors
        with ``template``'s quantization metadata."""
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

    @staticmethod
    def _runtime_parameter(
        weight: torch.Tensor, main_grad: torch.Tensor, *, overwrite: bool = True
    ) -> torch.nn.Parameter:
        """A runtime leaf that writes fused wgrads into ``main_grad`` without DDP hooks."""
        parameter = torch.nn.Parameter(weight)
        parameter.main_grad = main_grad
        parameter.overwrite_main_grad = overwrite
        # These leaves have no DDP hooks. Without grad_added_to_main_grad, TE returns None
        # after writing the fused wgrad, avoiding dummy gradients and AccumulateGrad work
        # whose result would otherwise be immediately discarded.
        return parameter

    def _slot_parameters(self, fc_layer: int, template) -> tuple[torch.nn.Parameter, ...]:
        """One runtime parameter per slot of ``fc_layer`` over its arena sections (MXFP8-wrapped
        with ``template``'s quantization metadata), its ``main_grad`` the matching gradient slot.
        """
        count, shape = self.num_local_experts, self.config.member_shapes[fc_layer]
        weights = self.weight_arena.split(self._weight_sections)
        data = weights[2 * fc_layer].view(count, *shape)
        grads = self.grad_arena.split(self._grad_sections)[fc_layer].view(count, *shape)
        if not self.config.mxfp8:
            return tuple(self._runtime_parameter(w, g) for w, g in zip(data, grads))
        scales = weights[2 * fc_layer + 1].view(count, -1)
        if _is_gtp(template):
            rowwise, columnwise = _GTPTEWeightBridge._gtp_scale_shapes(template, shape)
        else:
            rowwise, columnwise = (
                template._rowwise_scale_inv.shape,
                template._columnwise_scale_inv.shape,
            )
        views = tuple(
            (data[i], scales[i].view(rowwise), data[i], scales[i].view(columnwise))
            for i in range(count)
        )
        # set_() empties the components at teardown but would leave a view's arena _base alive.
        views = tuple(tuple(view.detach() for view in slot) for slot in views)
        weights = self._wrap_mxfp8(template, shape, views, data.device)
        return tuple(self._runtime_parameter(w, g) for w, g in zip(weights, grads))

    def clear_accumulating_grads(self) -> None:
        """Zero virtual partials for FC layers that accumulate directly into DDP buffers.

        TE uses the first native's accumulate flag for the whole grouped GEMM, but virtual
        slots have no optimizer history. Call on compute before the next expert backward,
        after the previous layer's input backward has finished all reads of these slots.
        """
        gtp = self.config.gtp
        if not any(gtp):
            self.grad_arena.zero_()
        elif not all(gtp):
            for sharded, section in zip(gtp, self.grad_arena.split(self._grad_sections)):
                if not sharded:
                    section.zero_()

    def destroy(self) -> None:
        """Release shared slots and NCCL registrations before destroying the process group.

        Empty the slot parameters in place so references held by layers and TE release the arenas.
        """
        if self.weight_arena is None and self.grad_arena is None:
            return
        torch.cuda.synchronize(self.config.device)
        with torch.no_grad():
            for slots in self.slot_weights:
                for slot in slots:
                    slot.main_grad = None
                    if self.config.mxfp8:
                        for name in _MXFP8_COMPONENTS:
                            getattr(slot, name).set_()
                    else:
                        slot.set_()
        self.weight_handle = self.grad_handle = self.weight_arena = self.grad_arena = None
        self.slot_weights = ()


class _PointerTable(NamedTuple):
    """Device addresses for a kernel and their host snapshot for stability checks."""

    tensor: torch.Tensor
    pointers: list[list[int]]
    row_views: tuple[torch.Tensor, ...]


class _GTPTEWeightBridge:
    """TE DistributedWeight protocol bound to BF16/MXFP8 runtime parameters.

    TE materializes weights and acquires wgrad buffers. The layer-input hook finalizes
    GTP only after remote virtual-expert gradients arrive."""

    def _gtp_runtime_shells(self, fc_layer: int):
        """Create native weight shells and empty gradients until GTP supplies their buffers."""
        config, storage = self.config, self.storage
        weights = self.parameters[fc_layer]
        if config.mxfp8:
            # Borrow slot storage until the weight push binds the gathered native weights.
            weights = storage._wrap_mxfp8(
                weights[0],
                config.member_shapes[fc_layer],
                tuple(
                    tuple(getattr(slot, name) for name in _MXFP8_COMPONENTS)
                    for slot in storage.slot_weights[fc_layer]
                ),
                config.device,
            )
        else:
            weights = (torch.empty(0, dtype=torch.bfloat16, device=config.device),) * len(weights)
        return weights, (self.placeholder,) * len(weights)

    @staticmethod
    def _gtp_scale_shapes(parameter, shape):
        """Get the gathered MXFP8 scale shapes from GTP's quantizer."""
        return tuple(
            parameter._gtp_gather_quantizer.get_scale_shape(shape, columnwise=c)
            for c in (False, True)
        )

    def _peek_gtp_weights(self, fc_layer: int, direction: WeightDirection):
        """Peek gathered weights and bind runtime shells without advancing GTP's prefetch chain."""
        leader = self.gtp_leaders[fc_layer]
        sources = _as_tuple(
            leader.peek_group_for_forward()
            if direction == WeightDirection.FORWARD
            else leader.peek_group_for_backward()
        )
        # bind the ready GTP buffers into the virtual expert weights
        for parameter, source in zip(self.native_weights[fc_layer], sources):
            if self.config.mxfp8:
                for name in self._components(direction):
                    setattr(parameter, name, getattr(source, name))
            else:
                parameter.data = source
        return sources

    def _bind_gtp_te_weights(self, fc_layer: int) -> None:
        """Bind TE's DistributedWeight API to the runtime weights for one FC layer."""
        weights = self.runtime_weights[fc_layer]
        num_native = len(self.parameters[fc_layer])
        # Bind unbound methods to a weak proxy: TE-held weights must not retain this layer.
        owner = weakref.proxy(self)
        leader = weights[0]
        leader.is_distributed_weight = True
        # Older TE fused GroupedMLP also checks these grouped-weight attributes.
        leader.is_routed_expert = True
        leader.weight_list = tuple(weakref.proxy(w) for w in weights)
        leader.materialize_group_for_forward = partial(
            type(self).materialize, owner, fc_layer, WeightDirection.FORWARD
        )
        leader.materialize_group_for_backward = partial(
            type(self).materialize, owner, fc_layer, WeightDirection.BACKWARD
        )
        leader.finalize_group_grads = partial(type(self)._on_te_wgrad_ready, owner, fc_layer)
        for index, weight in enumerate(weights):
            weight.grad_buffer = (
                partial(type(self)._gtp_grad_buffer, owner, fc_layer, index)
                if index < num_native
                else partial(getattr, weakref.proxy(weight), "main_grad")
            )

    def materialize(self, fc_layer: int, direction: WeightDirection):
        """TE API: consume GTP natives for forward/backward and append ready virtual slots."""
        leader = self.gtp_leaders[fc_layer]
        initializing = direction == WeightDirection.FORWARD and not leader.prefetch_initialized
        sources = _as_tuple(
            leader.materialize_group_for_forward()
            if direction == WeightDirection.FORWARD
            else leader.materialize_group_for_backward()
        )
        # The push already checked the runtime shells. TE computes with these returned sources,
        # so only verify that GTP handed out the same buffers the push read.
        if self._ptrs(direction, sources) != self._tables[fc_layer][direction].pointers:
            raise RuntimeError(
                f"FC{fc_layer + 1} {direction}: GTP source storage moved between push and materialization."
            )
        # First consume can replace its forward ticket while discovering the chain. The
        # next push binds the established allocation without constraining GTP's cache setup.
        if initializing:
            self._tables[fc_layer].pop(WeightDirection.FORWARD)
        return (*sources, *self.storage.slot_weights[fc_layer])

    def _gtp_grad_buffer(self, fc_layer: int, index: int) -> torch.Tensor:
        """TE API: ``grad_buffer()``. Acquire persistent native wgrad buffers on demand."""
        if self.native_grads[fc_layer] is None:
            self._bind_gtp_grads(
                fc_layer,
                tuple(
                    w.get_wgrad_tensor(persistent=True) for w in self.gtp_leaders[fc_layer]._weights
                ),
            )
        return self.native_grads[fc_layer][index]

    def _on_te_wgrad_ready(self, fc_layer: int, wgrads):
        """TE API: finalize_group_grads starts FC2's VE reduction; GTP finalizes after it."""
        expected = [w.main_grad for w in self.runtime_weights[fc_layer]]
        if self._ptrs("grad", wgrads) != self._ptrs("grad", expected):
            raise RuntimeError(
                "TE must write virtual-expert wgrads into DistributedWeight buffers."
            )
        if fc_layer == 1 and self._start_grad_reduce is not None:
            self._start_grad_reduce()(fc_layer)
        return (None,) * len(expected)

    def _bind_gtp_grads(self, fc_layer: int, grads) -> None:
        """Bind GTP's wgrad buffers before the GEMM, or park natives on empty placeholders.

        Check an existing pointer table without creating one; the first reduction creates it.
        """
        if grads is not None:
            if self.native_grads[fc_layer] is not None:
                raise RuntimeError(f"FC{fc_layer + 1}: native wgrads bound twice in one backward.")
            if "grad" in self._tables[fc_layer]:
                self.get_weight_table(fc_layer, "grad", grads)
        targets = (
            grads if grads is not None else (self.placeholder,) * len(self.parameters[fc_layer])
        )
        self.native_grads[fc_layer] = grads
        for parameter, grad in zip(self.native_weights[fc_layer], targets):
            parameter.main_grad = grad

    def _reduce_scatter_gtp_grads(self, fc_layer: int) -> tuple[torch.Tensor, ...]:
        """Finalize GTP after remote VE gradients arrive, then release the native bindings."""
        if self.native_grads[fc_layer] is None:
            raise RuntimeError(f"FC{fc_layer + 1}: no GTP wgrad scratch bound for this backward.")
        reduced = self.gtp_leaders[fc_layer].finalize_group_grads(list(self.native_grads[fc_layer]))
        self._bind_gtp_grads(fc_layer, None)
        return _as_tuple(reduced)


class _VirtualExperts(_GTPTEWeightBridge):
    """One layer's runtime weights, GTP bindings and pointer tables over shared storage."""

    def __init__(self, storage: _VirtualExpertStorage, parameters, start_grad_reduce=None) -> None:
        self.storage = storage
        self.config = config = storage.config
        self.parameters = parameters
        self.source_parameters = tuple(parameter for group in parameters for parameter in group)
        self._start_grad_reduce = (
            weakref.WeakMethod(start_grad_reduce) if start_grad_reduce is not None else None
        )
        self.gtp_leaders = tuple(p[0] if _is_gtp(p[0]) else None for p in parameters)
        self.native_grads = [None] * len(parameters)
        self.placeholder = torch.empty(0, dtype=config.grad_dtype, device=config.device)
        self.runtime_weights = []
        self.native_weights = []
        for i, weights in enumerate(parameters):
            slots = storage.slot_weights[i]
            sharded = self.gtp_leaders[i] is not None
            if sharded:
                weights, grads = self._gtp_runtime_shells(i)
            else:
                grads = tuple(p.main_grad for p in weights)
                self.native_grads[i] = grads
            natives = tuple(
                storage._runtime_parameter(w, g, overwrite=sharded) for w, g in zip(weights, grads)
            )
            self.native_weights.append(natives)
            self.runtime_weights.append((*natives, *slots))
            if sharded:
                self._bind_gtp_te_weights(i)
        self._tables: list[dict[WeightDirection | Literal["grad"], _PointerTable]] = [
            {} for _ in parameters
        ]

    def _components(self, key: WeightDirection | Literal["grad"]) -> tuple[str, ...]:
        if self.config.mxfp8 and key != "grad":
            offset = 2 if key == WeightDirection.BACKWARD else 0
            return _MXFP8_COMPONENTS[offset : offset + 2]
        return ("data",)

    def _ptrs(self, key: WeightDirection | Literal["grad"], sources) -> list[list[int]]:
        return [
            [getattr(source, name).data_ptr() for source in sources]
            for name in self._components(key)
        ]

    def get_weight_table(
        self, fc_layer: int, key: WeightDirection | Literal["grad"], sources
    ) -> torch.Tensor:
        """Create the ``forward``, ``backward`` or ``grad`` table once, then check stability."""
        rows = self._ptrs(key, sources)
        table = self._tables[fc_layer].get(key)
        natives = self.native_weights[fc_layer]
        if (table is not None and rows != table.pointers) or (
            key != "grad" and self._ptrs(key, natives) != rows
        ):
            raise RuntimeError(
                f"FC{fc_layer + 1} {key}: source or runtime storage moved after binding; "
                "storage must be static."
            )
        if table is None:
            tensor = torch.tensor(rows, dtype=torch.int64, device=self.config.device)
            table = _PointerTable(tensor, rows, tuple(tensor.unbind()))
            self._tables[fc_layer][key] = table
        return table.tensor

    def grad_table(self, fc_layer: int) -> torch.Tensor:
        """Get only this FC's targets; the other FC may not have acquired its wgrad buffers yet."""
        if "grad" not in self._tables[fc_layer]:
            self.get_weight_table(fc_layer, "grad", self.native_grads[fc_layer])
        return self._tables[fc_layer]["grad"].row_views[0]

    def weight_tables(self, direction: WeightDirection) -> tuple[torch.Tensor, ...]:
        """Publish DDP parameters or peek GTP gathers before the push, then check their tables.

        GTP consumption and its next prefetch remain at the expert GEMMs.
        """
        tables = []
        for i, sources in enumerate(self.parameters):
            if self.gtp_leaders[i] is not None:
                sources = self._peek_gtp_weights(i, direction)
            elif direction == WeightDirection.FORWARD:
                ensure_params_ready(sources)
            tables.append(self.get_weight_table(i, direction, sources))
        return tuple(tables)

    def prepare_backward(self) -> None:
        """Prepare accumulating DDP buffers; TE acquires GTP buffers through grad_buffer()."""
        if not all(self.config.gtp):
            self.storage.clear_accumulating_grads()
        for i, sharded in enumerate(self.config.gtp):
            if not sharded:
                self.get_weight_table(i, "grad", tuple(p.main_grad for p in self.parameters[i]))

    def hand_off_wgrads(self, fc_layer: int) -> tuple[torch.Tensor, ...]:
        """Finish GTP reduction or return dummy grads to trigger DDP's grad-ready hooks.

        Non-GTP weights already accumulated into main_grad; do not add their gradients again."""
        from transformer_engine.pytorch.module.base import get_dummy_wgrad

        if self.gtp_leaders[fc_layer] is not None:
            return self._reduce_scatter_gtp_grads(fc_layer)

        parameters = self.parameters[fc_layer]
        grads = []
        for parameter in parameters:
            parameter.grad_added_to_main_grad = True
            grads.append(
                get_dummy_wgrad(
                    list(parameter.shape),
                    parameter.dtype,
                    zero=getattr(parameter, "zero_out_wgrad", False),
                )
            )
        return tuple(grads)


# ---- Load balancer ---------------------------------------------------------------------------


class _VirtualExpertHook(torch.autograd.Function):
    """Call a method during backward, preserving the tensor gradient.

    Its return values are gradients for ``inputs``. Hold the method weakly: the manager
    retains forward tensors, so a strong reference would keep the autograd graph alive."""

    @staticmethod
    def forward(ctx, tensor, method, args, *inputs):
        ctx.method, ctx.args = weakref.WeakMethod(method), args
        return tensor

    @staticmethod
    def backward(ctx, grad):
        return (grad, None, None, *(ctx.method()(*ctx.args) or ()))


def _as_tuple(value) -> tuple:
    return tuple(value) if isinstance(value, (list, tuple)) else (value,)


class VirtualExpertLoadBalancer:
    """Balance HybridEP routes with virtual experts using the MoonEP algorithm.

    Every rank receives tokens * topk routes. Each rank copies experts from one owner into
    num_local_experts slots and reduces their gradients back during backward. HybridEP owns
    token transport; this mixin schedules planning, weight movement, and gradient reduction."""

    # Shared by every layer of the process, allocated by the first layer to bind. Two candidate
    # weight streams: a CUDA-graph capture stream comes from the same pool and may alias one.
    planner: VirtualExpertPlannerWorkspace | None = None
    planner_stream: torch.cuda.Stream | None = None
    storages: dict[_VirtualExpertConfig, _VirtualExpertStorage] = {}
    weight_streams: tuple[torch.cuda.Stream, torch.cuda.Stream] | None = None
    grad_stream: torch.cuda.Stream | None = None

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
        # Config bounds are validated at construction; check the actual topology before CUDA.
        if not (
            self.ep_size == config.expert_model_parallel_size
            and num_experts == config.num_moe_experts
            and num_experts == self.ep_size * num_local_experts
            and router_topk == config.moe_router_topk
        ):
            raise ValueError(
                "Virtual-expert runtime layout must match TransformerConfig; got "
                f"EP={self.ep_size}, experts={num_experts}, local={num_local_experts}, "
                f"topk={router_topk}."
            )
        self.group = group
        self.config = config
        self.router_topk = router_topk
        self.num_owned_experts = num_local_experts
        # HybridEP pads every runtime expert's segment to the quantization block.
        self._alignment = get_align_size_for_quantization(config)
        # The layer's token count and this rank's transport capacity for it, sized at the first
        # forward.
        self.num_tokens: int | None = None
        self.rank_capacity: int | None = None
        self._experts_ref = None
        # The plan of the pass this manager currently owns (see VirtualExpertPlan).
        self._plan: VirtualExpertPlan | None = None
        self.device = torch.device("cuda", torch.cuda.current_device())
        # CUDA events are created lazily on first record; materialize them before training or
        # graph capture.
        self.prefetch_done = torch.cuda.Event()
        self.grad_reduce_done = torch.cuda.Event()
        self.planner_done = torch.cuda.Event()
        for event in (self.prefetch_done, self.grad_reduce_done, self.planner_done):
            event.record(torch.cuda.current_stream(self.device))
        # Bound at the first forward, once DDP has built the main gradients whose dtype the arenas
        # take.
        self.virtual_experts: _VirtualExperts | None = None

    # ---- binding -----------------------------------------------------------------------------

    def bind_experts(self, experts: torch.nn.Module) -> None:
        """Remember the expert module; runtime slots are bound after main-grad initialization."""
        self._experts_ref = weakref.ref(experts)

    def _runtime_init(self, hidden_states: torch.Tensor) -> None:
        """Check the previous pass unwound and the layer's token count is unchanged.

        Size transport capacity and bind runtime parameters over the shared slots once.
        """
        if self._plan is not None:
            raise RuntimeError(
                "Virtual-expert layer input wrapped while the previous pass is still in flight: "
                "its forward was not finalized or its backward did not finish."
            )
        num_tokens = hidden_states.numel() // hidden_states.shape[-1]
        if self.virtual_experts is not None:
            if num_tokens != self.num_tokens:
                raise ValueError(
                    "Virtual-expert layers require a fixed token count; "
                    f"expected {self.num_tokens}, got {num_tokens}."
                )
            return

        self.num_tokens = num_tokens
        self.rank_capacity = self._compute_rank_capacity(num_tokens)

        experts = self._experts_ref() if self._experts_ref is not None else None
        if experts is None:
            raise RuntimeError("Virtual-expert load balancer has no bound expert module.")

        linears = (experts.linear_fc1, experts.linear_fc2)
        parameters = tuple(
            tuple(linear.get_parameter(f"weight{i}") for i in range(self.num_owned_experts))
            for linear in linears
        )
        grad_dtypes = {parameter.main_grad.dtype for group in parameters for parameter in group}
        if len(grad_dtypes) != 1:
            raise RuntimeError(
                "Virtual-expert source main gradients must be initialized with one dtype for every "
                f"expert weight, got {grad_dtypes}."
            )
        grad_dtype = grad_dtypes.pop()
        num_sms = self.config.moe_flex_dispatcher_num_sms
        config = _VirtualExpertConfig(
            group_name=self.group.group_name,
            device=self.device,
            ep_size=self.ep_size,
            num_local_experts=self.num_owned_experts,
            member_shapes=tuple(
                (int(linear.out_features), int(linear.in_features)) for linear in linears
            ),
            mxfp8=is_mxfp8tensor(parameters[0][0]),
            grad_dtype=grad_dtype,
            num_sms=min(32 if num_sms is None else int(num_sms), MAX_VIRTUAL_EXPERT_WEIGHT_SMS),
            gtp=tuple(_is_gtp(p[0]) for p in parameters),
        )
        cls = VirtualExpertLoadBalancer
        if cls.storages:
            bound = next(iter(cls.storages))
            if (bound.group_name, bound.device, bound.ep_size, bound.num_local_experts) != (
                config.group_name,
                config.device,
                config.ep_size,
                config.num_local_experts,
            ):
                raise ValueError(
                    "Virtual-expert layers must share one expert topology. Call "
                    "VirtualExpertLoadBalancer.finalize() before rebuilding process groups."
                )
        if cls.planner is None:
            # The planner's allocation runs the first collective on the group, which creates the
            # device communicator the slots' NCCL window registrations need.
            cls.planner = VirtualExpertPlannerWorkspace(
                num_experts=self.ep_size * self.num_owned_experts,
                device=self.device,
                group=self.group,
            )
            cls.planner_stream = torch.cuda.Stream(device=self.device)
            cls.weight_streams = (
                torch.cuda.Stream(device=self.device),
                torch.cuda.Stream(device=self.device),
            )
            cls.grad_stream = torch.cuda.Stream(device=self.device)
        storage = cls.storages.get(config)
        if storage is None:
            storage = _VirtualExpertStorage(self.group, config, tuple(p[0] for p in parameters))
            cls.storages[config] = storage
        self.virtual_experts = _VirtualExperts(storage, parameters, self._start_grad_reduce)
        experts.bind_virtual_experts(self)
        # The fused MLP replays FC1's pre-hooks, so this covers both execution paths.
        experts.linear_fc1.register_forward_pre_hook(
            partial(type(self)._wait_weight_push, weakref.proxy(self))
        )

    @classmethod
    def finalize(cls) -> None:
        """Release the shared slots and planner scratch; idempotent. A normal exit needs no call;
        callers that destroy their process groups while a model is still alive (tests, an orderly
        shutdown) call it first so the NCCL windows are deregistered while their communicator
        exists."""
        for storage in cls.storages.values():
            storage.destroy()
        cls.storages.clear()
        if cls.planner is not None:
            cls.planner.destroy()
        cls.planner = cls.planner_stream = cls.weight_streams = cls.grad_stream = None
        # TE's op contexts and autograd graphs hold the runtime parameters in reference cycles;
        # collect now so anything still viewing the arenas is freed while the group is alive.
        gc.collect()

    @property
    def num_runtime_experts(self) -> int:
        """Natives plus virtual-expert slots per rank."""
        return 2 * self.num_owned_experts

    def runtime_weights(self, fc_layer: int) -> tuple[torch.nn.Parameter, ...]:
        """Native-then-virtual-expert runtime parameters of one FC layer (0 = FC1, 1 = FC2)."""
        return self.virtual_experts.runtime_weights[fc_layer]

    @property
    def source_parameters(self) -> tuple[torch.nn.Parameter, ...]:
        """The optimizer-owned FC1 then FC2 parameters."""
        return self.virtual_experts.source_parameters

    @contextmanager
    def _on_side_stream(self, stream, done):
        """Wait on the caller and record completion; a tuple selects a stream distinct from it."""
        current = torch.cuda.current_stream(self.device)
        if isinstance(stream, tuple):
            stream = next(s for s in stream if s.cuda_stream != current.cuda_stream)
        stream.wait_stream(current)
        with torch.cuda.stream(stream):
            yield stream
            done.record(stream)

    def _compute_rank_capacity(self, num_tokens: int) -> int:
        """A static, dropless route capacity for one transport rank: every rank receives
        exactly its own route count, plus HybridEP's per-runtime-expert segment padding.
        """
        num_routes = num_tokens * self.router_topk
        alignment = self._alignment
        padding = self.num_runtime_experts * max(alignment - 1, 0)
        capacity = max(
            int(num_routes * self.config.moe_expert_rank_capacity_factor), num_routes + padding
        )
        return capacity + (-capacity % alignment if alignment > 1 else 0)

    # ---- forward: dispatcher hooks -----------------------------------------------------------

    def wrap_layer_input(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Initialize the runtime and finalize gradients at the layer input's backward.

        This follows router/dispatch backward and both FC reductions."""
        self._runtime_init(hidden_states)
        return _VirtualExpertHook.apply(
            hidden_states, self._finish_grad_reduce, (), *self.source_parameters
        )

    # Called from the dispatcher's jit-fused preprocessing: the planner must launch eagerly
    # (Inductor re-emits user Triton kernels and drops their device asserts), so dynamo breaks
    # the graph around this call instead of tracing into it.
    @torch.compiler.disable
    @nvtx_decorator(message="virtual_expert_plan")
    def plan_dispatch(self, top_indices: torch.Tensor) -> None:
        """Overlap planning with independent shared-expert or paired-attention compute."""
        with self._on_side_stream(self.planner_stream, self.planner_done):
            top_indices.record_stream(self.planner_stream)
            self._plan = plan_virtual_expert_routes(top_indices, self.planner)
        self._plan.ready = self.planner_done
        self._start_weight_push(WeightDirection.FORWARD)

    def prepare_virtual_expert_dispatch(
        self, hidden_states: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Attach dispatch-backward work and return what the transport dispatches: the hidden
        states and the int16 ``[num_tokens, topk]`` runtime expert ids (the router's
        probabilities are unchanged by the remap)."""
        hidden_states = _VirtualExpertHook.apply(
            hidden_states, self._start_pending_grad_reduces, ()
        )
        if self._plan.ready is not None:
            current = torch.cuda.current_stream(self.device)
            current.wait_event(self._plan.ready)
            self._plan.virtual_experts.record_stream(current)
            self._plan.experts_to_copy.record_stream(current)
        return hidden_states, self._plan.virtual_experts

    def prepare_virtual_expert_combine(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Attach the backward weight-push wait before transport combine."""
        return _VirtualExpertHook.apply(hidden_states, self._prepare_expert_backward, ())

    def finalize_output(self, output: torch.Tensor) -> torch.Tensor:
        """Close the forward; the layer output's backward hook hands the plan back for its
        backward and starts the backward weight push."""
        plan, self._plan = self._plan, None
        return _VirtualExpertHook.apply(output, self._start_backward, (plan,))

    # ---- weight push -------------------------------------------------------------------------

    @torch.no_grad()
    @nvtx_decorator(message="virtual_expert_weight_push_start")
    def _start_weight_push(self, direction: WeightDirection) -> None:
        """Push both FCs' weights; GTP consumes them and advances prefetch at the GEMMs."""
        if self._plan.push_in_flight:
            raise RuntimeError("Virtual-expert weight prefetch is already outstanding.")
        virtual_experts = self.virtual_experts
        tables = virtual_experts.weight_tables(direction)
        with self._on_side_stream(self.weight_streams, self.prefetch_done) as weight_stream:
            if direction == WeightDirection.FORWARD and self._plan.ready is not None:
                weight_stream.wait_event(self._plan.ready)
                self._plan.experts_to_copy.record_stream(weight_stream)
            launch_virtual_expert_weight_prefetch(
                virtual_experts.storage,
                sources=tuple(table[0] for table in tables),
                scale_sources=(
                    tuple(table[1] for table in tables) if virtual_experts.config.mxfp8 else None
                ),
                experts_to_copy=self._plan.experts_to_copy,
            )
        self._plan.push_in_flight = True

    @torch.no_grad()
    @nvtx_decorator(message="virtual_expert_weight_push_wait")
    def _wait_weight_push(self, *_) -> None:
        """Make the current stream wait for the pass's push; exactly once per push."""
        if not self._plan.push_in_flight:
            raise RuntimeError("Virtual-expert weights require a started, unwaited prefetch.")
        torch.cuda.current_stream(self.device).wait_event(self.prefetch_done)
        self._plan.push_in_flight = False

    @torch.no_grad()
    def _start_backward(self, plan: VirtualExpertPlan) -> None:
        """At the layer output's backward: take the plan back and start its backward push."""
        if self._plan is not None:
            raise RuntimeError("Virtual-expert backward started while another pass is outstanding.")
        self._plan = plan
        self._start_weight_push(WeightDirection.BACKWARD)

    @torch.no_grad()
    def _prepare_expert_backward(self) -> None:
        """Wait for the backward push and prepare accumulating gradients before TE's backward."""
        self._wait_weight_push()
        self.virtual_experts.prepare_backward()

    # ---- gradient reduction ------------------------------------------------------------------

    @torch.no_grad()
    @nvtx_decorator(message="virtual_expert_grad_reduce_start")
    def _start_grad_reduce(self, fc_layer: int) -> None:
        """Enqueue the virtual-expert-gradient reduction of one FC layer (0 = FC1, 1 = FC2)
        on the reduction stream, behind everything the compute stream has issued."""
        plan = self._plan
        if plan is None or plan.push_in_flight:
            raise RuntimeError("Virtual-expert gradient reduction needs the prepared backward.")
        if fc_layer in plan.started:
            raise RuntimeError(
                f"Virtual-expert gradient reduction of FC{fc_layer + 1} started twice."
            )
        native_grads = self.virtual_experts.grad_table(fc_layer)
        # Both reductions run in order, FC2 first, so the last completion event covers both.
        with self._on_side_stream(self.grad_stream, self.grad_reduce_done):
            launch_virtual_expert_grad_reduce(
                self.virtual_experts.storage,
                # This tile range only accesses fc_layer's gradients. The other FC may not have
                # reached its wgrad GEMM yet; the same table safely fills the unused pointer slot.
                native_grads=(native_grads, native_grads),
                experts_to_copy=plan.experts_to_copy,
                fc_layers=(fc_layer,),
            )
        plan.started.add(fc_layer)

    def start_fc2_grad_reduce(self) -> None:
        """Start FC2's reduction from the expert backward, right behind FC2's wgrad GEMM."""
        self._start_grad_reduce(1)

    def _start_pending_grad_reduces(self) -> None:
        """Start pending reductions FC2 first. TE normally starts FC2 at its wgrad GEMM;
        FC1 starts here after dispatch, overlapping latent/shared-expert/router backward."""
        for fc_layer in (1, 0):
            if fc_layer not in self._plan.started:
                self._start_grad_reduce(fc_layer)

    @torch.no_grad()
    @nvtx_decorator(message="virtual_expert_grad_reduce_wait")
    def _finish_grad_reduce(self) -> tuple[torch.Tensor, ...]:
        """Wait for VE reductions, finalize GTP FC2 first, and return gradients FC1 first.

        Finalize on the compute stream: GTP adds shard gradients to main_grad and fires
        DDP hooks, which may scale/copy buckets shared with compute-stream parameters."""
        plan, self._plan = self._plan, None
        if plan is None or plan.started != {0, 1}:
            raise RuntimeError(
                "Virtual-expert gradient reduction of both fc_layers must be started."
            )
        torch.cuda.current_stream(self.device).wait_event(self.grad_reduce_done)
        fc2_grads = self.virtual_experts.hand_off_wgrads(1)
        return (*self.virtual_experts.hand_off_wgrads(0), *fc2_grads)
