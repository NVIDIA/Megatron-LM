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
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

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

# Push directions. MXFP8 forward GEMMs read the rowwise components, backward the columnwise.
FORWARD, BACKWARD = 0, 1
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
    """One pass's plan and what it holds in flight, from the planner to the end of its backward.

    ``virtual_experts``: int16 ``[num_tokens, router_topk]`` runtime expert ids, the compact routes
    the transport dispatches; ``experts_to_copy``: int32 ``[ep_size, num_local_experts]`` semantic
    ids per virtual-expert slot, ``-1`` if unused. The manager owns the plan during its forward and
    again during its backward, taking it back at the layer-output hook (a repeated MTP layer has
    several forwards outstanding, so that hook carries the plan). Plain data and scalars only: the
    backward hooks capture the plan inside autograd contexts.
    """

    virtual_experts: torch.Tensor
    experts_to_copy: torch.Tensor
    # The plan's weight push was started and not yet waited for (one wait per push).
    push_in_flight: bool = False
    # The FC layers whose reduction was launched in the backward.
    started: set = field(default_factory=set)


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
    """What sizes the shared slots; every virtual-expert layer of a process must agree on it."""

    group_name: str
    device: torch.device
    ep_size: int
    num_local_experts: int
    member_shapes: tuple  # (out_features, in_features) of FC1 and FC2
    mxfp8: bool
    grad_dtype: torch.dtype
    num_sms: int


class _VirtualExperts:
    """The process-wide virtual-expert slots: ``L`` runtime parameters per FC layer over a NCCL
    symmetric weight arena (``fc1 data, fc1 scales, fc2 data, fc2 scales``; the scale sections are
    empty for BF16, MXFP8 keeps one since one GEMM orientation is live at a time), their
    ``main_grad`` the slots of a symmetric gradient arena (``fc1, fc2``), plus the plain natives'
    wgrad staging the reduction accumulates into. One set serves every layer of the process: one
    layer's virtual experts are live at a time, the planner's histogram exchange ordering each push
    after the GEMMs that read the previous contents and the reduction's exit rendezvous ordering
    each slot rewrite after the owners' reads. The kernels take this object for the arenas, their
    symmetric handles and the grid barriers."""

    def __init__(self, group, config: _VirtualExpertConfig, templates) -> None:
        import torch.distributed._symmetric_memory as symm_mem

        self.config = config
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
        self.parameters = tuple(
            self._slot_parameters(i, template) for i, template in enumerate(templates)
        )
        # Plain natives' wgrad staging per FC layer: TE's GEMM overwrites it, the reduction adds the
        # virtual-expert partials, autograd hands it on. GTP natives write per-backward scratch.
        self.native_staging = tuple(
            None if _is_gtp(t) else torch.empty((count, *s), dtype=config.grad_dtype, device=device)
            for t, s in zip(templates, config.member_shapes)
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
    def _runtime_parameter(weight: torch.Tensor, main_grad: torch.Tensor) -> torch.nn.Parameter:
        """A parameter whose fused wgrad GEMM overwrites ``main_grad`` on every backward."""
        parameter = torch.nn.Parameter(weight)
        parameter.main_grad = main_grad
        parameter.grad_added_to_main_grad = True
        parameter.overwrite_main_grad = True
        # TE returns a dummy leaf grad once the fused wgrad is in main_grad; drop it.
        parameter.register_post_accumulate_grad_hook(lambda p: setattr(p, "grad", None))
        return parameter

    def _slot_parameters(self, fc_layer: int, template) -> tuple[torch.nn.Parameter, ...]:
        """One runtime parameter per slot of ``fc_layer`` over its arena sections (MXFP8-wrapped
        with ``template``'s quantization metadata), its ``main_grad`` the matching gradient slot."""
        count, shape = self.num_local_experts, self.config.member_shapes[fc_layer]
        weights = self.weight_arena.split(self._weight_sections)
        data = weights[2 * fc_layer].view(count, *shape)
        grads = self.grad_arena.split(self._grad_sections)[fc_layer].view(count, *shape)
        if not self.config.mxfp8:
            return tuple(self._runtime_parameter(w, g) for w, g in zip(data, grads))
        scales = weights[2 * fc_layer + 1].view(count, -1)
        if _is_gtp(template):
            rowwise, columnwise = (
                template._gtp_gather_quantizer.get_scale_shape(shape, columnwise=c)
                for c in (False, True)
            )
        else:
            rowwise, columnwise = (
                template._rowwise_scale_inv.shape,
                template._columnwise_scale_inv.shape,
            )
        views = tuple(
            (data[i], scales[i].view(rowwise), data[i], scales[i].view(columnwise))
            for i in range(count)
        )
        weights = self._wrap_mxfp8(template, shape, views, data.device)
        return tuple(self._runtime_parameter(w, g) for w, g in zip(weights, grads))

    def runtime_parameters(self, fc_layer: int, parameters) -> tuple[torch.nn.Parameter, ...]:
        """The ``2L`` runtime parameters TE runs over for one layer's ``fc_layer``: its natives,
        then the shared slots. Plain natives alias ``parameters`` and their wgrads land in the
        staging. GTP natives are shells that the first weight table points at the gathered buffers
        (MXFP8 shells borrow the slots' storage until then, so TE sees complete tensors) with an
        empty ``main_grad`` placeholder until a backward binds its scratch: TE's forward only
        checks that a fused-accumulation weight has *a* main_grad, and the placeholder fails
        loudly if a backward ever ran without one."""
        slots = self.parameters[fc_layer]
        if not _is_gtp(parameters[0]):
            staging = self.native_staging[fc_layer]
            return (*(self._runtime_parameter(p, g) for p, g in zip(parameters, staging)), *slots)
        device = self.config.device
        placeholder = torch.empty(0, dtype=self.config.grad_dtype, device=device)
        if self.config.mxfp8:
            shells = self._wrap_mxfp8(
                parameters[0],
                self.config.member_shapes[fc_layer],
                tuple(tuple(getattr(slot, name) for name in _MXFP8_COMPONENTS) for slot in slots),
                device,
            )
        else:
            shells = (torch.empty(0, dtype=torch.bfloat16, device=device),) * len(parameters)
        return (*(self._runtime_parameter(shell, placeholder) for shell in shells), *slots)

    def destroy(self) -> None:
        """Release the arenas from every runtime parameter that views them and drop the NCCL
        window registrations, while the process group is still alive. The slot parameters are the
        shared objects every layer and every TE op hold, so emptying their storage in place frees
        the arenas without visiting the layers; a forward after this fails."""
        if self.weight_arena is None:
            return
        torch.cuda.synchronize(self.config.device)
        with torch.no_grad():
            for slots in self.parameters:
                for slot in slots:
                    slot.main_grad = None
                    if self.config.mxfp8:
                        for name in _MXFP8_COMPONENTS:
                            getattr(slot, name).set_()
                    else:
                        slot.set_()
        self.weight_handle = self.grad_handle = self.weight_arena = self.grad_arena = None


# ---- Pointer tables --------------------------------------------------------------------------


class _FCLayerPointerTables:
    """One FC layer's ``2L`` runtime parameters (the natives, then the shared slots) and the
    pointer tables the kernels read for them: the ``[components, L]`` weight table per push
    direction and the ``[L]`` table of the natives' wgrad targets, so the kernels' launch
    arguments stay fixed. Storage is static: a weight table is written on the first call and
    every later call asserts that the pointers it was built from still hold; the wgrad table
    follows :meth:`bind_native_grads`, rewritten in place only when GTP hands out other scratch.
    """

    def __init__(self, virtual_experts: _VirtualExperts, index: int, parameters) -> None:
        self.config, self.index, self.parameters = virtual_experts.config, index, parameters
        self.name = f"FC{index + 1}"
        self.numel = math.prod(self.config.member_shapes[index])
        self.runtime_parameters = virtual_experts.runtime_parameters(index, parameters)
        self.gtp_leader = parameters[0] if _is_gtp(parameters[0]) else None
        # What the natives' main_grad points at: the fixed staging (plain), or the GTP scratch
        # bound for the current backward, None in between while they sit on the empty placeholder.
        self.placeholder = None if self.gtp_leader is None else self.runtime_parameters[0].main_grad
        self.native_grads = None
        self._tables: dict[str, tuple[torch.Tensor, list[list[int]]]] = {}
        if virtual_experts.native_staging[index] is not None:
            self.bind_native_grads(virtual_experts.native_staging[index])

    def _components(self, direction: int) -> tuple[str, ...]:
        if self.config.mxfp8:
            return _MXFP8_COMPONENTS[2 * direction : 2 * direction + 2]
        return ("data",)

    def _ptrs(self, direction: int, sources) -> list[list[int]]:
        return [
            [getattr(source, name).data_ptr() for source in sources]
            for name in self._components(direction)
        ]

    def _set_table(self, key: str, rows: list[list[int]]) -> torch.Tensor:
        """Write table ``key`` from ``rows`` of ``data_ptr()`` values; on change, rewrite it in
        place."""
        table = self._tables.get(key)
        if table is None:
            table = (torch.tensor(rows, dtype=torch.int64, device=self.config.device), rows)
        elif rows != table[1]:
            table[0].copy_(torch.tensor(rows, dtype=torch.int64))
            table = (table[0], rows)
        self._tables[key] = table
        return table[0]

    def weight_table(self, direction: int, sources) -> torch.Tensor:
        """The ``[components, L]`` table of ``sources`` for ``direction``. The first call validates
        the sources, points the runtime natives at them and writes the table; later calls assert
        that the pointers have not changed."""
        key, rows = ("forward", "backward")[direction], self._ptrs(direction, sources)
        if key in self._tables:
            if rows != self._tables[key][1]:
                raise RuntimeError(
                    f"{self.name} {key} weights: the storage moved after the virtual-expert "
                    "pointer table was built; virtual experts assume static storage."
                )
            return self._tables[key][0]
        components = self._components(direction)
        expected = (
            ((self.numel, torch.uint8), (self.numel // 32, torch.uint8))
            if self.config.mxfp8
            else ((self.numel, torch.bfloat16),)
        )
        if not all(
            (s := getattr(source, name)).is_contiguous() and s.numel() == count and s.dtype == dtype
            for source in sources
            for name, (count, dtype) in zip(components, expected)
        ):
            raise ValueError(
                f"{self.name}: virtual-expert weight sources must be contiguous {expected} per "
                "expert."
            )
        if self.gtp_leader is not None:  # point the GTP shells at the gathers
            for parameter, source in zip(self.runtime_parameters, sources):
                if self.config.mxfp8:
                    for name in components:
                        setattr(parameter, name, getattr(source, name))
                else:
                    parameter.data = source
        return self._set_table(key, rows)

    def bind_native_grads(self, grads) -> None:
        """Point the natives' ``main_grad`` and the wgrad table at ``grads``, one ``[*shape]``
        buffer per native: the fixed staging, or GTP scratch for one backward. ``None`` parks GTP
        natives back on the placeholder, so a backward without bound scratch fails instead of
        writing into recycled buffers."""
        if grads is None:
            targets = (self.placeholder,) * len(self.parameters)
        else:
            if self.native_grads is not None:
                raise RuntimeError(f"{self.name}: native wgrads bound twice in one backward.")
            if len(grads) != len(self.parameters) or not all(
                g.dtype == self.config.grad_dtype
                and g.numel() == self.numel
                and g.is_contiguous()
                and g.data_ptr() % 16 == 0
                for g in grads
            ):
                raise ValueError(
                    f"{self.name}: native wgrads must be {len(self.parameters)} contiguous 16-byte "
                    f"aligned {self.config.grad_dtype} buffers of {self.numel} elements."
                )
            self._set_table("grad", [[g.data_ptr() for g in grads]])
            targets = grads
        self.native_grads = grads
        for parameter, grad in zip(self.runtime_parameters, targets):
            parameter.main_grad = grad

    def grad_table(self) -> torch.Tensor:
        """The ``[L]`` table of the natives' wgrad bases the reduction accumulates into."""
        return self._tables["grad"][0][0]


# ---- Load balancer ---------------------------------------------------------------------------


class _VirtualExpertHook(torch.autograd.Function):
    """Call ``method(*args)`` when the gradient passes this point. The tensor's gradient is
    unchanged; whatever the method returns becomes the gradients of ``inputs`` (the source
    parameters, at the layer input). The method is held through a weak reference: its owner, the
    dispatcher's manager, holds differentiable tensors across a forward, so a strong reference
    from an autograd context would close a reference cycle through the graph that Python's
    collector cannot see, leaking every layer's graph. The model keeps the owner alive while
    hooks can fire."""

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
    """MoonEP's load balancing (https://github.com/moonshotAI/moonep) as a mixin for
    ``moe.token_dispatcher.MoEFlexTokenDispatcher``'s HybridEP manager: every EP rank receives
    exactly ``tokens x topk`` routes however skewed the routing is, because redundant ("virtual")
    experts are pushed into local slots before expert compute and their gradients are reduced back
    to the owning rank in backward. A rank duplicates experts from a single overloaded home rank,
    so ``num_local_experts`` slots always suffice. The token transport is the underlying
    dispatcher's; the planner, weight push and gradient reduction are the Triton kernels in
    ``virtual_expert_triton.py``, scheduled from here, as are GTP's protocol calls for sharded
    expert weights."""

    # Shared by every layer of the process, allocated by the first layer to bind. Two candidate
    # weight streams: a CUDA-graph capture stream comes from the same pool and may alias one.
    planner: VirtualExpertPlannerWorkspace | None = None
    virtual_experts: _VirtualExperts | None = None
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
        if num_experts != self.ep_size * num_local_experts:
            raise ValueError(
                "Virtual-expert load balancing requires an even expert distribution: "
                f"num_experts={num_experts}, world_size={self.ep_size}, "
                f"num_local_experts={num_local_experts}."
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
        for event in (self.prefetch_done, self.grad_reduce_done):
            event.record(torch.cuda.current_stream(self.device))
        # Bound at the first forward, once DDP has built the main gradients whose dtype the arenas
        # take.
        self.fc_layers: list[_FCLayerPointerTables] = []

    # ---- binding -----------------------------------------------------------------------------

    def bind_experts(self, experts: torch.nn.Module) -> None:
        """Remember the expert module; runtime slots are bound after main-grad initialization."""
        self._experts_ref = weakref.ref(experts)

    def _runtime_init(self, hidden_states: torch.Tensor) -> None:
        """The start of a layer forward: check that the previous pass unwound completely, size
        this rank's transport capacity for the layer's token count (re-sized only if it changes)
        and, once per process, build the runtime parameters over the shared slots."""
        if self._plan is not None:
            raise RuntimeError(
                "Virtual-expert layer input wrapped while the previous pass is still in flight: "
                "its forward was not finalized or its backward did not finish."
            )
        num_tokens = hidden_states.numel() // hidden_states.shape[-1]
        if num_tokens != self.num_tokens:
            self.num_tokens = num_tokens
            self.rank_capacity = self._compute_rank_capacity(num_tokens)

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
        grad_dtypes = {
            None if (grad := getattr(parameter, "main_grad", None)) is None else grad.dtype
            for group in parameters
            for parameter in group
        }
        if len(grad_dtypes) != 1:
            raise RuntimeError(
                "Virtual-expert source main gradients must be initialized with one dtype for every "
                f"expert weight, got {grad_dtypes}."
            )
        # A no-DDP caller has no main gradients: keep the FP32 arena default. Training takes DDP's.
        grad_dtype = grad_dtypes.pop() or torch.float32
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
        )
        cls = VirtualExpertLoadBalancer
        if cls.virtual_experts is None:
            # The planner's allocation runs the first collective on the group, which creates the
            # device communicator the slots' NCCL window registrations need.
            cls.planner = VirtualExpertPlannerWorkspace(
                num_experts=self.ep_size * self.num_owned_experts,
                device=self.device,
                group=self.group,
            )
            cls.virtual_experts = _VirtualExperts(
                self.group, config, templates=tuple(group[0] for group in parameters)
            )
            cls.weight_streams = (
                torch.cuda.Stream(device=self.device),
                torch.cuda.Stream(device=self.device),
            )
            cls.grad_stream = torch.cuda.Stream(device=self.device)
        elif cls.virtual_experts.config != config:
            raise ValueError(
                "All virtual-expert MoE layers of a process run over one set of slots; the first "
                f"layer bound {cls.virtual_experts.config}, this layer needs {config}. Callers "
                "that rebuild their process groups must call VirtualExpertLoadBalancer.finalize() "
                "first."
            )
        self.fc_layers = [
            _FCLayerPointerTables(cls.virtual_experts, i, p) for i, p in enumerate(parameters)
        ]
        experts.bind_virtual_experts(self)

    @classmethod
    def finalize(cls) -> None:
        """Release the shared slots and planner scratch; idempotent. A normal exit needs no call;
        callers that destroy their process groups while a model is still alive (tests, an orderly
        shutdown) call it first so the NCCL windows are deregistered while their communicator
        exists."""
        if cls.virtual_experts is not None:
            cls.virtual_experts.destroy()
        if cls.planner is not None:
            cls.planner.destroy()
        cls.virtual_experts = cls.planner = cls.weight_streams = cls.grad_stream = None
        # TE's op contexts and autograd graphs hold the runtime parameters in reference cycles;
        # collect now so anything still viewing the arenas is freed while the group is alive.
        gc.collect()

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

    def _weight_stream(self, current_stream: torch.cuda.Stream) -> torch.cuda.Stream:
        """Return a weight stream distinct from ``current_stream``."""
        return next(s for s in self.weight_streams if s.cuda_stream != current_stream.cuda_stream)

    def _fc_layers(self, direction: int) -> list:
        """The FC layers in GEMM order: the expert backward computes FC2 before FC1, and
        GTP's linked gathers are consumed in that order."""
        return self.fc_layers[:: -1 if direction == BACKWARD else 1]

    def _compute_rank_capacity(self, num_tokens: int) -> int:
        """A static, dropless route capacity for one transport rank: every rank receives
        exactly its own route count, plus HybridEP's per-runtime-expert segment padding."""
        num_routes = num_tokens * self.router_topk
        alignment = self._alignment
        padding = self.num_runtime_experts * max(alignment - 1, 0)
        capacity = max(
            int(num_routes * self.config.moe_expert_rank_capacity_factor), num_routes + padding
        )
        return capacity + (-capacity % alignment if alignment > 1 else 0)

    # ---- GTP protocol ------------------------------------------------------------------------

    def _weight_sources(self, fc_layer, direction: int, peek: bool = True):
        """The weights one FC layer's push reads for ``direction``: plain parameters, once DDP has
        published them (the push runs ahead of the expert module's pre-forward hook, where DDP would
        otherwise finish; a backward re-reads what the forward waited for), or GTP's gathered
        buffers, peeked at for the push and consumed right before the expert GEMMs, where TE would
        consume them, so virtual experts leave GTP's gather and prefetch schedule alone."""
        if fc_layer.gtp_leader is None:
            if direction == FORWARD:
                ensure_params_ready(fc_layer.parameters)
            return fc_layer.parameters

        leader = fc_layer.gtp_leader
        if direction == FORWARD:
            gathered = (
                leader.peek_group_for_forward() if peek else leader.materialize_group_for_forward()
            )
        else:
            gathered = (
                leader.peek_group_for_backward()
                if peek
                else leader.materialize_group_for_backward()
            )
        return _as_tuple(gathered)

    def _consume_gtp_weights(self, fc_layer, direction: int) -> None:
        """GTP's real chain step for one FC layer. The buffers must be the ones the push peeked at;
        anything else means the push copied other bytes than the GEMM reads."""
        if fc_layer.gtp_leader is not None:
            fc_layer.weight_table(direction, self._weight_sources(fc_layer, direction, peek=False))

    def _hand_off_wgrads(self, fc_layer) -> tuple[torch.Tensor, ...]:
        """After the reduction: what autograd delivers to each of one FC layer's source parameters.
        Plain parameters accumulate the staging into ``main_grad`` and get TE's dummy, so
        AccumulateGrad still fires DDP's grad-ready hook without adding the dummy again (without a
        main_grad, autograd owns a copy). GTP parameters reduce-scatter the bound scratch through
        the protocol call TE issues after a wgrad GEMM (``finalize_group_grads``), which adds into
        the shards' ``main_grad`` itself; the natives then park on the placeholder."""
        from transformer_engine.pytorch.module.base import get_dummy_wgrad

        if fc_layer.gtp_leader is not None:
            if fc_layer.native_grads is None:
                raise RuntimeError(
                    f"{fc_layer.name}: no GTP wgrad scratch bound for this backward."
                )
            reduced = fc_layer.gtp_leader.finalize_group_grads(list(fc_layer.native_grads))
            fc_layer.bind_native_grads(None)
            return _as_tuple(reduced)
        grads = []
        for parameter, wgrad in zip(fc_layer.parameters, fc_layer.native_grads):
            if getattr(parameter, "main_grad", None) is None:
                grads.append(wgrad.clone())
            else:
                parameter.main_grad.add_(wgrad)
                parameter.grad_added_to_main_grad = True
                grads.append(
                    get_dummy_wgrad(
                        list(parameter.shape),
                        parameter.dtype,
                        zero=getattr(parameter, "zero_out_wgrad", False),
                    )
                )
        return tuple(grads)

    # ---- forward: dispatcher hooks -----------------------------------------------------------

    def wrap_layer_input(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Start a layer forward and attach its gradient completion hook to the whole MoE layer
        input; the hook returns the source parameters' gradients. It needs the router's input
        gradient, hence the dispatch backward; the FC1 reduction start sits on the dispatch input,
        is created later in the forward and so runs first under autograd's ordering."""
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
        """Plan the router's ``[num_tokens, topk]`` routes and begin the weight push, before
        shared-expert compute."""
        self._plan = plan_virtual_expert_routes(top_indices, self.planner)
        self._start_weight_push(FORWARD)

    def prepare_virtual_expert_dispatch(
        self, hidden_states: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Attach dispatch-backward work and return what the transport dispatches: the hidden
        states and the int16 ``[num_tokens, topk]`` runtime expert ids (the router's
        probabilities are unchanged by the remap)."""
        hidden_states = _VirtualExpertHook.apply(
            hidden_states, self._start_pending_grad_reduces, ()
        )
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
    def _start_weight_push(self, direction: int) -> None:
        """Enqueue the owner push of the pass's FC1/FC2 weights on the weight stream. Only reads
        the weights: GTP consumes them, and issues its next prefetch, at the expert GEMMs."""
        if self._plan.push_in_flight:
            raise RuntimeError("Virtual-expert weight prefetch is already outstanding.")
        # Peeks need no GEMM order: both precede this direction's consumes, and neither gather
        # depends on the other.
        tables = [
            fc_layer.weight_table(direction, self._weight_sources(fc_layer, direction))
            for fc_layer in self.fc_layers
        ]
        virtual_experts = self.virtual_experts
        current_stream = torch.cuda.current_stream(self.device)
        weight_stream = self._weight_stream(current_stream)
        weight_stream.wait_stream(current_stream)
        with torch.cuda.stream(weight_stream):
            launch_virtual_expert_weight_prefetch(
                virtual_experts,
                sources=tuple(table[0] for table in tables),
                scale_sources=(
                    tuple(table[1] for table in tables) if virtual_experts.config.mxfp8 else None
                ),
                experts_to_copy=self._plan.experts_to_copy,
            )
            self.prefetch_done.record(weight_stream)
        self._plan.push_in_flight = True

    @torch.no_grad()
    @nvtx_decorator(message="virtual_expert_weight_push_wait")
    def _wait_weight_push(self) -> None:
        """Make the current stream wait for the pass's push; exactly once per push."""
        if not self._plan.push_in_flight:
            raise RuntimeError("Virtual-expert weights require a started, unwaited prefetch.")
        torch.cuda.current_stream(self.device).wait_event(self.prefetch_done)
        self._plan.push_in_flight = False

    @torch.no_grad()
    def prepare_expert_forward(self) -> None:
        """Right before the expert forward GEMMs: wait for the forward push, let GTP consume."""
        self._wait_weight_push()
        for fc_layer in self._fc_layers(FORWARD):
            self._consume_gtp_weights(fc_layer, FORWARD)

    @torch.no_grad()
    def _start_backward(self, plan: VirtualExpertPlan) -> None:
        """At the layer output's backward: take the plan back and start its backward push."""
        if self._plan is not None:
            raise RuntimeError("Virtual-expert backward started while another pass is outstanding.")
        self._plan = plan
        self._start_weight_push(BACKWARD)

    @torch.no_grad()
    def _prepare_expert_backward(self) -> None:
        """Right before the expert backward, FC2 first: wait for the backward push, let GTP consume
        and bind GTP's wgrad scratch for this backward through TE's protocol (``get_wgrad_tensor``).
        The wgrad GEMM writes the natives' gradients straight into the buffers the reduce-scatter
        sends and the reduction adds the virtual-expert partials there, so nothing is copied; GTP
        recycles the buffers once its reduce-scatter has read them, hence one bind per backward."""
        self._wait_weight_push()
        for fc_layer in self._fc_layers(BACKWARD):
            self._consume_gtp_weights(fc_layer, BACKWARD)
            if fc_layer.gtp_leader is not None:
                fc_layer.bind_native_grads(
                    tuple(weight.get_wgrad_tensor() for weight in fc_layer.gtp_leader._weights)
                )

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
        self.grad_stream.wait_stream(torch.cuda.current_stream(self.device))
        with torch.cuda.stream(self.grad_stream):
            launch_virtual_expert_grad_reduce(
                self.virtual_experts,
                native_grads=tuple(fc_layer.grad_table() for fc_layer in self.fc_layers),
                experts_to_copy=plan.experts_to_copy,
                fc_layers=(fc_layer,),
            )
            # Both reductions run in order on the reduction stream, FC2 first, so the event
            # recorded after the last launch covers both.
            self.grad_reduce_done.record(self.grad_stream)
        plan.started.add(fc_layer)

    def start_fc2_grad_reduce(self) -> None:
        """Start FC2's reduction from the expert backward, right behind FC2's wgrad GEMM."""
        self._start_grad_reduce(1)

    def _start_pending_grad_reduces(self) -> None:
        """Start every reduction not yet started, FC2 first, once dispatch backward is done. FC2
        normally starts from the FC2 op's wgrad store; FC1 starts here and hides behind the
        latent, shared-expert and router backward."""
        for fc_layer in (1, 0):
            if fc_layer not in self._plan.started:
                self._start_grad_reduce(fc_layer)

    @torch.no_grad()
    @nvtx_decorator(message="virtual_expert_grad_reduce_wait")
    def _finish_grad_reduce(self) -> tuple[torch.Tensor, ...]:
        """Close the pass: wait for both reductions and hand the wgrads off, FC2 first as GTP's
        linked chain expects, returning one gradient per source parameter, FC1 then FC2.

        GTP's reduce-scatter is issued here, later than TE would, because the natives' gradient
        is complete only after the cross-rank reduction; and on the compute stream, because GTP's
        finalize also adds the shard gradients into ``main_grad`` and fires DDP's grad-ready hooks,
        which DDP orders against the compute stream (a bucket shared with compute-stream
        parameters scales and copies the grad buffer from whichever hook completes it). This is
        the earliest compute-stream point after both reductions: FC2's started at its wgrad GEMM
        and FC1's at the dispatch backward, so the waits are normally free.
        """
        plan, self._plan = self._plan, None
        if plan is None or plan.started != {0, 1}:
            raise RuntimeError(
                "Virtual-expert gradient reduction of both fc_layers must be started."
            )
        torch.cuda.current_stream(self.device).wait_event(self.grad_reduce_done)
        grads = [None, None]
        for fc_layer in self._fc_layers(BACKWARD):
            grads[fc_layer.index] = self._hand_off_wgrads(fc_layer)
        return tuple(grad for fc_layer in grads for grad in fc_layer)
