# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.


"""Virtual-expert load balancing, virtual-expert planning, and runtime weight movement.
The load-balancing algorithm implemented here is taken from MoonEP
(https://github.com/moonshotAI/moonep): perfectly balanced expert parallelism through
redundant experts planned online from the router output.
"""

from __future__ import annotations

import gc
import weakref
from contextlib import contextmanager
from functools import partial
from typing import TYPE_CHECKING

import torch

from megatron.core.fp8_utils import is_mxfp8tensor
from megatron.core.transformer.moe.moe_utils import get_align_size_for_quantization

try:
    from megatron.core.transformer.moe.ops.virtual_expert_triton import (
        MAX_VIRTUAL_EXPERT_WEIGHT_SMS,
        VirtualExpertPlannerWorkspace,
    )

    _TRITON_AVAILABLE = True
except ImportError:
    _TRITON_AVAILABLE = False
    MAX_VIRTUAL_EXPERT_WEIGHT_SMS = VirtualExpertPlannerWorkspace = None
from megatron.core.transformer.moe.moe_scheduler_plan import MoonEPPlanner, VirtualExpertPlan
from megatron.core.transformer.moe.moe_scheduler_runtime import (
    WeightDirection,
    _is_gtp,
    _VirtualExpertConfig,
    _VirtualExpertHook,
    _VirtualExperts,
    _VirtualExpertStorage,
)
from megatron.core.transformer.moe.moonep_peer_transport import MoonEPPeerTransport
from megatron.core.utils import nvtx_decorator

if TYPE_CHECKING:
    from megatron.core.transformer.transformer_config import TransformerConfig


class MoonEPExecutionRuntime:
    """Balance HybridEP routes with virtual experts using the MoonEP algorithm.

    Every rank receives tokens * topk routes. Each rank copies experts from one owner into
    num_local_experts slots and reduces their gradients back during backward. HybridEP owns
    token transport; this mixin schedules planning, weight movement, and gradient reduction."""

    load_planner = MoonEPPlanner()
    transport = MoonEPPeerTransport()

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
        # Match HybridEP's config-based alignment; the planner represents no padding as 1.
        self._alignment = max(1, get_align_size_for_quantization(config))
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
        for fc_layer, group in enumerate(parameters, start=1):
            for index, parameter in enumerate(group):
                if getattr(parameter, "main_grad", None) is None:
                    raise RuntimeError(
                        "MoonEP requires eagerly allocated expert main_grad buffers before its "
                        f"first forward; FC{fc_layer}.weight{index} has no main_grad. "
                        "Lazy gradient allocation is not supported."
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
        cls = MoonEPExecutionRuntime
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
        """Release shared slots and planner scratch before the expert process group is destroyed.

        Called by ``destroy_model_parallel``; safe to call explicitly or repeatedly.
        """
        if (
            not cls.storages
            and cls.planner is None
            and cls.planner_stream is None
            and cls.weight_streams is None
            and cls.grad_stream is None
        ):
            return
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
        """Dropless capacity from the worst-case number of globally rounded expert blocks."""
        return self.load_planner.rank_capacity(
            num_tokens * self.router_topk,
            self.ep_size * self.num_owned_experts,
            self.ep_size,
            self._alignment,
        )

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
            self._plan = self.load_planner.plan(top_indices, self.planner, self._alignment)
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

    def finalize_layer_output(self, output: torch.Tensor) -> torch.Tensor:
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
            self.transport.push(
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
            self.transport.reduce(
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
        """Wait for VE reductions, clear accumulating slots, and hand off native gradients.

        Finalize GTP FC2 first on the compute stream, returning gradients FC1 first.
        GTP adds shard gradients to main_grad and fires DDP hooks, which may scale/copy
        buckets shared with compute-stream parameters."""
        plan, self._plan = self._plan, None
        if plan is None or plan.started != {0, 1}:
            raise RuntimeError(
                "Virtual-expert gradient reduction of both fc_layers must be started."
            )
        torch.cuda.current_stream(self.device).wait_event(self.grad_reduce_done)
        self.virtual_experts.storage.clear_accumulating_grads()
        fc2_grads = self.virtual_experts.hand_off_wgrads(1)
        return (*self.virtual_experts.hand_off_wgrads(0), *fc2_grads)
