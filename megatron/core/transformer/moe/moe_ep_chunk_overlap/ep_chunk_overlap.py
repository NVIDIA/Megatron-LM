# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""The token-chunked EP overlap path: parameters, TE grouped MLP, schedule.

Compute is the caller's current stream; dispatch and combine share ONE comm stream.
Chunk ``c``'s compute is left of the ``||``::

    forward   (ascending)    FC1(c) -> FC2(c)   ||   Dispatch(c+1) or Combine(c-1)
    backward  (descending)   dgrad(c)           ||   Dispatch(c-1) or Combine(c+1)
                             wgrad(c)           ||   Combine(c)
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import chain

import torch
import torch.nn.functional as F
from torch.autograd.function import once_differentiable

from megatron.core.activations import squared_relu
from megatron.core.fusions.fused_bias_geglu import quick_gelu
from megatron.core.pipeline_parallel.fine_grained_activation_offload import (
    FineGrainedActivationOffloadingInterface as off_interface,
)
from megatron.core.tensor_parallel.gtp_api import is_gtp_param
from megatron.core.transformer.transformer_config import TransformerConfig

from .transport import DispatchedChunk, MoEEPTransport, MoEEPTransportType, make_ep_transport

EXPERT_ALIGNMENT_TO_GROUPED_GEMM = 256


def balanced_chunk_sizes(num_tokens: int, num_chunks: int) -> tuple[int, ...]:
    base, remainder = divmod(num_tokens, num_chunks)
    return tuple(base + (1 if index < remainder else 0) for index in range(num_chunks))


def chunk_offsets(chunk_sizes: tuple[int, ...]) -> tuple[int, ...]:
    offsets = [0]
    for size in chunk_sizes:
        offsets.append(offsets[-1] + size)
    return tuple(offsets)


@dataclass(frozen=True)
class _MoEEPChunkOverlapConfig:
    num_local_tokens: int
    hidden_size: int
    ffn_hidden_size: int
    num_experts: int
    router_topk: int
    num_token_chunks: int
    world_size: int
    num_comm_SMs: int

    def __post_init__(self) -> None:
        if self.num_token_chunks < 1:
            raise ValueError(f"num_token_chunks must be >= 1, got {self.num_token_chunks}")
        if self.num_local_tokens < self.num_token_chunks:
            raise ValueError(
                f"num_local_tokens ({self.num_local_tokens}) must be >= num_token_chunks "
                f"({self.num_token_chunks}); an empty chunk is not a supported dispatch shape"
            )

    @property
    def chunk_sizes(self) -> tuple[int, ...]:
        return balanced_chunk_sizes(self.num_local_tokens, self.num_token_chunks)

    @property
    def chunk_offsets(self) -> tuple[int, ...]:
        return chunk_offsets(self.chunk_sizes)

    @property
    def largest_token_chunk_size(self) -> int:
        return -(-self.num_local_tokens // self.num_token_chunks)

    @property
    def num_local_experts(self) -> int:
        return self.num_experts // self.world_size

    @property
    def max_num_expanded_tokens_per_chunk(self) -> int:
        num_local_experts = self.num_local_experts
        rows = self.world_size * self.largest_token_chunk_size * min(
            self.router_topk, num_local_experts
        ) + num_local_experts * (EXPERT_ALIGNMENT_TO_GROUPED_GEMM - 1)
        return -(-rows // EXPERT_ALIGNMENT_TO_GROUPED_GEMM) * EXPERT_ALIGNMENT_TO_GROUPED_GEMM


@dataclass(frozen=True)
class CachedForwardChunkActivation:
    dispatched_tokens: torch.Tensor
    routing_probs: torch.Tensor
    fc2_output: torch.Tensor
    dispatch_handle: object


def ensure_expert_parameters_ready(experts: torch.nn.Module) -> None:
    """Fire the expert linears' forward-pre hooks; skipping silently desyncs the DP all-gather."""
    for submodule in chain(experts.linear_fc1.modules(), experts.linear_fc2.modules()):
        for hook_id, hook in list(submodule._forward_pre_hooks.items()):
            if hook_id in submodule._forward_pre_hooks_with_kwargs:
                returned = hook(submodule, (), {})
            else:
                returned = hook(submodule, ())
            if returned is not None:
                raise RuntimeError(
                    f"a {type(submodule).__name__} forward-pre hook returned a replacement "
                    "input, which moe_ep_chunk_overlap cannot honour: it never calls forward"
                )
    experts._ensure_main_grad_for_fused_impl()


def expert_weight_parameters(
    experts: torch.nn.Module, num_local_experts: int
) -> tuple[torch.nn.Parameter, ...]:
    return tuple(
        linear.get_parameter(f"weight{index}")
        for linear in (experts.linear_fc1, experts.linear_fc2)
        for index in range(num_local_experts)
    )


def check_fused_grouped_mlp(fc1_op, activation_op, fc2_op) -> None:
    """Raise unless TE will fuse this triple and honour its recompute; both failures are silent."""
    from transformer_engine.pytorch.ops.fused.grouped_mlp import (
        GroupedMLP_CuTeGEMMGLU,
        GroupedMLP_CuTeGEMMUnary,
        ScaledSReLU,
        _grouped_gemm_dsrelu_backward_supported,
        _nvidia_cudnn_frontend_supports_wgrad,
        is_glu_activation,
        validate_grouped_mlp_dims,
    )

    fused = GroupedMLP_CuTeGEMMGLU if is_glu_activation(activation_op) else GroupedMLP_CuTeGEMMUnary
    if not fused.is_supported():
        raise RuntimeError(
            f"TransformerEngine did not register {fused.__name__}, so this path would silently "
            "run three unfused ops. Export NVTE_CUTEDSL_FUSED_GROUPED_MLP=1 BEFORE importing TE."
        )

    validate_grouped_mlp_dims(fc1_op, activation_op, fc2_op)

    if activation_op.activation_recompute_in_mlp and not (
        isinstance(activation_op, ScaledSReLU)
        and _grouped_gemm_dsrelu_backward_supported()
        and _nvidia_cudnn_frontend_supports_wgrad()
    ):
        raise RuntimeError(
            "'moe_act' in recompute_modules asks TE for in-MLP recompute it will not do: "
            f"op={type(activation_op).__name__} dsrelu={_grouped_gemm_dsrelu_backward_supported()}"
            f" fe_wgrad={_nvidia_cudnn_frontend_supports_wgrad()}; TE saves the FC2 input anyway"
        )


def build_activation_op(
    config: TransformerConfig, *, activation_recompute: bool
) -> torch.nn.Module:
    """The TE activation this config asks for; only these three take the routing probability."""
    from transformer_engine.pytorch import ops as te_ops

    glu = {"glu_interleave_size": config.moe_mlp_glu_interleave_size}
    clamp = config.activation_func_clamp_value

    recompute = {"activation_recompute_in_mlp": activation_recompute}
    if config.gated_linear_unit and config.activation_func == F.silu:
        if clamp is None:
            return te_ops.ScaledSwiGLU(**glu, **recompute)
        return te_ops.ScaledClampedQGeGLU(
            **glu, **recompute, alpha=1.0, limit=clamp, glu_linear_offset=0.0
        )
    if config.gated_linear_unit and config.activation_func == quick_gelu:
        return te_ops.ScaledClampedQGeGLU(
            **glu, **recompute, **({} if clamp is None else {"limit": clamp})
        )
    if not config.gated_linear_unit and config.activation_func == squared_relu:
        return te_ops.ScaledSReLU(**recompute)
    raise RuntimeError(
        f"cannot fold the routing probability into {config.activation_func} "
        f"(gated={config.gated_linear_unit}): only ScaledSwiGLU/ScaledClampedQGeGLU/ScaledSReLU"
    )


def build_expert_ops(
    config: _MoEEPChunkOverlapConfig, experts: torch.nn.Module
) -> tuple[torch.nn.Module, torch.nn.Module, torch.nn.Module]:
    from transformer_engine.pytorch import ops as te_ops

    num_local_experts = config.num_local_experts
    params = expert_weight_parameters(experts, num_local_experts)
    ops = []

    for weights, linear in (
        (params[:num_local_experts], experts.linear_fc1),
        (params[num_local_experts:], experts.linear_fc2),
    ):
        op = te_ops.GroupedLinear(
            linear.num_gemms,
            linear.in_features,
            linear.out_features,
            bias=False,
            device="meta",
            dtype=weights[0].dtype,
            accumulate_into_main_grad=True,
            delay_wgrad_compute=True,
        )
        for index, weight in enumerate(weights):
            op.register_parameter(f"weight{index}", weight)
        ops.append(op)
    fc1_op, fc2_op = ops

    activation_op = build_activation_op(
        experts.config, activation_recompute=experts.activation_recompute
    )
    check_fused_grouped_mlp(fc1_op, activation_op, fc2_op)
    return te_ops.Sequential(fc1_op, activation_op, fc2_op), fc1_op, fc2_op


def expert_row_counts(cumulative_recv_tokens: torch.Tensor) -> torch.Tensor:
    """Derive TE's per-expert split sizes from the transport's cumulative receive counts."""
    alignment = EXPERT_ALIGNMENT_TO_GROUPED_GEMM
    ends = ((cumulative_recv_tokens.to(torch.int64) + (alignment - 1)) // alignment) * alignment
    starts = torch.cat([ends.new_zeros(1), ends[:-1]])
    return ends - starts


def autocast_recipe() -> object:
    from transformer_engine.pytorch.quantization import FP8GlobalStateManager

    if not FP8GlobalStateManager.is_fp8_enabled():
        raise RuntimeError(
            "no live MXFP8 quantization autocast around this layer's forward. FIX: launch with "
            "the mxfp8 recipe, or clear first_last_layers_bf16 (fp8_utils.py:714-724) here"
        )
    recipe = FP8GlobalStateManager.get_fp8_recipe()
    if not recipe.mxfp8():
        raise RuntimeError(
            f"the live autocast carries a {type(recipe).__name__} recipe; this path implements "
            "MXFP8 block scaling only, and going ahead would silently discard the configured one"
        )
    return recipe


def _needs_backward_gather(shadow) -> bool:
    from transformer_engine.pytorch.ops._common import is_quantized_tensor

    if not is_quantized_tensor(shadow):
        return False
    return shadow._columnwise_data is None


def _install_backward_gather(op, weights) -> int:
    """Rematerialize an EGTP expert group COLUMNWISE for the backward."""
    from transformer_engine.pytorch.cpu_offload import mark_not_offload

    num_local_experts = len(weights)
    shadows = [op.__dict__.get(f"weight{index}") for index in range(num_local_experts)]
    stale = [i for i, s in enumerate(shadows) if s is not None and _needs_backward_gather(s)]
    if not stale:
        return 0

    bwd_weights = list(weights[0].materialize_group_for_backward())

    for index in stale:
        shadow, bwd_w = shadows[index], bwd_weights[index]
        columnwise_data = getattr(bwd_w, "_columnwise_data", None)
        columnwise_scale = getattr(bwd_w, "_columnwise_scale_inv", None)

        mark_not_offload(bwd_w)
        shadow._columnwise_data = columnwise_data
        shadow._columnwise_scale_inv = columnwise_scale

        shadow._with_gemm_swizzled_scales = getattr(bwd_w, "_with_gemm_swizzled_scales", False)

        shadow._rowwise_data = None
        shadow._rowwise_scale_inv = None

    op.__dict__["_egtp_bwd_weight_group"] = bwd_weights
    return len(bwd_weights)


def _attach_wgrad_accumulator(op, index: int, param) -> None:
    """Give one EGTP expert's weight slot a full-sized ZEROED wgrad accumulator as ``main_grad``."""
    accumulator = param.get_wgrad_tensor()
    accumulator.zero_()
    op.__dict__[f"weight{index}"].main_grad = accumulator


def run_expert_forward(
    runtime: MoEEPChunkOverlapRuntime,
    *,
    recv_x: torch.Tensor,
    received_topk_weights: torch.Tensor,
    cumulative_recv_tokens: torch.Tensor,
    dispatch_handle: object,
    stream: torch.cuda.Stream,
    offload: bool,
) -> tuple[torch.Tensor, CachedForwardChunkActivation]:
    """Run FC1, the weighted squared ReLU and FC2 as one fused op, inside its own offload group."""
    from transformer_engine.pytorch import autocast

    ops = runtime.expert_ops()
    split_sizes = expert_row_counts(cumulative_recv_tokens)
    dispatched_tokens = recv_x.detach().requires_grad_(True)
    routing_probs = received_topk_weights.detach().reshape(-1).requires_grad_(True)
    offload_group = off_interface(offload, dispatched_tokens, "fused_group_mlp")

    with (
        torch.cuda.stream(stream),
        torch.enable_grad(),
        autocast(enabled=True, recipe=autocast_recipe()),
    ):
        with offload_group as group_input:
            fc2_output = ops(group_input, split_sizes, routing_probs, split_sizes)
        fc2_output = offload_group.group_offload(fc2_output)
        transport_output = fc2_output.detach()
    return transport_output, CachedForwardChunkActivation(
        dispatched_tokens=dispatched_tokens,
        routing_probs=routing_probs,
        fc2_output=fc2_output,
        dispatch_handle=dispatch_handle,
    )


def run_expert_backward(
    *, recv_x: torch.Tensor, saved: CachedForwardChunkActivation, stream: torch.cuda.Stream
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run the dgrad pair; the weight gradients stay parked for ``run_chunk_wgrad`` to drain."""

    saved.dispatched_tokens.record_stream(stream)
    saved.routing_probs.record_stream(stream)
    saved.fc2_output.record_stream(stream)

    with torch.cuda.stream(stream):
        grad_input, grad_probs = torch.autograd.grad(
            outputs=(saved.fc2_output,),
            inputs=(saved.dispatched_tokens, saved.routing_probs),
            grad_outputs=(recv_x,),
            retain_graph=False,
        )
    return grad_input, grad_probs


class _MoEEPChunkOverlapFunction(torch.autograd.Function):

    @staticmethod
    def forward(
        ctx,
        hidden_states: torch.Tensor,
        topk_indices: torch.Tensor,
        topk_weights: torch.Tensor,
        runtime: "MoEEPChunkOverlapRuntime",
        *expert_weights: torch.Tensor,
    ) -> torch.Tensor:
        output, saved_fwd = runtime.moe_forward(
            hidden_states,
            topk_indices,
            topk_weights,
            offload=runtime._experts.offload_fused_group_mlp,
        )
        ctx.runtime = runtime
        ctx.saved_fwd = saved_fwd
        ctx.num_expert_weights = len(expert_weights)

        ctx.set_materialize_grads(False)
        return output

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output: torch.Tensor | None):
        runtime = ctx.runtime
        saved_fwd = ctx.saved_fwd

        ctx.runtime = None
        ctx.saved_fwd = None

        grad_hidden_states, grad_topk_weights, wgrad_placeholders = runtime.moe_backward(
            saved_fwd, grad_output
        )
        return (grad_hidden_states, None, grad_topk_weights, None) + wgrad_placeholders


class MoEEPChunkOverlapRuntime:
    """One layer's transport, compiled GEMMs and borrowed expert parameters."""

    def __init__(
        self,
        *,
        num_experts: int,
        router_topk: int,
        hidden_size: int,
        ffn_hidden_size: int,
        num_chunks: int,
        num_comm_SMs: int,
        experts: torch.nn.Module,
        tp_ep_group: torch.distributed.ProcessGroup,
        num_local_experts: int,
        transport_type: MoEEPTransportType = MoEEPTransportType.HYBRID_EP,
    ) -> None:
        if tp_ep_group is None:
            raise ValueError("moe_ep_chunk_overlap requires pg_collection.tp_ep, but it is None")

        self._experts = experts
        self._tp_ep_group = tp_ep_group
        self._num_local_experts = num_local_experts

        self._num_experts: int = num_experts
        self._router_topk: int = router_topk
        self._hidden_size: int = hidden_size
        self._ffn_hidden_size: int = ffn_hidden_size
        self._num_chunks: int = num_chunks
        self._num_comm_SMs: int = num_comm_SMs
        self._transport_type: MoEEPTransportType = transport_type

        self.config: _MoEEPChunkOverlapConfig | None = None
        self.device: torch.device | None = None
        self.expert_params: tuple[torch.nn.Parameter, ...] | None = None
        self._transport: MoEEPTransport | None = None
        self.comm_stream: torch.cuda.Stream | None = None

        self._ops: torch.nn.Module | None = None
        self._fc1_op: torch.nn.Module | None = None
        self._fc2_op: torch.nn.Module | None = None

    def forward(
        self, hidden_states: torch.Tensor, dense_probs: torch.Tensor
    ) -> tuple[torch.Tensor, None]:
        input_shape = hidden_states.shape
        flat_hidden = hidden_states.view(-1, self._hidden_size)

        topk_weights, topk_indices = torch.topk(
            dense_probs, k=self._router_topk, dim=-1, sorted=True
        )

        self._ensure_built(flat_hidden)
        flat_output = _MoEEPChunkOverlapFunction.apply(
            flat_hidden, topk_indices, topk_weights, self, *self.expert_params
        )
        return flat_output.view(input_shape), None

    def _ensure_built(self, flat_hidden: torch.Tensor) -> None:
        """Bind the device-side state to this batch on the first forward; a later one returns."""
        if self._transport is not None:
            return

        world_size = torch.distributed.get_world_size(self._tp_ep_group)
        overlap_config = _MoEEPChunkOverlapConfig(
            num_local_tokens=flat_hidden.shape[0],
            hidden_size=self._hidden_size,
            ffn_hidden_size=self._ffn_hidden_size,
            num_experts=self._num_experts,
            router_topk=self._router_topk,
            num_token_chunks=self._num_chunks,
            world_size=world_size,
            num_comm_SMs=self._num_comm_SMs,
        )
        self.config = overlap_config
        self.device = torch.device(flat_hidden.device)

        self.expert_params = expert_weight_parameters(
            self._experts, overlap_config.num_local_experts
        )
        self._transport = make_ep_transport(
            overlap_config, self._tp_ep_group, transport_type=self._transport_type
        )
        self.comm_stream = torch.cuda.Stream(device=self.device)

    def expert_ops(self) -> torch.nn.Module:
        if self._ops is None:
            self._ops, self._fc1_op, self._fc2_op = build_expert_ops(self.config, self._experts)
        return self._ops

    def destroy(self) -> None:
        """Release the transport's collective resources."""
        if self._transport is None:
            return
        self._transport.destroy_buffers()
        self._transport = None

    def quantize_expert_weights(self) -> None:
        """Quantize every expert weight ONCE per forward, not per chunk"""
        from transformer_engine.pytorch.cpu_offload import mark_not_offload
        from transformer_engine.pytorch.ops._common import is_quantized_tensor

        recipe = autocast_recipe()
        self.expert_ops()
        num_local_experts = self.config.num_local_experts
        params = self.expert_params
        for op, weights in (
            (self._fc1_op, params[:num_local_experts]),
            (self._fc2_op, params[num_local_experts:]),
        ):
            if op.get_quantizer("forward", 1) is None:
                op.reset_recipe_state(recipe=recipe)

            is_egtp = is_gtp_param(weights[0])
            if is_egtp:
                source_weights = list(weights[0].materialize_group_for_forward())
            else:
                source_weights = list(weights)
            for index, weight in enumerate(source_weights):
                if is_quantized_tensor(weight):
                    # fp8_param: TE reuses it untouched, but under EGTP the MATERIALIZED tensor must
                    # still become the shadow
                    if is_egtp:
                        mark_not_offload(weight)
                        op.__dict__[f"weight{index}"] = weight
                    continue
                quantizer = op.get_quantizer("forward", 2 * index + 1)

                quantizer.internal = False
                quantizer.set_usage(rowwise=True, columnwise=True)
                quantized = quantizer(weight).requires_grad_(True)

                mark_not_offload(quantized)
                op.__dict__[f"weight{index}"] = quantized

    def release_expert_weights(self) -> None:
        """Hand the backward a weight slot whose ``main_grad`` TE accumulates into."""
        num_local_experts = self.config.num_local_experts
        params = self.expert_params
        for op, weights in (
            (self._fc1_op, params[:num_local_experts]),
            (self._fc2_op, params[num_local_experts:]),
        ):
            if op is None:
                continue
            if is_gtp_param(weights[0]):
                _install_backward_gather(op, list(weights))
                for index, param in enumerate(weights):
                    _attach_wgrad_accumulator(op, index, param)
                continue
            for index in range(num_local_experts):
                op.__dict__.pop(f"weight{index}", None)

    def routed_experts_compute_forward(
        self, dispatched: DispatchedChunk, *, offload: bool
    ) -> tuple[torch.Tensor, CachedForwardChunkActivation]:
        """Run one chunk's expert compute on the ambient stream; return its rows and saved state."""
        stream = torch.cuda.current_stream(self.device)

        dispatched.recv_x.record_stream(stream)
        dispatched.received_topk_weights.record_stream(stream)
        self._transport.record_handle_stream(dispatched.dispatch_handle, stream)

        return run_expert_forward(
            self,
            recv_x=dispatched.recv_x,
            received_topk_weights=dispatched.received_topk_weights,
            cumulative_recv_tokens=dispatched.dispatch_handle.psum_num_recv_tokens_per_expert,
            dispatch_handle=dispatched.dispatch_handle,
            stream=stream,
            offload=offload,
        )

    def routed_experts_compute_backward(
        self, saved: CachedForwardChunkActivation, dispatched: DispatchedChunk
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run one chunk's dgrad pair on the ambient stream; the wgrads stay parked in TE FIFOs."""
        stream = torch.cuda.current_stream(self.device)
        dispatched.recv_x.record_stream(stream)

        grad_input, grad_probs = run_expert_backward(
            recv_x=dispatched.recv_x, saved=saved, stream=stream
        )
        grad_probs = grad_probs.reshape(-1)
        return grad_input, grad_probs

    def run_chunk_wgrad(self, *, stream: torch.cuda.Stream) -> None:
        with torch.cuda.stream(stream):
            for label, op in (("FC2", self._fc2_op), ("FC1", self._fc1_op)):
                op.backward_dw()

    def begin_wgrad_accumulation(self) -> None:
        """Fetch every expert parameter's ``main_grad`` and clear ``overwrite_main_grad`` if set."""
        from transformer_engine.pytorch.ops._common import (
            get_accumulate_flag_in_param,
            get_main_grad_from_param,
        )

        # The wgrad is the only reader that needs the real parameters back;
        self.release_expert_weights()

        for param in self.expert_params:
            main_grad = get_main_grad_from_param(param, op_label="moe_ep_chunk_overlap")
            if not get_accumulate_flag_in_param(param):
                main_grad.zero_()
                param.overwrite_main_grad = False

    def end_wgrad_accumulation(self) -> None:
        """Reduce-scatter each EGTP expert group's accumulated wgrad ONCE, then drop the shadows."""
        num_local_experts = self.config.num_local_experts
        params = self.expert_params
        for op, weights in (
            (self._fc2_op, params[num_local_experts:]),
            (self._fc1_op, params[:num_local_experts]),
        ):
            if op is None:
                continue
            leader = weights[0]
            if not is_gtp_param(leader):
                continue

            wgrads = []
            for index in range(num_local_experts):
                shadow = op.__dict__.get(f"weight{index}")
                wgrads.append(getattr(shadow, "main_grad", None))

            # ONE reduce-scatter per group
            leader.finalize_group_grads(wgrads)

            # Restore the non-EGTP post-backward state: an empty slot.
            for index in range(num_local_experts):
                op.__dict__.pop(f"weight{index}", None)

            op.__dict__.pop("_egtp_bwd_weight_group", None)

    def moe_forward(
        self,
        hidden_states: torch.Tensor,
        topk_indices: torch.Tensor,
        topk_weights: torch.Tensor,
        *,
        offload: bool,
    ) -> tuple[torch.Tensor, tuple[CachedForwardChunkActivation, ...]]:
        config = self.config
        transport = self._transport

        ensure_expert_parameters_ready(self._experts)
        self.quantize_expert_weights()
        offsets = config.chunk_offsets
        num_chunks = config.num_token_chunks

        compute_stream = torch.cuda.current_stream(self.device)
        comm_stream = self.comm_stream
        dispatch_ready = [torch.cuda.Event() for _ in range(num_chunks)]
        expert_ready = [torch.cuda.Event() for _ in range(num_chunks)]

        combine_done = torch.cuda.Event()

        dispatched: list[DispatchedChunk | None] = [None] * num_chunks
        saved_chunks: list[CachedForwardChunkActivation | None] = [None] * num_chunks
        output: torch.Tensor | None = None

        inputs_ready = torch.cuda.Event()
        inputs_ready.record(compute_stream)
        comm_stream.wait_event(inputs_ready)

        def issue_dispatch(c: int) -> None:
            """Enqueue one chunk's forward dispatch on the comm stream."""
            rows = slice(offsets[c], offsets[c + 1])
            dispatched[c] = transport.dispatch_forward(
                hidden_states[rows], topk_indices[rows], topk_weights[rows], comm_stream
            )
            dispatch_ready[c].record(comm_stream)

        issue_dispatch(0)
        for c in range(num_chunks):
            if c + 1 < num_chunks:
                issue_dispatch(c + 1)

            compute_stream.wait_event(dispatch_ready[c])
            fc2_output, saved_chunks[c] = self.routed_experts_compute_forward(
                dispatched[c], offload=offload
            )
            # The receive buffer stays alive because fc2_output is a view of it.
            dispatched[c] = None
            expert_ready[c].record(compute_stream)

            comm_stream.wait_event(expert_ready[c])
            handle = saved_chunks[c].dispatch_handle
            chunk_output = transport.combine_forward(fc2_output, handle, comm_stream)
            with torch.cuda.stream(comm_stream):
                if output is None:
                    output = chunk_output.new_empty((offsets[-1], *chunk_output.shape[1:]))
                output[offsets[c] : offsets[c + 1]] = chunk_output

            fc2_output.record_stream(comm_stream)
            transport.record_handle_stream(handle, comm_stream)
            combine_done.record(comm_stream)

        compute_stream.wait_event(combine_done)
        output.record_stream(compute_stream)

        return output, tuple(saved_chunks)

    def moe_backward(
        self, saved_fwd: tuple[CachedForwardChunkActivation, ...], grad_output: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, tuple[torch.Tensor | None, ...]]:
        offsets = self.config.chunk_offsets
        transport = self._transport
        self.begin_wgrad_accumulation()
        num_chunks = len(saved_fwd)

        compute_stream = torch.cuda.current_stream(self.device)
        comm_stream = self.comm_stream
        dispatch_ready = [torch.cuda.Event() for _ in range(num_chunks)]
        expert_ready = [torch.cuda.Event() for _ in range(num_chunks)]
        combine_done = torch.cuda.Event()

        dispatched: list[DispatchedChunk | None] = [None] * num_chunks
        grad_hidden: torch.Tensor | None = None
        grad_probs: torch.Tensor | None = None

        inputs_ready = torch.cuda.Event()
        inputs_ready.record(compute_stream)
        comm_stream.wait_event(inputs_ready)

        def issue_dispatch(c: int) -> None:
            """Enqueue one chunk's backward dispatch on the comm stream."""
            dispatched[c] = transport.dispatch_backward(
                grad_output[offsets[c] : offsets[c + 1]], saved_fwd[c].dispatch_handle, comm_stream
            )
            dispatch_ready[c].record(comm_stream)

        issue_dispatch(num_chunks - 1)
        for c in range(num_chunks - 1, -1, -1):
            if c > 0:
                issue_dispatch(c - 1)

            saved_chunk = saved_fwd[c]
            compute_stream.wait_event(dispatch_ready[c])
            grad_permuted, grad_probs_permuted = self.routed_experts_compute_backward(
                saved_chunk, dispatched[c]
            )
            dispatched[c] = None
            expert_ready[c].record(compute_stream)

            comm_stream.wait_event(expert_ready[c])
            handle = saved_chunk.dispatch_handle
            chunk_hidden, chunk_probs = transport.combine_backward(
                grad_permuted, grad_probs_permuted, handle, comm_stream
            )
            with torch.cuda.stream(comm_stream):
                if grad_hidden is None:
                    grad_hidden = chunk_hidden.new_empty((offsets[-1], *chunk_hidden.shape[1:]))
                    grad_probs = chunk_probs.new_empty((offsets[-1], *chunk_probs.shape[1:]))
                rows = slice(offsets[c], offsets[c + 1])
                grad_hidden[rows] = chunk_hidden
                grad_probs[rows] = chunk_probs

            grad_permuted.record_stream(comm_stream)
            grad_probs_permuted.record_stream(comm_stream)
            transport.record_handle_stream(handle, comm_stream)
            combine_done.record(comm_stream)

            self.run_chunk_wgrad(stream=compute_stream)

        # Combine(0) is enqueued last, so its completion proves every earlier one.
        compute_stream.wait_event(combine_done)
        grad_hidden.record_stream(compute_stream)
        grad_probs.record_stream(compute_stream)

        from transformer_engine.pytorch.ops._common import get_dummy_wgrads_for_params

        wgrad_placeholders = tuple(get_dummy_wgrads_for_params(list(self.expert_params)))

        # this call fires DDP's grad-ready, and DDP's post hook otherwise adds a stale `param.grad`.
        self.end_wgrad_accumulation()

        return grad_hidden, grad_probs, wgrad_placeholders
