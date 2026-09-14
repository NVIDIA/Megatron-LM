# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""ChunkedEP expert execution with caller-owned grouped-GEMM outputs."""

from __future__ import annotations

import weakref
from contextlib import AbstractContextManager, nullcontext
from typing import Any, Callable

import torch  # pyright: ignore[reportMissingImports]
import torch.distributed as dist  # pyright: ignore[reportMissingImports]
import torch.nn as nn  # pyright: ignore[reportMissingImports]

from megatron.lite.primitive import transformer_engine as te
from megatron.lite.primitive.modules.experts import Experts as _BaseExperts
from megatron.lite.primitive.modules.experts import _AllReduceETP, swiglu_with_probs
from megatron.lite.primitive.modules.lora import (
    LoraConfig,
    SharedGroupedLinearLoRA,
    normalize_lora_config,
)
from megatron.lite.primitive.parallel import ParallelState
from megatron.lite.primitive.recompute import CheckpointWithoutOutput
from megatron.lite.primitive.utils import ensure_divisible

__all__ = ["ChunkedExperts"]


def _validate_caller_owned_buffer(
    name: str,
    buffer: torch.Tensor,
    expected_shape: tuple[int, ...],
    dtype: torch.dtype,
    device: torch.device,
) -> None:
    """Match TE's explicit-output contract before exposing an arena view to autograd."""
    if (
        tuple(buffer.shape) != expected_shape
        or buffer.dtype != dtype
        or buffer.device != device
        or not buffer.is_contiguous()
        or buffer.requires_grad
    ):
        raise RuntimeError(
            f"Caller-owned grouped GEMM {name} must be contiguous, non-grad, and match shape/dtype/device"
        )


def _tensor_byte_ranges_overlap(left: torch.Tensor, right: torch.Tensor) -> bool:
    """Return whether two contiguous tensor views overlap in addressable bytes."""
    if left.numel() == 0 or right.numel() == 0:
        return False
    left_start = left.data_ptr()
    left_end = left_start + left.numel() * left.element_size()
    right_start = right.data_ptr()
    right_end = right_start + right.numel() * right.element_size()
    return left_start < right_end and right_start < left_end


def _record_immediate_wgrad_context(store: Any) -> None:
    count = getattr(store, "_mlite_immediate_wgrad_contexts", 0)
    if not isinstance(count, int) or count < 0:
        raise RuntimeError("Invalid immediate delayed-wgrad context count")
    store._mlite_immediate_wgrad_contexts = count + 1


def _caller_owned_dummy_is_capturing(main_grad: torch.Tensor) -> bool:
    """Whether a dummy wgrad must retain CUDA-graph replay pointer stability."""
    return main_grad.is_cuda and bool(torch.cuda.is_current_stream_capturing())


def _caller_owned_dummy_wgrad(
    main_grad: torch.Tensor, weight: torch.Tensor, *, zero: bool
) -> torch.Tensor:
    """Return the custom-DDP hook sentinel without retaining eager allocations.

    MCore requires a non-None gradient to keep the parameter hook on the main
    backward thread when ``grad_added_to_main_grad`` is set.  TE's provider is
    process-global, which is correct for its native module but unnecessarily
    retains caller-owned ChunkedEP sentinels across eager steps.  CUDA graph
    capture is the narrow exception: TE's keyed cache supplies a replay-stable
    address until a graph lifecycle owner is available here.
    """
    if _caller_owned_dummy_is_capturing(main_grad):
        from transformer_engine.pytorch.module.base import get_dummy_wgrad

        return get_dummy_wgrad(list(main_grad.shape), weight.dtype, zero=zero)
    dummy = torch.empty(tuple(main_grad.shape), dtype=weight.dtype, device=main_grad.device)
    if zero:
        dummy.zero_()
    return dummy.detach()


class _CallerOwnedGroupedLinear(torch.autograd.Function):
    """TE 2.15 BF16 grouped-GEMM internals with explicit arena-owned outputs.

    The enclosing helper deliberately retains TE's public module lifecycle.
    This class changes only TE 2.15's two ``torch.empty`` allocation sites.
    """

    @staticmethod
    def forward(ctx, inp, out, dgrad_out, non_tensor_args, *weights):
        from transformer_engine.pytorch.cpp_extensions import general_grouped_gemm
        from transformer_engine.pytorch.module.base import _2X_ACC_FPROP

        (m_splits, is_first_microbatch, wgrad_store, fuse_wgrad_accumulation, activation_dtype) = (
            non_tensor_args
        )
        if inp.dtype != torch.bfloat16 or activation_dtype != torch.bfloat16:
            raise RuntimeError("Caller-owned grouped GEMM requires BF16 activation")
        if not fuse_wgrad_accumulation or not wgrad_store.delay_wgrad_compute():
            raise RuntimeError("Caller-owned grouped GEMM requires TE delayed fused wgrad")
        _validate_caller_owned_buffer(
            "output", out, (sum(m_splits), weights[0].shape[0]), activation_dtype, inp.device
        )
        if inp.requires_grad:
            if dgrad_out is None:
                raise RuntimeError("Caller-owned grouped GEMM requires dgrad output for grad input")
            _validate_caller_owned_buffer(
                "dgrad output", dgrad_out, tuple(inp.shape), activation_dtype, inp.device
            )
        elif dgrad_out is not None:
            raise RuntimeError("Caller-owned grouped GEMM received dgrad output for non-grad input")
        inputmats = list(torch.split(inp.reshape(-1, inp.shape[-1]), m_splits))
        general_grouped_gemm(
            list(weights),
            inputmats,
            [out],
            [None] * len(weights),
            activation_dtype,
            single_output=True,
            m_splits=m_splits,
            use_split_accumulator=_2X_ACC_FPROP,
        )
        if ctx is not None:
            ctx.m_splits = list(m_splits)
            ctx.inp_shape = inp.shape
            ctx.dgrad_out = dgrad_out
            ctx.wgrad_store = wgrad_store
            ctx.is_first_microbatch = is_first_microbatch
            ctx.fuse_wgrad_accumulation = fuse_wgrad_accumulation
            ctx.requires_dgrad = inp.requires_grad
            ctx.weights_requires_grad = weights[0].requires_grad
            ctx.origin_weights_overwrite_main_grad = False
            if ctx.fuse_wgrad_accumulation and ctx.weights_requires_grad:
                # Match TE2.15: only a weak reference preserves MCore's Python
                # attributes, and FSDP is allowed to materialize main_grad after
                # forward but before backward.
                ctx.origin_weight_refs = [weakref.ref(weight) for weight in weights]
                ctx.origin_weights_overwrite_main_grad = getattr(
                    weights[0], "overwrite_main_grad", False
                )
                if hasattr(weights[0], "__fsdp_param__"):
                    ctx.main_grad_funcs = [weight.get_main_grad for weight in weights]
                else:
                    ctx.main_grad_funcs = [
                        lambda index=index: weights[index].main_grad
                        for index in range(len(weights))
                    ]
            ctx.save_for_backward(inp, *weights)
        return out.view(-1, *inp.shape[1:-1], out.shape[-1])

    @staticmethod
    def backward(ctx, grad_output):
        from functools import partial

        from transformer_engine.pytorch.cpp_extensions import general_grouped_gemm
        from transformer_engine.pytorch.module.base import _2X_ACC_DGRAD, _2X_ACC_WGRAD

        inp, *weights = ctx.saved_tensors
        activation_dtype = inp.dtype
        grad_view = grad_output.contiguous().view(-1, grad_output.shape[-1])
        grad_mats = list(torch.split(grad_view, ctx.m_splits))
        dgrad = None
        dgrad_out = None
        if ctx.requires_dgrad:
            dgrad_out = ctx.dgrad_out
            if dgrad_out is None:
                raise RuntimeError("Caller-owned grouped GEMM lost its dgrad output")
            dgrad = dgrad_out.view(-1, inp.shape[-1])
        origin_weights = [None] * len(weights)
        main_grads = [None] * len(weights)
        if ctx.fuse_wgrad_accumulation and ctx.weights_requires_grad:
            origin_weights = [ref() for ref in ctx.origin_weight_refs]
            ctx.origin_weight_refs = None
            if any(weight is None for weight in origin_weights):
                raise RuntimeError("Caller-owned grouped GEMM lost an original TE weight")
            main_grads = [func() for func in ctx.main_grad_funcs]
            if any(main_grad is None for main_grad in main_grads):
                raise RuntimeError("Caller-owned grouped GEMM requires prepared main_grad sinks")
            for weight, main_grad in zip(origin_weights, main_grads, strict=True):
                weight.main_grad = main_grad
        if ctx.is_first_microbatch is not None:
            accumulate = ctx.fuse_wgrad_accumulation and not ctx.is_first_microbatch
        else:
            accumulate = ctx.fuse_wgrad_accumulation
        wgrad = partial(
            general_grouped_gemm,
            quantization_params=[None] * len(weights),
            out_dtype=activation_dtype,
            layout="NT",
            grad=True,
            m_splits=ctx.m_splits,
            use_bias=False,
            use_split_accumulator=_2X_ACC_WGRAD,
            accumulate=accumulate and not ctx.origin_weights_overwrite_main_grad,
        )
        inputmats = list(torch.split(inp.reshape(-1, inp.shape[-1]), ctx.m_splits))
        immediate_wgrad = (
            ctx.weights_requires_grad
            and dgrad is not None
            and _tensor_byte_ranges_overlap(grad_view, dgrad)
        )
        if immediate_wgrad:
            # A caller-owned dgrad view overlaps the live grad-output bytes.
            # Wgrad must consume them before Dgrad writes the same storage.
            wgrad(inputmats, grad_mats, main_grads)
            _record_immediate_wgrad_context(ctx.wgrad_store)
        if dgrad is not None:
            general_grouped_gemm(
                list(weights),
                grad_mats,
                [dgrad],
                [None] * len(weights),
                activation_dtype,
                layout="NN",
                single_output=True,
                m_splits=ctx.m_splits,
                grad=True,
                use_split_accumulator=_2X_ACC_DGRAD,
            )
        if ctx.weights_requires_grad:
            if not immediate_wgrad:
                ctx.wgrad_store.put([inputmats, grad_mats, main_grads], wgrad)
        wgrad_returns = []
        for weight, main_grad in zip(origin_weights, main_grads, strict=True):
            if weight is not None and hasattr(weight, "grad_added_to_main_grad"):
                weight.grad_added_to_main_grad = True
                wgrad_returns.append(
                    _caller_owned_dummy_wgrad(
                        main_grad, weight, zero=getattr(weight, "zero_out_wgrad", False)
                    )
                )
            else:
                wgrad_returns.append(None)
        return (
            dgrad.view(ctx.inp_shape) if dgrad is not None else None,
            None,
            None,
            None,
            *wgrad_returns,
        )


def _caller_owned_grouped_linear(
    linear: Any,
    x: torch.Tensor,
    m_splits: list[int],
    out: torch.Tensor,
    dgrad_out: torch.Tensor | None,
) -> torch.Tensor:
    """Reuse TE2.15 ``GroupedLinear.forward`` lifecycle around its two buffers."""
    if (
        getattr(linear, "fp8", False)
        or getattr(linear, "use_bias", False)
        or getattr(linear, "return_bias", False)
        or getattr(linear, "save_original_input", False)
    ):
        raise RuntimeError("Caller-owned grouped GEMM supports only BF16 bias-free experts")
    if len(m_splits) != linear.num_gemms:
        raise RuntimeError("Caller-owned grouped GEMM split count does not match TE module")
    prepared_x = linear.prepare_forward(x, num_gemms=linear.num_gemms)
    try:
        weights = linear._get_weight_tensors()
        linear._get_bias_tensors()
        quantizers = linear._get_quantizers()
        if any(item is not None for group in quantizers for item in group):
            raise RuntimeError("Caller-owned grouped GEMM does not support TE quantizers")
        non_tensor_args = (
            list(m_splits),
            None,
            linear.wgrad_store,
            linear.fuse_wgrad_accumulation,
            linear.activation_dtype,
        )
        if torch.is_grad_enabled():
            return _CallerOwnedGroupedLinear.apply(
                prepared_x, out, dgrad_out, non_tensor_args, *weights
            )
        return _CallerOwnedGroupedLinear.forward(
            None, prepared_x, out, dgrad_out, non_tensor_args, *weights
        )
    finally:
        linear.end_forward()


def _record_cuda_tensor_tree_stream(value: Any, stream: Any) -> None:
    """Record every nested CUDA tensor and its view bases on one stream."""
    seen: set[int] = set()

    def record(item: Any) -> None:
        item_id = id(item)
        if item_id in seen:
            return
        if torch.is_tensor(item):
            seen.add(item_id)
            if item.is_cuda:
                item.record_stream(stream)
            base = getattr(item, "_base", None)
            if base is not None:
                record(base)
            return
        if isinstance(item, dict):
            seen.add(item_id)
            for nested in item.values():
                record(nested)
            return
        if isinstance(item, (list, tuple, set)):
            seen.add(item_id)
            for nested in item:
                record(nested)

    record(value)


class ChunkedExperts(nn.Module):
    def __init__(
        self,
        config: Any,
        ps: ParallelState,
        *,
        fp8: bool = False,
        moe_act_recompute: bool = False,
        delay_wgrad_compute: bool = False,
        lora_config: LoraConfig | dict | None = None,
    ):
        super().__init__()
        self.num_local_experts = ensure_divisible(config.num_experts, ps.ep_size)
        self.fp8 = fp8
        self.moe_act_recompute = moe_act_recompute
        self._delay_wgrad_compute = delay_wgrad_compute
        self._owned_main_grad_aliases: dict[int, weakref.ReferenceType[torch.Tensor]] = {}
        # ETP is untested and its backward disagrees with lora.py's; refuse it.
        if ps.etp_size > 1:
            raise NotImplementedError(f"etp_size={ps.etp_size} unsupported; use 1.")
        self.etp_group = ps.etp_group if ps.etp_size > 1 else None
        self.swiglu_limit = float(getattr(config, "swiglu_limit", 0.0) or 0.0)
        self.fc1 = te.GroupedLinear(
            self.num_local_experts,
            config.hidden_size,
            config.moe_intermediate_size * 2 // ps.etp_size,
            bias=False,
            params_dtype=torch.bfloat16,
            delay_wgrad_compute=delay_wgrad_compute,
            fuse_wgrad_accumulation=delay_wgrad_compute,
        )
        self.fc2 = te.GroupedLinear(
            self.num_local_experts,
            config.moe_intermediate_size // ps.etp_size,
            config.hidden_size,
            bias=False,
            params_dtype=torch.bfloat16,
            delay_wgrad_compute=delay_wgrad_compute,
            fuse_wgrad_accumulation=delay_wgrad_compute,
        )
        lora = normalize_lora_config(lora_config)
        self.fc1_lora: SharedGroupedLinearLoRA | None = None
        self.fc2_lora: SharedGroupedLinearLoRA | None = None
        if lora.enabled and lora.targets_module("linear_fc1"):
            self.fc1_lora = SharedGroupedLinearLoRA(
                self.num_local_experts,
                config.hidden_size,
                config.moe_intermediate_size * 2 // ps.etp_size,
                lora.rank,
                alpha=lora.alpha,
                dropout=lora.dropout,
            )
        if lora.enabled and lora.targets_module("linear_fc2"):
            self.fc2_lora = SharedGroupedLinearLoRA(
                self.num_local_experts,
                config.moe_intermediate_size // ps.etp_size,
                config.hidden_size,
                lora.rank,
                alpha=lora.alpha,
                dropout=lora.dropout,
            )
        if ps.tp_size > 1 and ps.ep_size == 1 and ps.etp_size == 1:
            tp_group = ps.tp_group
            for module in (self.fc1, self.fc2, self.fc1_lora, self.fc2_lora):
                if module is None:
                    continue
                for param in module.parameters():

                    def _ar(grad, g=tp_group):
                        dist.all_reduce(grad, op=dist.ReduceOp.SUM, group=g)
                        return grad

                    param.register_hook(_ar)

    def _delayed_weight_parameters(self):
        for linear in (self.fc1, self.fc2):
            for idx in range(linear.num_gemms):
                yield getattr(linear, f"weight{idx}")

    @staticmethod
    def _validate_weight_grad_sink(param: nn.Parameter, sink: torch.Tensor) -> None:
        if (
            not torch.is_tensor(sink)
            or sink.shape != param.shape
            or sink.device != param.device
            or not sink.is_contiguous()
        ):
            raise RuntimeError(
                "Expert delayed weight-gradient sink must be a contiguous tensor "
                "with matching shape and device"
            )

    def _prepare_delayed_weight_grad_sinks(self) -> None:
        """Prepare the sink selected by TE for each delayed expert wgrad."""
        if not self._delay_wgrad_compute:
            return
        for param in self._delayed_weight_parameters():
            # Frozen TE saves this accessor only for its FSDP parameter wrapper,
            # then resolves it during delayed backward and writes main_grad back.
            if hasattr(param, "__fsdp_param__"):
                if not callable(getattr(param, "get_main_grad", None)):
                    raise RuntimeError("Expert FSDP parameter requires a callable get_main_grad")
                continue

            param_id = id(param)
            owned_ref = self._owned_main_grad_aliases.get(param_id)
            owned = None if owned_ref is None else owned_ref()
            main_grad = getattr(param, "main_grad", None)
            if owned_ref is not None:
                if main_grad is not owned:
                    self._owned_main_grad_aliases.pop(param_id, None)
                elif param.grad is owned:
                    self._validate_weight_grad_sink(param, owned)
                    continue
                else:
                    delattr(param, "main_grad")
                    self._owned_main_grad_aliases.pop(param_id, None)
                    main_grad = None

            if main_grad is not None:
                self._validate_weight_grad_sink(param, main_grad)
                continue
            if param.grad is None:
                param.grad = torch.zeros_like(param, memory_format=torch.preserve_format)
            self._validate_weight_grad_sink(param, param.grad)
            param.main_grad = param.grad
            self._owned_main_grad_aliases[param_id] = weakref.ref(param.grad)

    def release_delayed_weight_grad_aliases(self) -> None:
        """Drop owned TE aliases without changing standard parameter gradients."""
        for param in self._delayed_weight_parameters():
            owned_ref = self._owned_main_grad_aliases.pop(id(param), None)
            owned = None if owned_ref is None else owned_ref()
            if owned is not None and getattr(param, "main_grad", None) is owned:
                delattr(param, "main_grad")

    def flush_delayed_weight_grads(self, *, num_contexts: int, stream: Any | None = None) -> None:
        """Execute queued TE wgrads directly into their selected gradient sinks."""
        for linear in (self.fc1, self.fc2):
            store = linear.wgrad_store
            if not store.delay_wgrad_compute():
                raise RuntimeError("Expert delayed weight gradients are not enabled.")
            if linear.use_bias:
                raise RuntimeError("Chunked EP expert grouped linears must not use bias.")
            immediate_contexts = getattr(store, "_mlite_immediate_wgrad_contexts", 0)
            if (
                not isinstance(immediate_contexts, int)
                or immediate_contexts < 0
                or immediate_contexts > num_contexts
            ):
                raise RuntimeError(
                    "Expert delayed wgrad immediate/deferred context accounting "
                    "does not match the requested flush"
                )
            for _ in range(num_contexts - immediate_contexts):
                if store.context is None or store.context.empty():
                    raise RuntimeError("Expert delayed weight-gradient queue is empty.")
                result, tensors = store.pop()
                if stream is not None:
                    _record_cuda_tensor_tree_stream((result, tensors), stream)
                _, _grad_biases, _ = result
                weight_grads = tensors[2]
                for idx, grad in enumerate(weight_grads):
                    param = getattr(linear, f"weight{idx}")
                    sink = getattr(param, "main_grad", None)
                    if sink is None:
                        raise RuntimeError(
                            "Expert delayed wgrad produced no selected gradient sink"
                        )
                    self._validate_weight_grad_sink(param, sink)
                    self._validate_weight_grad_sink(param, grad)
                    if grad.data_ptr() != sink.data_ptr():
                        raise RuntimeError(
                            "Expert delayed wgrad did not reuse its selected gradient sink"
                        )
            store._mlite_immediate_wgrad_contexts = 0
            if store.context is not None and not store.context.empty():
                raise RuntimeError("Expert delayed weight-gradient queue was not drained.")
        self.release_delayed_weight_grad_aliases()

    def forward(
        self,
        x: torch.Tensor,
        tokens_per_expert: torch.Tensor | None,
        permuted_probs: torch.Tensor | None = None,
        tokens_per_expert_list: list[int] | None = None,
        activation_allocation: Callable[[], AbstractContextManager[None]] | None = None,
        output_allocation: Callable[[str, tuple[int, int]], torch.Tensor] | None = None,
    ) -> torch.Tensor:
        if tokens_per_expert_list is None:
            if tokens_per_expert is None:
                raise RuntimeError("Experts requires token counts in tensor or host-list form")
            m_splits = tokens_per_expert.tolist()
        else:
            m_splits = list(tokens_per_expert_list)
        pad_mask = None
        if self.fp8:
            x, permuted_probs, m_splits, pad_mask = self._fp8_pad(x, permuted_probs, m_splits)

        etp_real_len = x.shape[0]
        if self.etp_group is not None:
            max_len = torch.tensor([etp_real_len], device=x.device, dtype=torch.int64)
            dist.all_reduce(max_len, op=dist.ReduceOp.MAX, group=self.etp_group)
            max_len = int(max_len.item())
            if etp_real_len < max_len:
                x = torch.cat(
                    [
                        x,
                        torch.zeros(
                            max_len - etp_real_len, x.shape[1], dtype=x.dtype, device=x.device
                        ),
                    ],
                    dim=0,
                )
                if permuted_probs is not None:
                    permuted_probs = torch.cat(
                        [
                            permuted_probs,
                            torch.zeros(
                                max_len - etp_real_len, dtype=permuted_probs.dtype, device=x.device
                            ),
                        ],
                        dim=0,
                    )
                m_splits = list(m_splits)
                m_splits[-1] += max_len - etp_real_len

        probs = permuted_probs.unsqueeze(-1) if permuted_probs is not None else None
        caller_owned_outputs = output_allocation is not None
        if caller_owned_outputs and (
            self.fp8
            or self.fc1_lora is not None
            or self.fc2_lora is not None
            or self.etp_group is not None
            or self.moe_act_recompute
        ):
            raise RuntimeError(
                "Caller-owned ChunkedEP outputs support only BF16 Qwen3 without "
                "FP8, LoRA, ETP, or activation recompute"
            )
        allocation_scope = nullcontext if activation_allocation is None else activation_allocation
        with allocation_scope():
            if caller_owned_outputs:
                fc1_out = _caller_owned_grouped_linear(
                    self.fc1,
                    x,
                    m_splits,
                    output_allocation("fc1_output", (x.shape[0], self.fc1.out_features)),
                    (output_allocation("fc1_dgrad", tuple(x.shape)) if x.requires_grad else None),
                )
            else:
                fc1_out = self.fc1(x, m_splits)
            if self.fc1_lora is not None:
                fc1_out = fc1_out + self.fc1_lora(x, m_splits)
            if self.moe_act_recompute and probs is not None:
                act_ckpt = CheckpointWithoutOutput(preserve_rng_state=True)
                h = act_ckpt.checkpoint(swiglu_with_probs, fc1_out, probs, self.swiglu_limit)
            else:
                act_ckpt = None
                h = swiglu_with_probs(fc1_out, probs, self.swiglu_limit)
        if caller_owned_outputs:
            out = _caller_owned_grouped_linear(
                self.fc2,
                h,
                m_splits,
                output_allocation("fc2_output", (h.shape[0], self.fc2.out_features)),
                (output_allocation("fc2_dgrad", tuple(h.shape)) if h.requires_grad else None),
            )
        else:
            out = self.fc2(h, m_splits)
        if self.fc2_lora is not None:
            out = out + self.fc2_lora(h, m_splits)
        if act_ckpt is not None:
            act_ckpt.discard_output_and_register_recompute(out)

        if self.etp_group is not None:
            out = _AllReduceETP.apply(out, self.etp_group)
            out = out[:etp_real_len]

        if pad_mask is not None:
            out = out[pad_mask]
        return out

    _fp8_pad = staticmethod(_BaseExperts._fp8_pad)
