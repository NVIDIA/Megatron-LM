# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""ChunkedEP expert execution with caller-owned grouped-GEMM outputs."""

from __future__ import annotations

import weakref
from typing import Any

import torch  # pyright: ignore[reportMissingImports]
import torch.nn as nn  # pyright: ignore[reportMissingImports]

from megatron.lite.primitive import transformer_engine as te
from megatron.lite.primitive.modules.experts import Experts as _BaseExperts
from megatron.lite.primitive.modules.experts import swiglu_with_probs

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


def _caller_owned_dummy_wgrad(
    main_grad: torch.Tensor, weight: torch.Tensor, *, zero: bool
) -> torch.Tensor:
    """Supply MCore's non-None main-thread hook sentinel for grad_added_to_main_grad.
    Avoid TE's process-global cache retaining eager sentinels; only capture uses
    its replay-stable addresses until a graph lifecycle owner is available here.
    """
    if main_grad.is_cuda and torch.cuda.is_current_stream_capturing():
        from transformer_engine.pytorch.module.base import get_dummy_wgrad

        return get_dummy_wgrad(list(main_grad.shape), weight.dtype, zero=zero)
    dummy = torch.empty(tuple(main_grad.shape), dtype=weight.dtype, device=main_grad.device)
    if zero:
        dummy.zero_()
    return dummy.detach()


class _CallerOwnedGroupedLinear(torch.autograd.Function):
    """TE 2.15 BF16 grouped GEMM with caller-owned outputs and alias-safe wgrad."""

    @staticmethod
    def forward(ctx, inp, out, dgrad_out, non_tensor_args, *weights):
        from transformer_engine.pytorch.cpp_extensions import general_grouped_gemm
        from transformer_engine.pytorch.module.base import _2X_ACC_FPROP

        m_splits, wgrad_store, fuse_wgrad_accumulation, activation_dtype = non_tensor_args
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
            ctx.requires_dgrad = inp.requires_grad
            ctx.weights_requires_grad = weights[0].requires_grad
            ctx.origin_weights_overwrite_main_grad = False
            if ctx.weights_requires_grad:
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
        if ctx.weights_requires_grad:
            origin_weights = [ref() for ref in ctx.origin_weight_refs]
            ctx.origin_weight_refs = None
            if any(weight is None for weight in origin_weights):
                raise RuntimeError("Caller-owned grouped GEMM lost an original TE weight")
            main_grads = [func() for func in ctx.main_grad_funcs]
            if any(main_grad is None for main_grad in main_grads):
                raise RuntimeError("Caller-owned grouped GEMM requires prepared main_grad sinks")
            for weight, main_grad in zip(origin_weights, main_grads, strict=True):
                weight.main_grad = main_grad
        wgrad = partial(
            general_grouped_gemm,
            quantization_params=[None] * len(weights),
            out_dtype=activation_dtype,
            layout="NT",
            grad=True,
            m_splits=ctx.m_splits,
            use_bias=False,
            use_split_accumulator=_2X_ACC_WGRAD,
            accumulate=not ctx.origin_weights_overwrite_main_grad,
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
    pending = [value]
    while pending:
        item = pending.pop()
        item_id = id(item)
        if item_id in seen:
            continue
        seen.add(item_id)
        if torch.is_tensor(item):
            if item.is_cuda:
                item.record_stream(stream)
            base = getattr(item, "_base", None)
            if base is not None:
                pending.append(base)
        elif isinstance(item, (dict, list, tuple, set)):
            children = item.values() if isinstance(item, dict) else item
            pending.extend(reversed(tuple(children)))


class ChunkedExperts(_BaseExperts):
    def __init__(self, config, ps, *, delay_wgrad_compute=False, **kwargs):
        self._delay_wgrad_compute = delay_wgrad_compute
        super().__init__(config, ps, **kwargs)
        self._owned_main_grad_aliases = {}

    def _make_linear(self, *args, **kwargs):
        return te.GroupedLinear(
            *args,
            delay_wgrad_compute=self._delay_wgrad_compute,
            fuse_wgrad_accumulation=self._delay_wgrad_compute,
            **kwargs,
        )

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
            raise RuntimeError("Expert wgrad sink must be contiguous with matching shape/device")

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
                raise RuntimeError("Expert immediate/deferred wgrad counts do not match flush")
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
                        raise RuntimeError("Expert delayed wgrad did not reuse its gradient sink")
            store._mlite_immediate_wgrad_contexts = 0
            if store.context is not None and not store.context.empty():
                raise RuntimeError("Expert delayed weight-gradient queue was not drained.")
        self.release_delayed_weight_grad_aliases()

    def forward(
        self,
        x,
        tokens_per_expert,
        permuted_probs=None,
        tokens_per_expert_list=None,
        output_allocation=None,
    ):
        if output_allocation is None:
            # The saved-context route uses the unchanged expert computation.
            return super().forward(x, tokens_per_expert, permuted_probs, tokens_per_expert_list)
        if (
            self.fp8
            or self.fc1_lora is not None
            or self.fc2_lora is not None
            or self.etp_group is not None
            or self.moe_act_recompute
        ):
            raise RuntimeError(
                "Caller-owned ChunkedEP outputs support only BF16 without "
                "FP8, LoRA, ETP, or activation recompute"
            )
        if tokens_per_expert_list is None and tokens_per_expert is None:
            raise RuntimeError("Experts requires token counts in tensor or host-list form")
        splits = (
            tokens_per_expert.tolist()
            if tokens_per_expert_list is None
            else list(tokens_per_expert_list)
        )
        probs = permuted_probs.unsqueeze(-1) if permuted_probs is not None else None
        fc1_out = _caller_owned_grouped_linear(
            self.fc1,
            x,
            splits,
            output_allocation("fc1_output", (x.shape[0], self.fc1.out_features)),
            output_allocation("fc1_dgrad", tuple(x.shape)) if x.requires_grad else None,
        )
        h = swiglu_with_probs(fc1_out, probs, self.swiglu_limit)
        return _caller_owned_grouped_linear(
            self.fc2,
            h,
            splits,
            output_allocation("fc2_output", (h.shape[0], self.fc2.out_features)),
            output_allocation("fc2_dgrad", tuple(h.shape)) if h.requires_grad else None,
        )
