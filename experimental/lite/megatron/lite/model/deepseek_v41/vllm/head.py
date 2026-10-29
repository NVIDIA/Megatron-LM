# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Native deployment head logits and log-softmax; Lite owns model execution."""

import torch


def log_softmax(logits):
    if not logits.is_cuda:
        return torch.log_softmax(logits.float(), -1)  # CPU derivative reference only.
    from vllm.model_executor.kernels.batch_invariant import native_log_softmax

    return native_log_softmax(logits.float(), -1)


def head_logits(hidden, weight, *, dtype="float32"):
    """The whole FP32 deployment logits, same arithmetic as one head block."""
    if dtype not in ("float32", "bfloat16"):
        raise ValueError(f"Unsupported head dtype {dtype!r}")
    if hidden.is_cuda:
        from .linear import linear

        logits = linear(hidden, weight, fp8=False)
    else:
        logits = torch.nn.functional.linear(hidden.float(), weight.float())
    return _RoundBF16.apply(logits) if dtype == "bfloat16" else logits


def _token_rows(hidden, weight, labels, dtype, temperature, calculate_entropy):
    logits = head_logits(hidden, weight, dtype=dtype)
    if float(temperature) != 1:
        logits = logits / float(temperature)
    distribution = log_softmax(logits.reshape(-1, weight.shape[0]))
    picked = distribution.gather(-1, labels.reshape(-1, 1)).squeeze(-1)
    entropy = -(distribution.exp() * distribution).sum(-1) if calculate_entropy else None
    return picked, entropy


class _FrozenTokenLogProbs(torch.autograd.Function):
    """Same row-wise deployment kernels, with a bounded replay of the frozen head.

    Save only hidden rows and the immutable head, not the T-by-vocabulary
    distribution. Backward replays one row block at a time through the same
    native log-softmax and BF16 GEMM VJP; no cross-block reduction is introduced.
    """

    @staticmethod
    def forward(ctx, hidden, weight, labels, chunk_size, dtype, temperature, calculate_entropy):
        ctx.save_for_backward(hidden, weight, labels)
        ctx.options = chunk_size, dtype, temperature, calculate_entropy
        picked = torch.empty(labels.numel(), device=hidden.device, dtype=torch.float32)
        entropy = torch.empty_like(picked) if calculate_entropy else None
        for start in range(0, hidden.shape[0], chunk_size):
            rows = slice(start, start + chunk_size)
            lp, ent = _token_rows(
                hidden[rows], weight, labels[rows], dtype, temperature, calculate_entropy
            )
            picked[rows] = lp
            if entropy is not None:
                entropy[rows] = ent
        ctx.set_materialize_grads(False)
        return picked, entropy

    @staticmethod
    def backward(ctx, grad_lp, grad_entropy):
        hidden, weight, labels = ctx.saved_tensors
        chunk_size, dtype, temperature, calculate_entropy = ctx.options
        if not ctx.needs_input_grad[0]:
            return (None,) * 7
        dx = torch.zeros_like(hidden)
        if grad_lp is None and grad_entropy is None:
            return dx, None, None, None, None, None, None
        for start in range(0, hidden.shape[0], chunk_size):
            rows = slice(start, start + chunk_size)
            with torch.enable_grad():
                x = hidden[rows].detach().requires_grad_()
                lp, entropy = _token_rows(
                    x, weight, labels[rows], dtype, temperature, calculate_entropy
                )
            outputs, grads = [], []
            if grad_lp is not None:
                outputs.append(lp)
                grads.append(grad_lp[rows])
            if grad_entropy is not None:
                outputs.append(entropy)
                grads.append(grad_entropy[rows])
            dx[rows] = torch.autograd.grad(outputs, x, grads)[0]
        return dx, None, None, None, None, None, None


def token_log_probs(
    hidden,
    weight,
    labels,
    *,
    chunk_size=1024,
    dtype="float32",
    temperature=1.0,
    calculate_entropy=False,
):
    """Exact full-vocabulary LP rows; chunk only the frozen head's token axis.

    ``chunk_size=0`` is the unchunked reference. A trainable head keeps its full
    autograd path, so this optimization never suppresses weight gradients.
    """
    if not isinstance(chunk_size, int) or chunk_size < 0:
        raise ValueError("head_chunk_size must be a nonnegative integer")
    flat = hidden.reshape(-1, hidden.shape[-1])
    labels = labels.reshape(-1)
    if labels.numel() != flat.shape[0]:
        raise ValueError("Every head row requires exactly one label")
    if chunk_size == 0 or weight.requires_grad:
        return _token_rows(flat, weight, labels, dtype, temperature, calculate_entropy)
    return _FrozenTokenLogProbs.apply(
        flat, weight, labels, chunk_size, dtype, temperature, calculate_entropy
    )


class _RoundBF16(torch.autograd.Function):
    """BF16 logits of the deployment head; identity (STE) derivative."""

    @staticmethod
    def forward(ctx, x):
        return x.to(torch.bfloat16)

    @staticmethod
    def backward(ctx, grad):
        return grad.float()
