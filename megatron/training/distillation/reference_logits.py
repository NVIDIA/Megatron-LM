# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""FP16 reference logits with an explicit tail and a stable coarse KL loss.

Format v3 stores K values for K selected vocabulary indices: the tail log-sum-exp in
slot 0, and the K-1 non-anchor logit differences in the remaining slots.
The first selected index is the anchor, whose relative logit is exactly zero.
Masked selected slots use -inf. The tail includes *all* unselected indices.
No probabilities, or differences from one, are stored.

Format v4 can instead store normalized top-K log-probabilities on tokens where
their measured round-trip error is smaller (coarse KL by default, or tail error).
A redundant sign bit in slot 1
tags the encoding; format-aware decoding preserves the same K-value footprint.
"""

import torch
import torch.distributed as dist
import torch.nn.functional as F

# Bound temporary FP32 vocabulary buffers independently of sequence length.
_VOCAB_WORKSPACE_ELEMENTS = 1024 * 1024


def _working_dtype(logits: torch.Tensor) -> torch.dtype:
    return torch.float64 if logits.dtype == torch.float64 else torch.float32


def _all_reduce(tensor, op, tp_size, tp_group):
    if tp_size > 1:
        dist.all_reduce(tensor, op=op, group=tp_group)
    return tensor


def stable_log_softmax(logits: torch.Tensor) -> torch.Tensor:
    """Normalize log weights without adding a dominant exponential to its tail."""
    maximum, pivot = logits.max(dim=-1, keepdim=True)
    shifted = logits - maximum
    others = shifted.scatter(-1, pivot, -torch.inf)
    log_normalizer = F.softplus(torch.logsumexp(others, dim=-1, keepdim=True))
    return shifted - log_normalizer


def decode_reference_logits(values: torch.Tensor) -> torch.Tensor:
    """Decode K stored values into K+1 unnormalized log weights.

    The returned order is anchor, other selected indices, aggregated tail.
    Conversion to FP32 happens only on the small selected-vocabulary tensor.
    """
    values = values.to(_working_dtype(values))
    return torch.cat((torch.zeros_like(values[..., :1]), values[..., 1:], values[..., :1]), -1)


def reference_logprobs(values: torch.Tensor) -> torch.Tensor:
    """Return normalized log probabilities, with the aggregated tail last."""
    return stable_log_softmax(decode_reference_logits(values))


def _local_indices(indices, valid, local_vocab_size, tp_rank):
    local = indices - local_vocab_size * tp_rank
    owned = valid & (local >= 0) & (local < local_vocab_size)
    # A dedicated dummy column makes masked scatter indices harmless, including
    # duplicate masked indices. Clamping to a real vocabulary entry is unsafe.
    scatter_indices = torch.where(owned, local, local_vocab_size)
    return local.clamp(0, local_vocab_size - 1), owned, scatter_indices


def coarsen_logits(
    logits: torch.Tensor,
    indices: torch.Tensor,
    valid: torch.Tensor,
    tp_size: int = 1,
    tp_rank: int = 0,
    tp_group: dist.ProcessGroup | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Gather reference logits and directly reduce the excluded vocabulary.

    All TP ranks receive the same K+1 log weights and absolute anchor logit.
    This is a numerical primitive for the saver and the custom KL forward;
    differentiating the distributed reductions directly is not supported.
    """
    vocab = logits.size(-1)
    dtype = _working_dtype(logits)
    local, owned, scatter = _local_indices(indices, valid, vocab, tp_rank)
    selected = logits.gather(-1, local).to(dtype).masked_fill(~owned, 0)
    _all_reduce(selected, dist.ReduceOp.SUM, tp_size, tp_group)
    anchor = selected[..., :1].clone()
    relative = (selected - anchor).masked_fill(~valid, -torch.inf)

    flat_logits = logits.reshape(-1, vocab)
    flat_anchor = anchor.reshape(-1, 1)
    flat_scatter = scatter.reshape(-1, indices.size(-1))
    local_lse = torch.empty_like(flat_anchor)
    rows_per_chunk = max(1, _VOCAB_WORKSPACE_ELEMENTS // (vocab + 1))
    for start in range(0, flat_logits.size(0), rows_per_chunk):
        end = start + rows_per_chunk
        chunk = torch.empty(
            (flat_logits[start:end].size(0), vocab + 1), dtype=dtype, device=logits.device
        )
        chunk[:, :vocab] = flat_logits[start:end].to(dtype) - flat_anchor[start:end]
        chunk[:, vocab] = -torch.inf
        chunk.scatter_(-1, flat_scatter[start:end], -torch.inf)
        local_lse[start:end] = torch.logsumexp(chunk[:, :vocab], -1, keepdim=True)

    tail = local_lse.reshape_as(anchor)
    if tp_size > 1:
        maximum = tail.clone()
        _all_reduce(maximum, dist.ReduceOp.MAX, tp_size, tp_group)
        # Empty tails (including fully selected vocabularies) have -inf on
        # every rank. Avoid evaluating -inf - -inf in the reduction.
        shift = maximum.masked_fill(~torch.isfinite(maximum), 0)
        total = (tail - shift).exp()
        _all_reduce(total, dist.ReduceOp.SUM, tp_size, tp_group)
        tail = shift + total.log()
    return torch.cat((relative, tail), -1), anchor


def encode_reference_logits(
    coarse_logits: torch.Tensor, keep: torch.Tensor, dtype: torch.dtype = torch.float16
) -> torch.Tensor:
    """Pack K+1 log weights into K values, folding top-P exclusions into the tail.

    ``keep[..., 0]`` must be true: the highest-logit selected token is the
    reference. Top-P selection must happen before rounding to the cache dtype.
    """
    folded = _fold_excluded_logits(coarse_logits, keep)
    return torch.cat((folded[..., -1:], folded[..., 1:-1]), -1).to(dtype)


def _fold_excluded_logits(coarse_logits, keep):
    selected = coarse_logits[..., :-1]
    excluded = selected.masked_fill(keep, -torch.inf)
    tail = torch.logsumexp(torch.cat((coarse_logits[..., -1:], excluded), -1), -1, keepdim=True)
    return torch.cat((selected.masked_fill(~keep, -torch.inf), tail), -1)


def uses_reference_logits(values: torch.Tensor) -> torch.Tensor:
    """Read the per-token v4 encoding tag without allocating a saved mode tensor.

    For K >= 2 the second value's sign is redundant: both log probabilities
    and reference-logit gaps are nonpositive. Positive (including +0 and
    +inf) selects reference logits, negative selects normalized log-probs.
    K=1 always uses the v3 reference encoding and needs no tag.
    """
    if values.size(-1) == 1:
        return torch.ones_like(values[..., 0], dtype=torch.bool)
    return ~torch.signbit(values[..., 1])


def _normalized_topk_to_reference(values):
    mass = values.exp().sum(-1, keepdim=True)
    # A cache candidate with mass > 1 is rejected by the encoder. Clamping
    # here also makes the unselected decode branch safe for mixed tensors.
    tail = torch.log1p(-mass.clamp(max=1))
    return torch.cat((values, tail), -1) - values[..., :1]


def decode_adaptive_logits(values: torch.Tensor) -> torch.Tensor:
    """Decode a v4 mixed cache into K+1 reference logits, without changing K."""
    if values.size(-1) == 1:
        return decode_reference_logits(values)
    reference_mode = uses_reference_logits(values)
    raw = values.to(_working_dtype(values)).clone()
    raw[..., 1] = -raw[..., 1].abs()  # Restore the magnitude's original negative sign.
    reference = decode_reference_logits(raw)
    # Use a harmless one-hot placeholder on reference-mode rows, so their
    # slot-0 tail LSE cannot produce invalid log-prob arithmetic in this branch.
    probabilities = raw.masked_fill(reference_mode.unsqueeze(-1), -torch.inf)
    probabilities[..., 0] = torch.where(reference_mode, 0, raw[..., 0])
    probability_reference = _normalized_topk_to_reference(probabilities)
    return torch.where(reference_mode.unsqueeze(-1), reference, probability_reference)


def adaptive_logprobs(values: torch.Tensor) -> torch.Tensor:
    """Return v4 normalized log-probabilities in selected-token, tail order."""
    return stable_log_softmax(decode_adaptive_logits(values))


def encode_adaptive_logits(
    coarse_logits: torch.Tensor,
    keep: torch.Tensor,
    dtype: torch.dtype = torch.float16,
    metric: str = "kl",
) -> torch.Tensor:
    """Choose the smaller round-trip error for each token at cache creation.

    The teacher distribution is available here, so no confidence threshold or
    trained selector is needed. Both candidates are decoded in the same FP32
    arithmetic as training. Scoring uses FP64 only on K+1 categories, with no
    FP64 vocabulary buffer and no persistent storage beyond K cache values.
    Reference logits win ties and invalid normalized-probability candidates.
    ``metric='tail'`` minimizes absolute tail error, equivalent to relative
    tail error for a fixed positive teacher tail. It need not minimize KL.
    """
    if metric not in ("kl", "tail"):
        raise ValueError(f"Unsupported encoding selection metric: {metric}")
    reference = encode_reference_logits(coarse_logits, keep, dtype)
    if reference.size(-1) == 1:
        return reference
    target = _fold_excluded_logits(coarse_logits.float(), keep)
    probabilities = stable_log_softmax(target)[..., :-1].to(dtype)
    decoded_probability = _normalized_topk_to_reference(probabilities.float())
    decoded_reference = decode_reference_logits(reference.float())
    if metric == "kl":
        probability_error, _ = _macro_kl(decoded_probability.double(), target.double())
        reference_error, _ = _macro_kl(decoded_reference.double(), target.double())
    else:
        true_tail = stable_log_softmax(target.double())[..., -1].exp()
        probability_tail = stable_log_softmax(decoded_probability)[..., -1].exp().double()
        reference_tail = stable_log_softmax(decoded_reference)[..., -1].exp().double()
        probability_error = (probability_tail - true_tail).abs()
        reference_error = (reference_tail - true_tail).abs()
    valid_probability = probabilities.float().exp().sum(-1) <= 1
    use_probability = (
        valid_probability
        & torch.isfinite(probability_error)
        & (probability_error < reference_error)
    )
    reference_mode = ~use_probability
    selected = torch.where(reference_mode.unsqueeze(-1), reference, probabilities)
    sign = torch.where(
        reference_mode, torch.ones_like(selected[..., 1]), -torch.ones_like(selected[..., 1])
    )
    selected[..., 1] = torch.copysign(selected[..., 1].abs(), sign)
    return selected


def _macro_kl(student, teacher):
    """Return a nonnegative coarse KL and cancellation-resistant logit gradient."""
    log_q = stable_log_softmax(student)
    log_p = stable_log_softmax(teacher)
    q, p = log_q.exp(), log_p.exp()
    finite = torch.isfinite(student) & torch.isfinite(teacher)
    delta = torch.where(finite, student - teacher, 0)
    d = delta - (p * delta).sum(-1, keepdim=True)
    same_support = (torch.isfinite(student) == torch.isfinite(teacher)).all(-1, keepdim=True)
    near = same_support & (d.abs().amax(-1, keepdim=True) <= 0.5)
    # Restrict this branch's exponentials even when a distant example will
    # select the general branch below. A plain expm1(d) can overflow otherwise.
    small_d = d.clamp(-0.5, 0.5)
    near_ratio = small_d - torch.log1p((p * torch.expm1(small_d)).sum(-1, keepdim=True))
    far_ratio = torch.where(torch.isneginf(log_q) & torch.isneginf(log_p), 0, log_q - log_p)
    ratio = torch.where(near, near_ratio, far_ratio)

    # Compute q-p from its log ratio. Both exponential arguments are <= 0,
    # and the q branch also handles p=0 without 0*exp(+inf).
    positive = ratio.clamp(min=0)
    negative = ratio.clamp(max=0)
    u = torch.where(ratio >= 0, -q * torch.expm1(-positive), p * torch.expm1(negative))
    g = u - q * u.sum(-1, keepdim=True)
    pivot = q.argmax(-1, keepdim=True)
    g = g.scatter(-1, pivot, 0)
    g = g.scatter(-1, pivot, -g.sum(-1, keepdim=True))

    # KL = sum_i [q_i - p_i - p_i*log(q_i/p_i)]. Each term is
    # nonnegative in exact arithmetic. A series preserves its quadratic
    # behavior at equality instead of subtracting first-order quantities.
    finite_positive = positive.masked_fill(~torch.isfinite(positive), 0)
    pos_term = q * (-torch.expm1(-finite_positive) - finite_positive * torch.exp(-finite_positive))
    pos_term = torch.where(p == 0, q, pos_term)
    neg_term = p * (torch.expm1(negative) - negative)
    neg_term = torch.where(p == 0, 0, neg_term)
    x = ratio.clamp(-0.05, 0.05)
    series = (
        p
        * x.square()
        * (0.5 + x * (1 / 6 + x * (1 / 24 + x * (1 / 120 + x * (1 / 720 + x / 5040)))))
    )
    terms = torch.where(ratio.abs() < 0.05, series, torch.where(ratio >= 0, pos_term, neg_term))
    return terms.sum(-1), g


class _ReferenceLogitsKL(torch.autograd.Function):
    @staticmethod
    def forward(ctx, logits, values, indices, tp_size, tp_rank, tp_group, cache_format_version):
        decoder = decode_adaptive_logits if cache_format_version == 4 else decode_reference_logits
        teacher = decoder(values).to(_working_dtype(logits))
        valid = torch.isfinite(teacher[..., :-1])
        student, anchor = coarsen_logits(logits, indices, valid, tp_size, tp_rank, tp_group)
        loss, gradient = _macro_kl(student, teacher)
        ctx.save_for_backward(logits, indices, valid, gradient, anchor, student[..., -1:])
        ctx.tp_size, ctx.tp_rank, ctx.tp_group = tp_size, tp_rank, tp_group
        return loss

    @staticmethod
    def backward(ctx, grad_output):
        logits, indices, valid, gradient, anchor, tail = ctx.saved_tensors
        vocab = logits.size(-1)
        dtype = _working_dtype(logits)
        _, owned, scatter = _local_indices(indices, valid, vocab, ctx.tp_rank)
        macro_grad = gradient * grad_output.unsqueeze(-1)
        selected_grad = macro_grad[..., :-1].masked_fill(~owned, 0).clone()
        # The anchor is filled from the global sum of all other vocabulary
        # gradients below. This retains its tiny gradient at high confidence.
        selected_grad[..., 0] = 0
        output = torch.empty_like(logits, memory_format=torch.contiguous_format)
        flat_logits = logits.reshape(-1, vocab)
        flat_output = output.reshape(-1, vocab)
        flat_anchor = anchor.reshape(-1, 1)
        flat_tail = tail.reshape(-1, 1)
        flat_tail_grad = macro_grad[..., -1:].reshape(-1, 1)
        flat_selected_grad = selected_grad.reshape(-1, indices.size(-1))
        flat_scatter = scatter.reshape_as(flat_selected_grad)
        other_sum = torch.empty_like(flat_anchor)
        rows_per_chunk = max(1, _VOCAB_WORKSPACE_ELEMENTS // (vocab + 1))
        for start in range(0, flat_logits.size(0), rows_per_chunk):
            end = start + rows_per_chunk
            chunk = torch.zeros(
                (flat_logits[start:end].size(0), vocab + 1), dtype=dtype, device=logits.device
            )
            # A missing tail has no gradient; shifting by zero avoids infs.
            tail_lse = flat_tail[start:end].masked_fill(~torch.isfinite(flat_tail[start:end]), 0)
            weights = (flat_logits[start:end].to(dtype) - flat_anchor[start:end] - tail_lse).exp()
            chunk[:, :vocab] = weights * flat_tail_grad[start:end]
            chunk.scatter_(-1, flat_scatter[start:end], flat_selected_grad[start:end])
            other_sum[start:end] = chunk[:, :vocab].sum(-1, keepdim=True)
            flat_output[start:end] = chunk[:, :vocab]
        _all_reduce(other_sum, dist.ReduceOp.SUM, ctx.tp_size, ctx.tp_group)
        anchor_owned = owned[..., :1]
        anchor_local = (indices[..., :1] - ctx.tp_rank * vocab).clamp(0, vocab - 1)
        previous = output.gather(-1, anchor_local)
        anchor_grad = torch.where(anchor_owned, -other_sum.reshape_as(anchor), previous)
        output.scatter_(-1, anchor_local, anchor_grad.to(logits.dtype))
        return output, None, None, None, None, None, None


def reference_logits_kl_div(
    student_logits: torch.Tensor,
    teacher_values: torch.Tensor,
    teacher_indices: torch.Tensor,
    tp_size: int = 1,
    tp_rank: int = 0,
    tp_group: dist.ProcessGroup | None = None,
    cache_format_version: int = 3,
) -> torch.Tensor:
    """Compute KL on selected tokens plus one tail, replicated across TP ranks.

    Backward returns each rank's local vocabulary gradient. The teacher only
    constrains total tail mass; its gradient is distributed within that tail
    according to the student's own conditional probabilities. The caller must
    not sum this replicated loss across TP ranks a second time.
    """
    if cache_format_version not in (3, 4):
        raise ValueError(f"Unsupported reference-logits format_version={cache_format_version}")
    return _ReferenceLogitsKL.apply(
        student_logits,
        teacher_values,
        teacher_indices,
        tp_size,
        tp_rank,
        tp_group,
        cache_format_version,
    )
