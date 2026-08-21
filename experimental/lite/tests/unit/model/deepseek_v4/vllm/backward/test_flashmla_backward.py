from __future__ import annotations

import importlib.util
import math

import pytest
import torch
import torch.nn.functional as F

from megatron.lite.model.deepseek_v4.vllm.attention import (
    _default_sparse_backward,
)


@pytest.mark.gpus(1)
@pytest.mark.optional
@pytest.mark.skipif(
    not torch.cuda.is_available()
    or importlib.util.find_spec("flash_mla") is None
    or importlib.util.find_spec("cudnn") is None,
    reason="requires CUDA, FlashMLA, and cuDNN DSA",
)
def test_real_flashmla_sparse_backward_matches_reference_direction() -> None:
    from flash_mla import flash_mla_sparse_fwd

    torch.manual_seed(37)
    # Validate the production DS4 training contract.  The earlier five-row
    # diagnostic is not representative of the required 1024-token sequence.
    sequence, physical_kv, heads, dim, topk = 1024, 1024, 64, 576, 128
    scale = dim**-0.5
    q = (
        torch.randn(sequence, heads, dim, device="cuda") / math.sqrt(dim)
    ).bfloat16()
    kv = torch.zeros(physical_kv, dim, device="cuda", dtype=torch.bfloat16)
    kv[:sequence] = (
        torch.randn(sequence, dim, device="cuda") / math.sqrt(dim)
    ).bfloat16()
    indices = torch.full(
        (sequence, topk), -1, device="cuda", dtype=torch.int32
    )
    lengths = torch.arange(
        1, sequence + 1, device="cuda", dtype=torch.int32
    ).clamp_max_(topk)
    for row in range(sequence):
        length = min(row + 1, topk)
        indices[row, :length] = torch.arange(
            row + 1 - length, row + 1, device="cuda", dtype=torch.int32
        )
    sink = torch.zeros(heads, device="cuda", dtype=torch.float32)
    out, _max_logits, lse = flash_mla_sparse_fwd(
        q,
        kv.unsqueeze(1),
        indices.unsqueeze(1),
        scale,
        d_v=512,
        attn_sink=sink,
        topk_length=lengths,
    )[:3]
    grad_out = torch.randn_like(out)
    dq, dkv = _default_sparse_backward(
        q, kv, out, grad_out, lse, sink, indices, scale, lengths
    )
    assert torch.isfinite(dq).all()
    assert torch.isfinite(dkv).all()

    with torch.enable_grad():
        q_ref = q.detach().float().requires_grad_(True)
        kv_ref = kv.detach().float().requires_grad_(True)
        selected = kv_ref.index_select(0, indices.clamp_min(0).long().flatten())
        selected = selected.view(sequence, topk, dim)
        ordinal = torch.arange(topk, device="cuda")
        valid = (ordinal.unsqueeze(0) < lengths.unsqueeze(1)) & (indices >= 0)
        logits = torch.einsum("shd,std->sht", q_ref, selected) * scale
        logits = logits.masked_fill(~valid.unsqueeze(1), float("-inf"))
        logits = torch.cat(
            (logits, sink.view(1, -1, 1).expand(sequence, -1, -1)), dim=-1
        )
        probabilities = torch.softmax(logits, dim=-1)
        reference_out = torch.einsum(
            "sht,std->shd", probabilities[..., :-1], selected[..., :512]
        )
        reference_dq, reference_dkv = torch.autograd.grad(
            reference_out, (q_ref, kv_ref), grad_out.float()
        )
    assert F.cosine_similarity(
        dq.float().flatten(), reference_dq.flatten(), dim=0
    ) > 0.98
    assert F.cosine_similarity(
        dkv[:sequence].float().flatten(),
        reference_dkv[:sequence].flatten(),
        dim=0,
    ) > 0.98
