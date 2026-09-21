# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""MSA flex forward vs the official MiniMax-AI/MSA SM100 kernel (train/inference consistency).

Optional: needs ``fmha_sm100`` importable (clone of https://github.com/MiniMax-AI/MSA,
``PYTHONPATH=<MSA>/python``) and an SM100 GPU. The official kernel is forward-only,
so it serves as an oracle for the *inference* semantics of per-token block selection
(token-level causal inside the local block included).

Both kernels run in bf16 on identical inputs and indices; each is compared against the
fp32 masked-dense reference with the same indices (kernel-error scale), and against each other.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

pytestmark = [pytest.mark.gpus(1, min_architecture="blackwell"), pytest.mark.optional]
DEV = "cuda"
KERNEL_REL = 1e-2


def _official():
    return pytest.importorskip("fmha_sm100")


def _rel(a, b):
    return ((a.float() - b.float()).abs().max() / b.float().abs().max().clamp_min(1e-30)).item()


def _selection(B, H_idx, S, K, blk):
    iq = torch.randn(B, H_idx, S, 64, device=DEV)
    ik = torch.randn(B, 1, S, 64, device=DEV)
    pos = torch.arange(S, device=DEV).unsqueeze(0).expand(B, -1)
    n_kv = -(-S // blk)
    scores = torch.matmul(iq, ik.transpose(-1, -2))
    scores = scores.masked_fill(torch.arange(S, device=DEV)[None, None, None, :] > pos[:, None, :, None], float("-inf"))
    scores = F.pad(scores, (0, n_kv * blk - S), value=float("-inf")).view(B, H_idx, S, n_kv, blk).amax(-1)
    scores.scatter_(-1, (pos // blk)[:, None, :, None].expand(-1, H_idx, -1, 1), float("inf"))
    top_scores, top_idx = scores.topk(min(K, n_kv), dim=-1)
    return top_idx.masked_fill(top_scores == float("-inf"), -1).to(torch.int32), pos


def _masked_reference(q, k, v, idx, pos, blk, scale):
    B, H_idx, S, _ = idx.shape
    n_kv = -(-S // blk)
    keep = torch.zeros(B, H_idx, S, n_kv + 1, dtype=torch.bool, device=DEV)
    keep.scatter_(-1, idx.masked_fill(idx < 0, n_kv).long(), True)
    keep = keep[..., :n_kv].repeat_interleave(blk, dim=-1)[..., :S].repeat_interleave(q.shape[1] // H_idx, dim=1)
    keep &= ~(torch.arange(S, device=DEV)[None, None, None, :] > pos[:, None, :, None])
    n_rep = q.shape[1] // k.shape[1]
    q, k, v = q.float(), k.float().repeat_interleave(n_rep, 1), v.float().repeat_interleave(n_rep, 1)
    return torch.softmax((q @ k.transpose(-1, -2) * scale).masked_fill(~keep, float("-inf")), dim=-1) @ v


def _ascending_with_tail_pad(idx: torch.Tensor) -> torch.Tensor:
    big = idx.masked_fill(idx < 0, torch.iinfo(torch.int32).max)
    srt = big.sort(-1).values
    return srt.masked_fill(srt == torch.iinfo(torch.int32).max, -1).contiguous()


# M3 shapes only (GQA ratio 16, top-16): in the validated env (cutlass-dsl 4.6.2 + libs-cu13 4.5.2) the official
# CuTe kernel compiles for ratio 16 but fails NVVM compilation for ratios 4/8 -- environment, not semantics.
@pytest.mark.parametrize("B,Hq,Hkv,S,K", [(1, 64, 4, 4096, 16), (2, 64, 4, 2048, 16), (1, 64, 4, 8192, 16)])
def test_flex_vs_official_msa_forward(B, Hq, Hkv, S, K):
    m = _official()
    from megatron.lite.primitive.kernels import msa_kernels as mk

    torch.manual_seed(0)
    D, blk = 128, 128
    idx, pos = _selection(B, Hkv, S, K, blk)
    q = torch.randn(B, Hq, S, D, device=DEV, dtype=torch.bfloat16)
    k = torch.randn(B, Hkv, S, D, device=DEV, dtype=torch.bfloat16)
    v = torch.randn_like(k)
    scale = D**-0.5

    o_flex = mk.msa_core_attention(q, k, v, idx, pos, block_size=blk, scale=scale, backend="flex")
    o_ref = _masked_reference(q, k, v, idx, pos, blk, scale)

    # official: varlen [total, H, D], per-token block lists [Hkv, total_q, K] ascending, -1 tail
    q_v = q.permute(0, 2, 1, 3).reshape(B * S, Hq, D).contiguous()
    k_v = k.permute(0, 2, 1, 3).reshape(B * S, Hkv, D).contiguous()
    v_v = v.permute(0, 2, 1, 3).reshape(B * S, Hkv, D).contiguous()
    q2k = _ascending_with_tail_pad(idx).permute(1, 0, 2, 3).reshape(Hkv, B * S, K).contiguous()
    cu = torch.arange(0, (B + 1) * S, S, device=DEV, dtype=torch.int32)
    row_ptr, q_ind, schedule = m.build_k2q_csr(
        q2k, cu, cu, blk, total_k=B * S, max_seqlen_k=S, max_seqlen_q=S,
        total_rows=B * (S // blk), qhead_per_kv=Hq // Hkv, return_schedule=True,
    )
    out = m.sparse_atten_func(
        q_v, k_v, v_v, row_ptr, q_ind, K, cu_seqlens_q=cu, cu_seqlens_k=cu, max_seqlen_q=S, max_seqlen_k=S,
        blk_kv=blk, causal=True, softmax_scale=scale, schedule=schedule,
    )
    o_off = (out[0] if isinstance(out, tuple) else out).view(B, S, Hq, D).permute(0, 2, 1, 3)

    r_flex, r_off, r_x = _rel(o_flex, o_ref), _rel(o_off, o_ref), _rel(o_flex, o_off)
    print(f"B={B} Hq={Hq} Hkv={Hkv} S={S} K={K}: flex vs fp32 {r_flex:.2e} | official vs fp32 {r_off:.2e} | flex vs official {r_x:.2e}")
    assert r_flex < KERNEL_REL and r_off < KERNEL_REL and r_x < 2 * KERNEL_REL
