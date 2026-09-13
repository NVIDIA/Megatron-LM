# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Operator-level numerical test for the MagiAttention core-attention adapter.

This is the attention test to look at before any model E2E: identical global
varlen Q/K/V go through Lite's magi path (runtime key -> dispatch ->
MagiDotProductAttention -> undispatch) and through a per-sequence fp32 SDPA
causal reference. Forward outputs and dQ/dK/dV are compared token-by-token in
global order. Tolerances are anchored to the bf16-SDPA-vs-fp32-SDPA noise
floor so the test stays tight without tracking kernel-version jitter.
"""

from __future__ import annotations

import os

import pytest
import torch
import torch.distributed as dist
import torch.nn.functional as F

# Requires a real MagiAttention install (optional dependency): excluded from
# the standard workflow and driven by tests/run_magi_attention_e2e.sh.
pytestmark = [pytest.mark.gpus(4), pytest.mark.optional]

_SEQ_LENS = [192, 128, 320, 64]
_NUM_HEADS_Q = 8
_NUM_HEADS_KV = 2
_HEAD_DIM = 64
_NOISE_FLOOR_MULTIPLIER = 4.0
_ABS_FLOOR = 1e-3


@pytest.fixture(scope="module", autouse=True)
def _single_node_cuda_dist():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for the MagiAttention operator test.")

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size not in {2, 4}:
        pytest.skip("The MagiAttention operator test requires torchrun with 2 or 4 ranks.")

    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("LOCAL_RANK", "0")
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29533")
    os.environ.pop("MAGI_ATTENTION_SDPA_BACKEND", None)
    os.environ.pop("MAGI_ATTENTION_FA4_BACKEND", None)
    default_backend = "fa4" if torch.cuda.get_device_capability()[0] >= 10 else "ffa"
    os.environ.setdefault("MAGI_ATTENTION_KERNEL_BACKEND", default_backend)

    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    created_pg = False
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl", init_method="env://")
        created_pg = True
    yield
    if created_pg and dist.is_initialized():
        dist.destroy_process_group()


def _seeded_randn(generator, total_tokens: int, heads: int, dtype):
    """Rank-identical tensors: every rank draws the same sequence from the seed."""
    return torch.randn(
        (total_tokens, heads, _HEAD_DIM),
        generator=generator,
        device="cuda",
        dtype=torch.float32,
    ).to(dtype)


def _sdpa_reference(q, k, v, seq_lens, dtype):
    """Per-sequence causal SDPA in the requested dtype, GQA via head repeat."""
    group = _NUM_HEADS_Q // _NUM_HEADS_KV
    outputs = []
    offset = 0
    for seq_len in seq_lens:
        qs = q[offset : offset + seq_len].to(dtype).transpose(0, 1).unsqueeze(0)
        ks = (
            k[offset : offset + seq_len]
            .to(dtype)
            .repeat_interleave(group, dim=1)
            .transpose(0, 1)
            .unsqueeze(0)
        )
        vs = (
            v[offset : offset + seq_len]
            .to(dtype)
            .repeat_interleave(group, dim=1)
            .transpose(0, 1)
            .unsqueeze(0)
        )
        out = F.scaled_dot_product_attention(qs, ks, vs, is_causal=True)
        outputs.append(out.squeeze(0).transpose(0, 1))
        offset += seq_len
    return torch.cat(outputs, dim=0)


def _max_err(candidate, reference):
    return float((candidate.float() - reference.float()).abs().max())


def _assert_within_noise_floor(name, candidate, noisy_ref, exact_ref):
    err = _max_err(candidate, exact_ref)
    floor = _max_err(noisy_ref, exact_ref)
    limit = max(_NOISE_FLOOR_MULTIPLIER * floor, _ABS_FLOOR)
    assert torch.isfinite(candidate.float()).all(), f"{name}: non-finite values"
    assert err <= limit, (
        f"{name}: max err vs fp32 SDPA = {err:.3e} exceeds {limit:.3e} "
        f"(bf16 SDPA noise floor = {floor:.3e})"
    )


@pytest.mark.parametrize(
    "overlap_degree",
    [1, None],  # None = dynamic mode: the overlap solver picks the staging
    ids=["static-degree1", "dynamic"],
)
def test_magi_core_attention_matches_sdpa_reference(overlap_degree):
    pytest.importorskip("magi_attention")
    if os.environ.get("MAGI_ATTENTION_KERNEL_BACKEND", "ffa").lower() == "fa4":
        pytest.importorskip("flash_attn_cute")

    from megatron.lite.primitive.modules.attention.magi import (
        MagiAttentionConfig,
        MagiDotProductAttention,
        build_magi_attention_runtime_key,
        dispatch_magi_attention_tensor,
        undispatch_magi_attention_tensor,
    )
    from megatron.lite.primitive.utils.packed_seq import PackedSeqParams

    total_tokens = sum(_SEQ_LENS)
    cp_group = dist.group.WORLD
    assert total_tokens % cp_group.size() == 0

    cu_seqlens = torch.tensor(
        [0] + list(torch.tensor(_SEQ_LENS).cumsum(0).tolist()), dtype=torch.int32, device="cuda"
    )
    runtime_key = build_magi_attention_runtime_key(
        cu_seqlens,
        num_heads_q=_NUM_HEADS_Q,
        num_heads_kv=_NUM_HEADS_KV,
        head_dim=_HEAD_DIM,
        cp_group=cp_group,
        config=MagiAttentionConfig(chunk_size=64, overlap_degree=overlap_degree),
    )
    assert int(getattr(runtime_key, "pad_size", 0)) == 0

    # Every "global" tensor must be bit-identical on all CP ranks; one shared
    # seeded generator drawn in a fixed order guarantees that.
    generator = torch.Generator(device="cuda").manual_seed(20260717)

    # The dispatch/undispatch pair is pure data movement: bitwise round trip.
    probe = _seeded_randn(generator, total_tokens, _NUM_HEADS_KV, torch.bfloat16)
    restored = undispatch_magi_attention_tensor(
        dispatch_magi_attention_tensor(probe, runtime_key), runtime_key
    )
    assert torch.equal(restored, probe), "dispatch->undispatch round trip is not exact"

    # --- magi path: dispatch -> core attention -> undispatch, with autograd ---
    q = _seeded_randn(generator, total_tokens, _NUM_HEADS_Q, torch.bfloat16).requires_grad_(True)
    k = _seeded_randn(generator, total_tokens, _NUM_HEADS_KV, torch.bfloat16).requires_grad_(True)
    v = _seeded_randn(generator, total_tokens, _NUM_HEADS_KV, torch.bfloat16).requires_grad_(True)
    grad_out = _seeded_randn(generator, total_tokens, _NUM_HEADS_Q, torch.float32)

    core_attn = MagiDotProductAttention(head_dim=_HEAD_DIM)
    psp = PackedSeqParams(
        qkv_format="magi",
        cu_seqlens_q=cu_seqlens,
        cu_seqlens_kv=cu_seqlens,
        max_seqlen_q=max(_SEQ_LENS),
        max_seqlen_kv=max(_SEQ_LENS),
        magi_runtime_key=runtime_key,
    )
    out_local = core_attn(
        dispatch_magi_attention_tensor(q, runtime_key),
        dispatch_magi_attention_tensor(k, runtime_key),
        dispatch_magi_attention_tensor(v, runtime_key),
        packed_seq_params=psp,
    )
    out_magi = undispatch_magi_attention_tensor(out_local, runtime_key)
    dq_magi, dk_magi, dv_magi = torch.autograd.grad(
        (out_magi.float() * grad_out).sum(), (q, k, v)
    )

    # Undispatched activations must be identical on every rank.
    out_bcast = out_magi.detach().clone()
    dist.broadcast(out_bcast, src=dist.get_process_group_ranks(cp_group)[0], group=cp_group)
    assert torch.equal(out_bcast, out_magi.detach()), "undispatched output differs across ranks"

    # --- references: fp32 SDPA (exact) and bf16 SDPA (noise floor), same leaves ---
    refs = {}
    for dtype in (torch.float32, torch.bfloat16):
        q_ref = q.detach().clone().requires_grad_(True)
        k_ref = k.detach().clone().requires_grad_(True)
        v_ref = v.detach().clone().requires_grad_(True)
        out_ref = _sdpa_reference(q_ref, k_ref, v_ref, _SEQ_LENS, dtype)
        dq, dk, dv = torch.autograd.grad(
            (out_ref.float() * grad_out).sum(), (q_ref, k_ref, v_ref)
        )
        refs[dtype] = {"out": out_ref.detach(), "dq": dq, "dk": dk, "dv": dv}

    exact, noisy = refs[torch.float32], refs[torch.bfloat16]
    _assert_within_noise_floor("forward output", out_magi.detach(), noisy["out"], exact["out"])
    _assert_within_noise_floor("dQ", dq_magi, noisy["dq"], exact["dq"])
    _assert_within_noise_floor("dK", dk_magi, noisy["dk"], exact["dk"])
    _assert_within_noise_floor("dV", dv_magi, noisy["dv"], exact["dv"])

    # Make any rank-local assertion failure visible to the whole torchrun job.
    ok = torch.tensor([1.0], device="cuda")
    dist.all_reduce(ok, op=dist.ReduceOp.MIN, group=cp_group)
    assert ok.item() == 1.0
