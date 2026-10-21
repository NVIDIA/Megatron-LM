# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Mega attention core: forward bits against the vLLM layer, owned VJP against FP64.

Forward gate: ``mega_attention`` output bytes (FP8 data and packed UE8M0 scales)
equal the vLLM ``DeepseekV4MegaAttnAttention.forward_mqa`` prefill of the same
inputs (caches written by the inference insert ops) and its decode rows, on the
model's attention shape (64 heads, d 512, window 128, top-k 512, ratios 0/2/1).

Backward gate (tolerances fixed before the first run): against an FP64 reference
VJP that shares no code with the backward (vLLM's permutations, complex RoPE,
dense softmax with the sink as an extra logit, autograd), with the STE contract
of the module docstring, every gradient has relative L2 <= 1e-2 and cosine >=
0.9999. Reasoning: the kernel rounds P, dS, dO and the rotated q to bf16 (2**-9
relative) and accumulates in FP32 over <= 640 slots, about 4e-3 relative L2; a
1e-2 bar still fails a 2% scale error and a dropped sink.
"""
import math
from types import SimpleNamespace

import pytest
import torch

pytestmark = [
    pytest.mark.gpus(1, min_architecture="blackwell"),
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA"),
]

WINDOW, TOP_K, HEADS, DIM = 128, 512, 64, 512
REL_L2, COSINE = 1e-2, 0.9999


@pytest.fixture(autouse=True)
def _batch_invariant(monkeypatch):
    monkeypatch.setenv("VLLM_BATCH_INVARIANT", "1")
    from vllm.model_executor.kernels.batch_invariant import require_batch_invariant_kernel

    require_batch_invariant_kernel()


def _cos_sin(max_pos, device):
    inv_freq = 1.0 / (10000 ** (torch.arange(0, 64, 2, device=device).float() / 64))
    freqs = torch.outer(torch.arange(max_pos, device=device).float(), inv_freq)
    return torch.cat([freqs.cos(), freqs.sin()], -1).contiguous()


def _topk(positions, ratio, generator, compressed):
    """Document-local top-k as the BI indexer returns it: ascending, -1 at the end."""
    device = positions.device
    visible = torch.div(positions + 1, ratio, rounding_mode="floor")
    scores = torch.rand(positions.shape[0], max(compressed, 1), device=device, generator=generator)
    column = torch.arange(scores.shape[1], device=device)
    scores = scores.masked_fill(column >= visible[:, None], -1.0)
    width = min(TOP_K, scores.shape[1])
    values, picked = scores.topk(width, dim=1)
    picked = torch.where(values >= 0, picked, torch.iinfo(torch.int32).max)
    picked = picked.sort(dim=1).values
    out = torch.full((positions.shape[0], TOP_K), -1, dtype=torch.int32, device=device)
    out[:, :width] = torch.where(picked == torch.iinfo(torch.int32).max, -1, picked).int()
    return out


def _inputs(length, ratio, seed=0, docs=None):
    device = torch.device("cuda")
    generator = torch.Generator(device=device).manual_seed(seed)
    bounds = [0, length] if docs is None else docs
    positions = torch.cat(
        [torch.arange(b - a, device=device) for a, b in zip(bounds[:-1], bounds[1:])]
    )
    randn = lambda *shape: torch.randn(*shape, device=device, generator=generator)
    inputs = SimpleNamespace(
        q=randn(length, HEADS, DIM).bfloat16(),
        swa_kv=randn(length, DIM).bfloat16(),
        latent=randn(length, DIM).bfloat16() if ratio else None,
        topk=None,
        positions=positions,
        sink=randn(HEADS).float(),
        cos_sin=_cos_sin(max(length, 2) + 1, device),
        ratio=ratio,
        docs=bounds,
    )
    if ratio:
        inputs.topk = _topk(positions, ratio, generator, max(bounds[1:]) // ratio)
    return inputs


def _ours(x, **kwargs):
    from megatron.lite.model.deepseek_v41.vllm.mega_attention import mega_attention

    # The interface takes the boundary-token rows of the per-token latent.
    latent = None if x.latent is None else x.latent[(x.positions + 1) % x.ratio == 0]
    return mega_attention(
        x.q,
        x.swa_kv,
        latent,
        x.topk,
        x.positions,
        x.sink,
        compress_ratio=x.ratio,
        cos_sin_cache=x.cos_sin,
        cu_seqlens=None if not torch.is_tensor(x.docs) and len(x.docs) == 2 else x.docs,
        **kwargs,
    )


# ---- the vLLM inference layer ---------------------------------------------


def _vllm_layer(x):
    """forward_mqa(start, end) of one request on caches the inference insert ops wrote."""
    from dataclasses import replace

    from vllm.config import CacheConfig, set_current_vllm_config
    from vllm.forward_context import ForwardContext, override_forward_context
    from vllm.models.deepseek_v41.common.ops.fused_compress_quant_cache import rope_quant_insert
    from vllm.models.deepseek_v41.nvidia.flash_mla_mega_attn import (
        DeepseekV4MegaAttnAttention,
        alloc_mega_attn_output,
    )
    from vllm.models.deepseek_v41.nvidia.flashmla import (
        DeepseekSparseSWAFlashMLAMetadataBuilder,
        DeepseekV4FlashMLAAttention,
    )
    from vllm.models.deepseek_v41.sparse_mla import DeepseekV4FlashMLAMetadataBuilder
    from vllm.v1.attention.backend import CommonAttentionMetadata
    from vllm.v1.attention.backends.mla.sparse_swa import DeepseekV4SWACache
    from vllm.v1.kv_cache_interface import MLAAttentionSpec
    from vllm.v1.worker.workspace import init_workspace_manager

    device = torch.device("cuda")
    length, ratio, block = x.q.shape[0], x.ratio, 32
    config = SimpleNamespace(
        model_config=SimpleNamespace(
            hf_config=SimpleNamespace(
                sliding_window=WINDOW,
                compress_ratios=[0, 0, ratio, ratio] if ratio else [0],
                index_topk=TOP_K,
            ),
            max_model_len=length,
        ),
        scheduler_config=SimpleNamespace(max_num_batched_tokens=length, max_num_seqs=3),
        parallel_config=SimpleNamespace(
            decode_context_parallel_size=1, prefill_context_parallel_size=1
        ),
        speculative_config=None,
        compilation_config=SimpleNamespace(static_forward_context={}),
    )
    with set_current_vllm_config(config):
        swa = DeepseekV4SWACache(
            head_dim=DIM,
            window_size=WINDOW,
            dtype=torch.uint8,
            prefix="swa",
            cache_config=CacheConfig(cache_dtype="nvfp4_ds_mla"),
            backend_cls=DeepseekV4FlashMLAAttention.swa_backend_cls,
            block_size=block,
            packed_bytes_per_token=528,
            packed_page_alignment=512,
        )
    builder = DeepseekSparseSWAFlashMLAMetadataBuilder(
        swa.get_kv_cache_spec(config), [swa.prefix], config, device
    )
    pages = (length + block - 1) // block
    table = torch.arange(pages, dtype=torch.int32, device=device).view(1, pages)
    packed = torch.zeros(pages, block * 528, device=device, dtype=torch.uint8)
    swa.kv_cache = packed.view(pages, block, 528)
    positions = torch.arange(length, device=device)
    # The layer's own fused insert: Q half is a no-op at 64 interleaved heads.
    torch.ops._C.fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert(
        x.q,
        x.swa_kv,
        packed,
        positions,
        positions,
        x.cos_sin,
        0,
        1e-6,
        block,
        False,
        True,
        False,
        True,
    )

    attention = DeepseekV4MegaAttnAttention.__new__(DeepseekV4MegaAttnAttention)
    torch.nn.Module.__init__(attention)
    attention.compress_ratio = ratio
    attention.compressed_cache_prefix = "compressed" if ratio else None
    attention.swa_cache_layer = swa
    attention.window_size = WINDOW
    attention.max_num_batched_tokens = length
    attention.topk_indices_buffer = torch.full(
        (length, TOP_K), -1, device=device, dtype=torch.int32
    )
    compressed_builder = compressed_table = None
    if ratio:
        per_page = 128 // ratio
        compressed_pages = (length // ratio + per_page - 1) // per_page + 1
        compressed_table = torch.arange(compressed_pages, device=device, dtype=torch.int32).view(
            1, compressed_pages
        )
        compressed = torch.zeros(compressed_pages, per_page, 288, device=device, dtype=torch.uint8)
        rope_quant_insert(x.latent, positions, x.cos_sin, compressed, positions // ratio, ratio)
        attention.is_kv_source = True
        attention.kv_cache = compressed
        spec = MLAAttentionSpec(
            block_size=128,
            num_kv_heads=1,
            head_size=DIM,
            dtype=torch.uint8,
            tokens_per_state=ratio,
            state_content_bytes=288,
            cache_dtype_str="nvfp4_ds_mla",
            model_version="deepseek_v4",
        )
        compressed_builder = DeepseekV4FlashMLAMetadataBuilder(spec, ["compressed"], config, device)
    attention.scale = DIM**-0.5
    attention.attn_sink = x.sink
    attention.padded_heads = HEADS
    attention.n_wv_group = HEADS // 8
    attention.rotary_emb = SimpleNamespace(cos_sin_cache=x.cos_sin)
    init_workspace_manager(device)

    def run(start, end):
        common = CommonAttentionMetadata(
            query_start_loc=torch.tensor([0, end - start], device=device).int(),
            query_start_loc_cpu=torch.tensor([0, end - start]).int(),
            seq_lens=torch.tensor([end], device=device, dtype=torch.int32),
            seq_lens_cpu_upper_bound=torch.tensor([end], dtype=torch.int32),
            num_reqs=1,
            num_actual_tokens=end - start,
            max_query_len=end - start,
            max_seq_len=end,
            block_table_tensor=table,
            slot_mapping=torch.arange(start, end, device=device, dtype=torch.int64),
            causal=True,
        )
        metadata = builder.build(0, common)
        everything = {swa.prefix: metadata}
        if compressed_builder is not None:
            everything["compressed"] = compressed_builder.build(
                0, replace(common, block_table_tensor=compressed_table)
            )
        if x.topk is not None:
            attention.topk_indices_buffer[: end - start] = x.topk[start:end]
        out = alloc_mega_attn_output(end - start, attention.n_wv_group, device)
        with override_forward_context(ForwardContext({}, everything, {})):
            attention.forward_mqa(x.q[start:end], None, positions[start:end], out)
        return metadata.num_decodes, out

    return run


def _assert_rows_equal(ours, data, scale, rows):
    assert data.view(torch.uint8).count_nonzero() > data.numel() // 2
    assert torch.equal(ours.data[rows].view(torch.uint8), data.view(torch.uint8)), rows
    assert torch.equal(ours.scale[rows], scale), rows


@pytest.mark.parametrize("ratio", [0, 2, 1])
@pytest.mark.parametrize("length", [129, 700, 8320])
def test_forward_matches_vllm_prefill_and_decode(ratio, length):
    from vllm.v1.worker.workspace import reset_workspace_manager

    x = _inputs(length, ratio, seed=length + ratio)
    with torch.no_grad():
        ours = _ours(x)
    run = _vllm_layer(x)
    try:
        _, full = run(0, length)
        _assert_rows_equal(ours, full.data, full.scale, slice(0, length))
        split = length // 3
        for start, end in ((0, split), (split, length)):
            _, chunk = run(start, end)
            _assert_rows_equal(ours, chunk.data, chunk.scale, slice(start, end))
        for position in sorted({0, 1, 127, 128, length // 2, length - 1}):
            decodes, row = run(position, position + 1)
            assert decodes == 1
            _assert_rows_equal(ours, row.data, row.scale, slice(position, position + 1))
    finally:
        torch.accelerator.synchronize()
        reset_workspace_manager()


@pytest.mark.parametrize("ratio", [0, 2, 1])
def test_packed_documents_match_single_documents(ratio):
    docs = [0, 300, 301, 1500, 1700]
    x = _inputs(docs[-1], ratio, seed=7, docs=docs)
    with torch.no_grad():
        packed = _ours(x)
        for a, b in zip(docs[:-1], docs[1:]):
            one = SimpleNamespace(**vars(x))
            for name in ("q", "swa_kv", "latent", "topk", "positions"):
                value = getattr(x, name)
                setattr(one, name, None if value is None else value[a:b])
            one.docs = [0, b - a]
            single = _ours(one)
            _assert_rows_equal(packed, single.data, single.scale, slice(a, b))


# ---- FP64 reference VJP ----------------------------------------------------


class _Visible(torch.autograd.Function):
    """Value of the forward, gradient of the surrogate (identity STE)."""

    @staticmethod
    def forward(ctx, visible, surrogate):
        return visible

    @staticmethod
    def backward(ctx, grad):
        return None, grad


def _rotate(x, cos_sin, positions, sign):
    """Complex GPT-J rotation of the last 64 dims by ``sign * theta(position)``."""
    cs = cos_sin[positions].double()
    angle = torch.complex(cs[..., :32], sign * cs[..., 32:])
    angle = angle.reshape(positions.shape + (1,) * (x.dim() - positions.dim() - 1) + (32,))
    pairs = torch.view_as_complex(x[..., -64:].reshape(*x.shape[:-1], 32, 2).contiguous())
    rotated = torch.view_as_real(pairs * angle).reshape(*x.shape[:-1], 64)
    return torch.cat([x[..., :-64], rotated], -1)


def _visible_kv(x):
    """The dequantized KV the forward reads: (SWA rows [S, 512], compressed rows [N, 512])."""
    from megatron.lite.model.deepseek_v41.vllm import mega_attention as mega

    length = x.q.shape[0]
    plan = mega._plan([0, length], length, 0, length, x.ratio, x.q.device)
    with torch.no_grad():
        *_, kv, _, _ = mega._forward(
            x.q,
            x.swa_kv,
            x.latent,
            x.topk,
            x.positions,
            x.sink,
            x.cos_sin,
            x.ratio,
            WINDOW,
            TOP_K,
            plan,
        )
    kv = kv[0].double()
    return kv[plan.n : plan.n + length], kv[: plan.n]


def _reference(x, grad_fused, *, chunk=128):
    """FP64 VJP of the contract surrogate for one document (dq fused, dswa, dlatent, dsink)."""
    from vllm.models.deepseek_v41.common.ops.fused_layout import (
        o_fused_permutation,
        q_fused_permutation,
    )

    length, ratio, device = x.q.shape[0], x.ratio, x.q.device
    q_perm = q_fused_permutation(HEADS, DIM).to(device)
    o_perm = o_fused_permutation(8, DIM).to(device)
    q_nat = torch.empty(length, HEADS * DIM, dtype=torch.float64, device=device)
    q_nat[:, q_perm] = x.q.reshape(length, -1).double()
    q_nat = q_nat.view(length, HEADS, DIM).requires_grad_()
    grad = torch.empty(length, 8, 8 * DIM, dtype=torch.float64, device=device)
    grad[:, :, o_perm] = grad_fused.double()
    grad = grad.view(length, HEADS, DIM)
    swa = x.swa_kv.double().requires_grad_()
    latent = None if x.latent is None else x.latent.double().requires_grad_()
    sink = x.sink.double().requires_grad_()
    swa_vis, comp_vis = _visible_kv(x)
    positions = x.positions
    q_vis_all = _rotate(q_nat.detach(), x.cos_sin, positions, 1).bfloat16().double()
    scale = DIM**-0.5
    outputs = []
    for start in range(0, length, chunk):
        rows = torch.arange(start, min(start + chunk, length), device=device)
        q_sur = _rotate(q_nat[rows], x.cos_sin, positions[rows], 1)
        q_r = _Visible.apply(q_vis_all[rows], q_sur)
        lo = (rows - WINDOW + 1).clamp(min=0)
        swa_ids = lo[:, None] + torch.arange(WINDOW, device=device)
        swa_ok = swa_ids <= rows[:, None]
        swa_ids = swa_ids.clamp(max=length - 1)
        keys = [_Visible.apply(swa_vis[swa_ids], _rotate(swa[swa_ids], x.cos_sin, swa_ids, 1))]
        valid = [swa_ok]
        if ratio:
            count = torch.div(rows + 1, ratio, rounding_mode="floor").clamp(max=TOP_K)
            ids = x.topk[rows].long()
            ok = (ids >= 0) & (torch.arange(TOP_K, device=device) < count[:, None])
            ids = ids.clamp(min=0)
            tokens = ids * ratio + ratio - 1
            keys.append(
                _Visible.apply(comp_vis[ids], _rotate(latent[tokens], x.cos_sin, ids * ratio, 1))
            )
            valid.append(ok)
        k = torch.cat(keys, 1)
        ok = torch.cat(valid, 1)
        logits = torch.einsum("shd,skd->shk", q_r, k) * scale
        logits = logits.masked_fill(~ok[:, None, :], float("-inf"))
        logits = torch.cat([logits, sink.view(1, HEADS, 1).expand(len(rows), HEADS, 1)], -1)
        probs = torch.softmax(logits, -1)[..., :-1]
        out = torch.einsum("shk,skd->shd", probs, k)
        out = _rotate(out, x.cos_sin, positions[rows], -1)
        outputs.append(out.detach())
        (out * grad[rows]).sum().backward()
    dq = torch.empty(length, HEADS * DIM, dtype=torch.float64, device=device)
    dq[:] = q_nat.grad.reshape(length, -1)[:, q_perm]
    return SimpleNamespace(
        dq=dq.view(length, HEADS, DIM),
        dswa=swa.grad,
        dlatent=None if latent is None else latent.grad,
        dsink=sink.grad,
        out=torch.cat(outputs),
    )


def _metrics(value, ref):
    value, ref = value.double().flatten(), ref.double().flatten()
    norm = ref.norm()
    return dict(
        rel_l2=((value - ref).norm() / norm).item(),
        cos=(value @ ref / (value.norm() * norm)).item(),
        max_abs=(value - ref).abs().max().item(),
        ref_max=ref.abs().max().item(),
    )


def _grads(x, grad):
    leaves = [x.q, x.swa_kv, x.sink] + ([x.latent] if x.latent is not None else [])
    leaves = [t.detach().clone().requires_grad_() for t in leaves]
    y = SimpleNamespace(**vars(x))
    y.q, y.swa_kv, y.sink = leaves[:3]
    y.latent = leaves[3] if x.latent is not None else None
    out = _ours(y)
    torch.autograd.backward(out.grad, grad)
    return out, SimpleNamespace(
        dq=y.q.grad,
        dswa=y.swa_kv.grad,
        dsink=y.sink.grad,
        dlatent=None if y.latent is None else y.latent.grad,
    )


def _upstream(length, generator, kind):
    grad = torch.randn(length, 8, 8 * DIM, device="cuda", generator=generator)
    if kind == "sparse":
        grad *= torch.rand(length, 1, 1, device="cuda", generator=generator) < 0.05
    return grad.bfloat16()


@pytest.mark.parametrize("kind", ["dense", "sparse"])
@pytest.mark.parametrize("ratio", [0, 2, 1])
@pytest.mark.parametrize("length", [129, 700, 8320])
def test_backward_matches_fp64_reference(ratio, length, kind, record_property):
    from megatron.lite.model.deepseek_v41.vllm.mega_attention import dequantize_output, o_from_fused

    x = _inputs(length, ratio, seed=3 * length + ratio)
    generator = torch.Generator(device="cuda").manual_seed(length)
    grad = _upstream(length, generator, kind)
    out, ours = _grads(x, grad)
    ref = _reference(x, grad)
    report = {
        "out": _metrics(o_from_fused(dequantize_output(out.data, out.scale)), ref.out),
        "dq": _metrics(ours.dq, ref.dq),
        "dswa": _metrics(ours.dswa, ref.dswa),
        "dsink": _metrics(ours.dsink, ref.dsink),
    }
    if ratio:
        report["dlatent"] = _metrics(ours.dlatent, ref.dlatent)
    record_property("metrics", report)
    print(f"MEGA_BWD ratio={ratio} length={length} kind={kind} {report}")
    # FP8 output: e4m3 rounding is <= 2**-4 relative per element.
    assert report["out"]["rel_l2"] < 5e-2, report
    for name in ("dq", "dswa", "dsink") + (("dlatent",) if ratio else ()):
        assert report[name]["rel_l2"] <= REL_L2 and report[name]["cos"] >= COSINE, (name, report)


def test_backward_rejects_planted_errors():
    """Discrimination: a dropped sink in the LSE and a 2% dQ scale both fail the bar."""
    from megatron.lite.model.deepseek_v41.vllm import mega_attention as mega

    x = _inputs(700, 2, seed=5)
    grad = _upstream(700, torch.Generator(device="cuda").manual_seed(1), "dense")
    _, ours = _grads(x, grad)
    ref = _reference(x, grad)
    scaled = _metrics(ours.dq * 1.02, ref.dq)
    assert scaled["rel_l2"] > REL_L2
    calls = []
    original = mega.attention_backward_kernel

    def no_sink(*args, **kwargs):
        calls.append(1)
        *head, lse_full, scale = args
        lse = lse_full + torch.log1p(-torch.exp(x.sink[None] - lse_full))
        return original(*head, lse, scale, **kwargs)

    mega.attention_backward_kernel = no_sink
    try:
        _, broken = _grads(x, grad)
    finally:
        mega.attention_backward_kernel = original
    assert calls
    bad = _metrics(broken.dq, ref.dq)
    assert bad["rel_l2"] > REL_L2 or bad["cos"] < COSINE, bad


def test_backward_is_finite_difference_consistent():
    """Tiny case: FD confirms the surrogate VJP at the visible point; the kernel matches it."""
    from vllm.models.deepseek_v41.common.ops.fused_layout import (
        o_fused_permutation,
        q_fused_permutation,
    )

    length, ratio = 40, 2
    x = _inputs(length, ratio, seed=11)
    generator = torch.Generator(device="cuda").manual_seed(2)
    grad = _upstream(length, generator, "dense")
    _, ours = _grads(x, grad)
    swa_vis, comp_vis = _visible_kv(x)
    q_perm = q_fused_permutation(HEADS, DIM).cuda()
    o_perm = o_fused_permutation(8, DIM).cuda()
    q_nat = torch.empty(length, HEADS * DIM, dtype=torch.float64, device="cuda")
    q_nat[:, q_perm] = x.q.reshape(length, -1).double()
    q_vis = _rotate(q_nat.view(length, HEADS, DIM), x.cos_sin, x.positions, 1).bfloat16().double()
    g = torch.empty(length, 8, 8 * DIM, dtype=torch.float64, device="cuda")
    g[:, :, o_perm] = grad.double()
    g = g.view(length, HEADS, DIM)
    rows = torch.arange(length, device="cuda")

    def loss(dq_fused, dswa, dlatent, dsink):
        dq_nat = torch.empty(length, HEADS * DIM, dtype=torch.float64, device="cuda")
        dq_nat[:, q_perm] = dq_fused.reshape(length, -1)
        q = q_vis + _rotate(dq_nat.view(length, HEADS, DIM), x.cos_sin, x.positions, 1)
        swa = swa_vis + _rotate(dswa, x.cos_sin, rows, 1)
        tokens = torch.arange(comp_vis.shape[0], device="cuda")
        comp = comp_vis + _rotate(dlatent[tokens * ratio + ratio - 1], x.cos_sin, tokens * ratio, 1)
        total = 0.0
        for i in range(length):
            ids = torch.arange(max(0, i - WINDOW + 1), i + 1, device="cuda")
            count = min((i + 1) // ratio, TOP_K)
            top = x.topk[i, :count].long()
            top = top[top >= 0]
            k = torch.cat([swa[ids], comp[top]])
            logits = torch.cat([q[i] @ k.T * DIM**-0.5, (x.sink.double() + dsink)[:, None]], 1)
            o = torch.softmax(logits, -1)[:, :-1] @ k
            o = _rotate(o[None], x.cos_sin, x.positions[i : i + 1], -1)[0]
            total = total + (o * g[i]).sum()
        return total

    shapes = [(length, HEADS, DIM), (length, DIM), (length, DIM), (HEADS,)]
    leaves = [
        torch.zeros(s, dtype=torch.float64, device="cuda", requires_grad=True) for s in shapes
    ]
    loss(*leaves).backward()
    names = ("dq", "dswa", "dlatent", "dsink")
    eps = 1e-4
    for which, (leaf, name) in enumerate(zip(leaves, names)):
        # 1. FD of the surrogate confirms its autograd VJP (no cancellation issue:
        #    both sides are exact up to O(eps**2) and FP64 rounding).
        direction = torch.randn(
            shapes[which], dtype=torch.float64, device="cuda", generator=generator
        )
        plus = [torch.zeros(s, dtype=torch.float64, device="cuda") for s in shapes]
        minus = [t.clone() for t in plus]
        plus[which] += eps * direction
        minus[which] -= eps * direction
        with torch.no_grad():
            numeric = ((loss(*plus) - loss(*minus)) / (2 * eps)).item()
        exact = (leaf.grad * direction).sum().item()
        scale = (leaf.grad.norm() * direction.norm()).item()
        # 2. The kernel against that VJP under the module's bar.
        metrics = _metrics(getattr(ours, name), leaf.grad)
        print(f"MEGA_FD {name} numeric={numeric:.9e} autograd={exact:.9e} kernel={metrics}")
        assert abs(numeric - exact) <= 1e-8 * scale, (name, numeric, exact)
        assert metrics["rel_l2"] <= REL_L2 and metrics["cos"] >= COSINE, (name, metrics)


def test_backward_rows_are_reproducible():
    """dq and dsink are per-row deterministic; dKV only differs by FP32 atomic order."""
    x = _inputs(2048, 2, seed=9)
    # FP32 KV leaves keep the gradients in FP32, so the bar measures the FP32
    # atomic order rather than bf16 rounding flips of the returned gradient.
    x.swa_kv, x.latent = x.swa_kv.float(), x.latent.float()
    grad = _upstream(2048, torch.Generator(device="cuda").manual_seed(4), "dense")
    _, a = _grads(x, grad)
    _, b = _grads(x, grad)
    assert torch.equal(a.dq, b.dq) and torch.equal(a.dsink, b.dsink)
    for name in ("dswa", "dlatent"):
        assert _metrics(getattr(a, name), getattr(b, name))["rel_l2"] < 1e-5


@pytest.mark.parametrize("ratio", [0, 2, 1])
def test_deterministic_dkv_is_bitwise_and_matches_atomics(ratio, monkeypatch):
    """Fixed-order dKV: two runs bitwise equal; within 2e-3 of the atomic mode (bf16 partials)."""
    from megatron.lite.model.deepseek_v41.vllm import mega_attention as mega

    x = _inputs(2048, ratio, seed=13)
    x.swa_kv = x.swa_kv.float()
    if ratio:
        x.latent = x.latent.float()
    grad = _upstream(2048, torch.Generator(device="cuda").manual_seed(6), "dense")
    _, atomic = _grads(x, grad)
    monkeypatch.setattr(mega, "DETERMINISTIC_DKV", True)
    monkeypatch.setattr(mega, "DETERMINISTIC_PARTIAL_BYTES", 300 * 640 * 512 * 2)  # 7 chunks
    _, a = _grads(x, grad)
    _, b = _grads(x, grad)
    names = ("dq", "dswa", "dsink") + (("dlatent",) if ratio else ())
    for name in names:
        assert torch.equal(getattr(a, name), getattr(b, name)), name
        bar = 2e-3 if name in ("dswa", "dlatent") else 1e-5
        assert _metrics(getattr(a, name), getattr(atomic, name))["rel_l2"] < bar, name


@pytest.mark.parametrize("ratio", [0, 2, 1])
def test_umma_kernel_matches_mma_kernel(ratio, monkeypatch):
    """The tcgen05 backward agrees with the mma.sync one up to summation order."""
    from megatron.lite.model.deepseek_v41.vllm import mega_attention as mega

    docs = [0, 700, 701, 2048]
    x = _inputs(docs[-1], ratio, seed=19, docs=docs)
    x.swa_kv = x.swa_kv.float()
    if ratio:
        x.latent = x.latent.float()
    grad = _upstream(docs[-1], torch.Generator(device="cuda").manual_seed(19), "dense")
    monkeypatch.setattr(mega, "BACKWARD_KERNEL", "umma")
    _, umma = _grads(x, grad)
    monkeypatch.setattr(mega, "BACKWARD_KERNEL", "mma")
    _, mma = _grads(x, grad)
    bars = {"dq": 1e-3, "dswa": 1e-4, "dlatent": 1e-4, "dsink": 1e-5}
    for name in ("dq", "dswa", "dsink") + (("dlatent",) if ratio else ()):
        m = _metrics(getattr(umma, name), getattr(mma, name))
        assert m["rel_l2"] < bars[name], (name, m)


def test_deterministic_dkv_matches_fp64_reference(monkeypatch):
    from megatron.lite.model.deepseek_v41.vllm import mega_attention as mega

    monkeypatch.setattr(mega, "DETERMINISTIC_DKV", True)
    x = _inputs(8320, 2, seed=3 * 8320 + 2)
    grad = _upstream(8320, torch.Generator(device="cuda").manual_seed(8320), "dense")
    _, ours = _grads(x, grad)
    ref = _reference(x, grad)
    for name in ("dq", "dswa", "dsink", "dlatent"):
        m = _metrics(getattr(ours, name), getattr(ref, name))
        print(f"MEGA_DET {name} {m}")
        assert m["rel_l2"] <= REL_L2 and m["cos"] >= COSINE, (name, m)


@pytest.mark.parametrize("ratio", [0, 2])
def test_layers_after_the_first_do_not_sync(ratio):
    """The plan is cached per cu_seqlens tensor: a second layer's forward and
    backward run without a host sync (CUDA sync debug mode raises on one)."""
    from megatron.lite.model.deepseek_v41.vllm.mega_attention import mega_attention

    docs = [0, 300, 301, 1500, 1700]
    x = _inputs(docs[-1], ratio, seed=17, docs=docs)
    cu = torch.tensor(docs, dtype=torch.int32, device="cuda")
    latent = None if x.latent is None else x.latent[(x.positions + 1) % ratio == 0]
    grad = _upstream(docs[-1], torch.Generator(device="cuda").manual_seed(3), "dense")

    def layer():
        q = x.q.detach().requires_grad_()
        out = mega_attention(
            q,
            x.swa_kv,
            latent,
            x.topk,
            x.positions,
            x.sink,
            compress_ratio=ratio,
            cos_sin_cache=x.cos_sin,
            cu_seqlens=cu,
        )
        torch.autograd.backward(out.grad, grad)
        return out.data, q.grad

    first = layer()  # builds the plan (one sync) and loads the extension
    torch.cuda.synchronize()
    torch.cuda.set_sync_debug_mode("error")
    try:
        second = layer()
    finally:
        torch.cuda.set_sync_debug_mode("default")
    assert torch.equal(first[0].view(torch.uint8), second[0].view(torch.uint8))
    assert torch.equal(first[1], second[1])
