# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Oracle for absorbed MLA decode: FlashMLA ``get_mla_metadata`` + ``flash_mla_with_kvcache``.

Call site: the MLA branch of ``Attention.flash_decode_and_prefill``
(megatron/core/transformer/attention.py), taken by a decode-only step of a ``cache_mla_latents``
MLA layer (``MultiLatentAttention.forward`` passes the absorbed queries and the latent pool).

Contract (from the MLA math, the call site and FlashMLA's documented interface):

* Inputs: ``q [R * S, 1, H, C + Pd]`` absorbed queries ``[q_nope W_UK (C = kv_lora_rank = 512),
  q_pe (Pd = 64)]``; ``k``: the latent pool ``[pages, 64, C + Pd]`` (one shared KV head), whose
  rows are ``[c (512), k_pe (64)]``; ``v`` unused; ``block_table [R, W]``; ``seqlens_k [R]``;
  ``cu_seqlens_q`` with ``S`` rows per live request and 0 for padding requests (which come last,
  with length 0 and an all-dummy block-table row).
* Computes, for live request ``r`` with ``L_r`` cached latents: query ``i`` of ``S`` sits at
  ``L_r - S + i`` and attends latents ``j <= L_r - S + i`` (bottom-right causal), with
  ``scale = layer.softmax_scale`` (``(qk_head_dim + qk_pos_emb_head_dim) ** -0.5`` for plain
  RoPE); values are the latents' first 512 channels; a sink adds ``exp(sink_h)`` to head ``h``'s
  denominator. Output ``[R * S, 1, H, 512]``; padding rows unspecified.
* FlashMLA's own output ``lse [R, H, S]`` is the natural-log sum of exponentiated scaled scores
  (the call site's sink correction relies on it).
* Memory: the pool is read only. Rows past ``L_r`` in a request's last page and the dummy page
  are never written on main (``_initialize_mha_value_pages`` skips the latent cache) and may hold
  any bits, so on SM90 they hold NaN here; FlashMLA masks scores past ``L_r`` and zero-fills
  out-of-bounds latent rows. On Blackwell the cases run with a large finite value (1e4) there
  instead: #8873 keeps the dummy page zero and #8872 zeroes new latent pages, the precondition its
  FlashInfer kernel relies on.

Reference: fp64 attention over the latents gathered through the block table
(``attention_reference.attention_fp64``), LSE included. Tolerance class: floating point, bounded by
``attention_lowp`` in bf16 per token row; LSE (fp32) bounded by torch's fp32 ``logsumexp`` of the
same scores on the same device.

Variants covered: lengths ``1, 63, 64, 65, 127, 129, 4097`` around FlashMLA's 64-token block (the
page size dynamic MLA forces) and split-KV chunks; ``S`` in 1, 2, 4 (MTP); heads 16, 32, 64, 128
(``S * H`` crosses FlashMLA's 64-row query tile); batch 1, 2, #SMs, #SMs + 1 (the scheduler's
split boundaries); zero-length padding requests; learnable sink; CUDA-graph replay with changed
lengths (the scheduler metadata is recomputed inside the captured call).

Hardware and open PRs (written against main):
* H100 needs a FlashMLA build with its SM90 kernels: the CI image builds it with
  ``FLASH_MLA_DISABLE_SM90=1`` (fixed by #8701, ci), and a build without them exits the process at
  the first launch, so a subprocess probes it first and the tests fail with that reason.
* GB200: FlashMLA has no dense decode kernel on SM100 ("Dense decode MLA is only supported on
  SM90a"), so every ``launch_on_gb200`` case fails on main (R11-F1) and passes with #8873
  (t3-r11f1), which runs FlashInfer's TRTLLM-GEN MLA decode there.
* #8571 (s8) does not change this call site's decode math; its sink plumbing is covered by
  ``test_mla_dynamic_path.py``.
"""

import functools
import subprocess
import sys

import pytest
import torch

from megatron.core.transformer import attention as attention_module
from tests.unit_tests.inference.oracles import oracle_utils as ou
from tests.unit_tests.inference.oracles.attention import attention_reference as ar
from tests.unit_tests.test_utilities import Utils

LATENT, ROPE, BLOCK = 512, 64, 64  # FlashMLA's latent shape; dynamic MLA forces 64-token pages
LENGTHS = [1, BLOCK - 1, BLOCK, BLOCK + 1, 2 * BLOCK - 1, 2 * BLOCK + 1, 4097]
HEADS = [16, 32, 64, 128]

_FLASH_MLA_DECODE_PROBE = """
import sys

import torch
from flash_mla import flash_mla_with_kvcache, get_mla_metadata

torch.cuda.set_device(int(sys.argv[1]))
q = torch.zeros(1, 1, 16, 576, device="cuda", dtype=torch.bfloat16)
kv_cache = torch.zeros(1, 64, 1, 576, device="cuda", dtype=torch.bfloat16)
cache_seqlens = torch.ones(1, device="cuda", dtype=torch.int32)
block_table = torch.zeros(1, 1, device="cuda", dtype=torch.int32)
metadata, num_splits = get_mla_metadata(cache_seqlens, 16, 1)
flash_mla_with_kvcache(q, kv_cache, block_table, cache_seqlens, 512, metadata, num_splits)
torch.cuda.synchronize()
"""


@functools.cache
def _flash_mla_decode_runs() -> bool:
    """FlashMLA's SM90 decode kernel runs here (probed in a subprocess: a build without SM90
    kernels exits the process at the first launch)."""
    probe = subprocess.run(
        [sys.executable, "-c", _FLASH_MLA_DECODE_PROBE, str(torch.cuda.current_device())],
        capture_output=True,
        timeout=600,
    )
    return probe.returncode == 0


def _require_flash_mla_on_hopper() -> None:
    if ou.device_sm_major() != ou.HOPPER:
        return
    if not attention_module.HAVE_FMLA:
        pytest.fail("flash_mla is not installed", pytrace=False)
    if not _flash_mla_decode_runs():
        pytest.fail(
            "this flash_mla build has no SM90 decode kernels (the CI image builds it with "
            "FLASH_MLA_DISABLE_SM90=1; #8701)",
            pytrace=False,
        )


DEVICES = [ou.requires_sm(ou.HOPPER, ou.BLACKWELL), pytest.mark.launch_on_gb200]


@pytest.fixture
def model_parallel():
    Utils.initialize_model_parallel(1, 1)
    yield
    Utils.destroy_model_parallel()


def _mla_attention(num_heads: int):
    """An MLA layer with FlashMLA's latent shape (512 + 64 RoPE) and the latent cache on."""
    from megatron.core.models.gpt.gpt_layer_specs import (
        get_gpt_layer_with_transformer_engine_submodules,
    )
    from megatron.core.transformer.enums import AttnMaskType
    from megatron.core.transformer.multi_latent_attention import MLASelfAttention
    from megatron.core.transformer.transformer_config import MLATransformerConfig

    config = MLATransformerConfig(
        num_layers=1,
        hidden_size=512,
        num_attention_heads=num_heads,
        q_lora_rank=128,
        kv_lora_rank=LATENT,
        qk_head_dim=64,
        qk_pos_emb_head_dim=ROPE,
        v_head_dim=128,
        qk_layernorm=True,
        normalization="RMSNorm",
        add_bias_linear=False,
        rope_type="rope",
        apply_rope_fusion=False,
        cache_mla_latents=True,
        bf16=True,
        params_dtype=torch.bfloat16,
        hidden_dropout=0.0,
        attention_dropout=0.0,
    )
    submodules = get_gpt_layer_with_transformer_engine_submodules(
        multi_latent_attention=True, qk_layernorm=True
    ).self_attention.submodules
    module = MLASelfAttention(
        config, submodules, layer_number=1, attn_mask_type=AttnMaskType.causal
    )
    return module.cuda().eval()


def _poison() -> float:
    """NaN on SM90 (main's contract: latent tails are never initialized); a large finite value
    elsewhere (#8872/#8873's precondition for FlashInfer: finite tails and dummy page)."""
    return float("nan") if ou.device_sm_major() == ou.HOPPER else 1e4


def _latent_batch(lengths, tokens, num_heads, padding, *, spare_pages=2):
    """Decode inputs over a poisoned latent pool; returns (inputs, batch, live_rows)."""
    g = ou.make_generator(device="cuda")
    poison = _poison()
    q_lens = [tokens] * len(lengths) + [0] * padding
    kv_lens = list(lengths) + [0] * padding
    batch = ar.make_paged_batch(
        q_lens,
        kv_lens,
        page_size=BLOCK,
        kv_heads=1,
        head_dim=LATENT + ROPE,
        generator=g,
        tail_key=poison,
        tail_value=poison,
        unused=poison,
        spare_pages=spare_pages,
    )
    # One pool: the values are the latents' first 512 channels.
    batch.v_pool = batch.k_pool[..., :LATENT]
    batch.v_pool_before = batch.v_pool.clone()
    rows = len(kv_lens) * tokens
    q = ou.poisoned_empty((rows, 1, num_heads, LATENT + ROPE), torch.bfloat16)
    live = len(lengths) * tokens
    q[:live] = torch.randn(live, 1, num_heads, LATENT + ROPE, device="cuda", generator=g).to(
        torch.bfloat16
    )
    inputs = dict(
        q=q,
        k=batch.k.squeeze(2),
        v=None,
        max_seqlen_q=tokens,
        max_seqlen_k=max(kv_lens),
        cu_seqlens_q=batch.cu(q_lens),
        cu_seqlens_k=batch.cu(kv_lens),
        seqlens_k=batch.int_tensor(kv_lens),
        block_table=batch.block_table,
        is_decode_only=True,
    )
    return inputs, batch, live


def _check(out, inputs, batch, live, scale, sink, what):
    q = inputs["q"].squeeze(1)
    ref, _ = ar.paged_attention_reference(q, batch, scale=scale, sink=sink)
    trusted, _ = ar.paged_attention_reference(q, batch, scale=scale, sink=sink, lowp=True)
    assert out.shape == (q.shape[0], 1, q.shape[1], LATENT), f"output shape {tuple(out.shape)}"
    flat = lambda t: t.reshape(t.shape[0], -1)  # noqa: E731
    ou.assert_close_to_reference(flat(out[:live, 0]), flat(ref), flat(trusted), dim=-1, what=what)


def _decode(module, inputs, sink=None):
    q_before = inputs["q"].clone()
    with torch.no_grad():
        out = module.flash_decode_and_prefill(**inputs, softmax_offset=sink)
    torch.cuda.synchronize()
    ou.assert_exact(inputs["q"], q_before, what="q (read-only)")
    return out


def _cases():
    factors = {"heads": HEADS, "tokens": [1, 2, 4], "padding": [0, 2], "sink": [None, "learnable"]}
    return [
        pytest.param(case, id="-".join(f"{k}={v}" for k, v in case.items()), marks=DEVICES)
        for case in ou.pairwise_cases(factors)
    ]


@pytest.mark.usefixtures("model_parallel")
@pytest.mark.parametrize("case", _cases())
def test_decode_matches_reference(case):
    """Every live request (lengths around the 64-token block, one past split-KV chunks) matches
    the fp64 latent attention; poisoned tails and dummy page do not leak in."""
    _require_flash_mla_on_hopper()
    heads, tokens = case["heads"], case["tokens"]
    module = _mla_attention(heads)
    sink = None
    if case["sink"]:
        g = ou.make_generator(seed=ou.DEFAULT_SEED + 1, device="cuda")
        sink = (torch.randn(heads, device="cuda", generator=g) * 2 + 1).to(torch.bfloat16)
    lengths = [n for n in LENGTHS if n >= tokens]
    inputs, batch, live = _latent_batch(lengths, tokens, heads, case["padding"])
    out = _decode(module, inputs, sink)
    batch.check_pool_unchanged()
    _check(out, inputs, batch, live, module.softmax_scale, sink, f"MLA decode {case}")


@pytest.mark.usefixtures("model_parallel")
@pytest.mark.parametrize(
    "requests", [pytest.param(r, marks=DEVICES) for r in ("1", "2", "sms", "sms+1")]
)
def test_batch_sizes_around_the_sm_count(requests):
    """Batches of 1, 2, #SMs and #SMs + 1 requests (the split scheduler's partition points)."""
    _require_flash_mla_on_hopper()
    sms = torch.cuda.get_device_properties(0).multi_processor_count
    count = {"1": 1, "2": 2, "sms": sms, "sms+1": sms + 1}[requests]
    module = _mla_attention(16)
    cycle = [1, BLOCK - 1, BLOCK, BLOCK + 1, 2 * BLOCK - 1, 2 * BLOCK + 1]
    lengths = [cycle[r % len(cycle)] for r in range(count)]
    inputs, batch, live = _latent_batch(lengths, 1, 16, padding=1)
    out = _decode(module, inputs)
    batch.check_pool_unchanged()
    _check(out, inputs, batch, live, module.softmax_scale, None, f"MLA decode, {count} requests")


@pytest.mark.usefixtures("model_parallel")
@pytest.mark.sanitizer
@pytest.mark.parametrize("heads", [pytest.param(16, marks=DEVICES)])
def test_small_batch_for_sanitizer(heads):
    """A small poisoned decode (compute-sanitizer memcheck case)."""
    _require_flash_mla_on_hopper()
    module = _mla_attention(heads)
    inputs, batch, live = _latent_batch([3, BLOCK + 1], 1, heads, padding=1)
    out = _decode(module, inputs)
    batch.check_pool_unchanged()
    _check(out, inputs, batch, live, module.softmax_scale, None, "MLA decode sanitizer case")


@pytest.mark.usefixtures("model_parallel")
@pytest.mark.large
@pytest.mark.parametrize("heads", [pytest.param(16, marks=DEVICES)])
def test_pool_offsets_past_int32(heads):
    """The live latent pages sit past 2**31 elements into the pool: offsets in 64 bits."""
    _require_flash_mla_on_hopper()
    page_elems = BLOCK * (LATENT + ROPE)
    offset = ou.INT32_LIMIT // page_elems + 1
    ou.require_free_memory(2 * (offset + 16) * page_elems * 2)  # pool and its copy, bf16
    module = _mla_attention(heads)
    g = ou.make_generator(device="cuda")
    poison = _poison()
    lengths = [1, BLOCK + 1]
    batch = ar.make_paged_batch(
        lengths,
        lengths,
        page_size=BLOCK,
        kv_heads=1,
        head_dim=LATENT + ROPE,
        generator=g,
        tail_key=poison,
        tail_value=poison,
        unused=poison,
        live_page_offset=offset,
    )
    batch.v_pool = batch.k_pool[..., :LATENT]
    batch.v_pool_before = batch.v_pool.clone()
    q = torch.randn(len(lengths), 1, heads, LATENT + ROPE, device="cuda", generator=g)
    inputs = dict(
        q=q.to(torch.bfloat16),
        k=batch.k.squeeze(2),
        v=None,
        max_seqlen_q=1,
        max_seqlen_k=max(lengths),
        cu_seqlens_q=batch.cu([1] * len(lengths)),
        cu_seqlens_k=batch.cu(lengths),
        seqlens_k=batch.int_tensor(lengths),
        block_table=batch.block_table,
        is_decode_only=True,
    )
    batch.q_lens = [1] * len(lengths)
    out = _decode(module, inputs)
    batch.check_pool_unchanged()
    _check(out, inputs, batch, len(lengths), module.softmax_scale, None, "MLA decode past 2**31")


@pytest.mark.usefixtures("model_parallel")
@pytest.mark.parametrize("heads", [pytest.param(h, marks=DEVICES) for h in (16, 128)])
def test_decode_replays_in_a_cuda_graph(heads):
    """A captured decode call serves the next step: every live request gains one latent (one
    crosses into a new page), and the replay matches the reference for the new lengths."""
    _require_flash_mla_on_hopper()
    module = _mla_attention(heads)
    lengths = [BLOCK, BLOCK + 1, 2 * BLOCK - 1, 300]
    inputs, batch, live = _latent_batch(lengths, 1, heads, padding=1, spare_pages=2)
    with torch.no_grad():
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            module.flash_decode_and_prefill(**inputs)
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            out = module.flash_decode_and_prefill(**inputs)

    g = ou.make_generator(seed=ou.DEFAULT_SEED + 2, device="cuda")
    spare = batch.dummy_page - 1  # a free (poisoned) page
    pool = batch.k.squeeze(2)
    for r, length in enumerate(lengths):
        if length % BLOCK == 0:
            batch.block_table[r, length // BLOCK] = spare
        page = int(batch.block_table[r, length // BLOCK])
        pool[page, length % BLOCK] = torch.randn(LATENT + ROPE, device="cuda", generator=g).to(
            pool.dtype
        )
    batch.kv_lens = [n + 1 for n in lengths] + [0]
    inputs["seqlens_k"].copy_(batch.int_tensor(batch.kv_lens))
    inputs["cu_seqlens_k"].copy_(batch.cu(batch.kv_lens))
    inputs["q"][:live] = torch.randn(live, 1, heads, LATENT + ROPE, device="cuda", generator=g).to(
        torch.bfloat16
    )
    graph.replay()
    torch.cuda.synchronize()
    _check(out, inputs, batch, live, module.softmax_scale, None, "MLA decode after graph replay")


@pytest.mark.usefixtures("model_parallel")
@ou.requires_sm(ou.HOPPER)
@pytest.mark.parametrize("tokens", [1, 2])
def test_flash_mla_lse_is_natural_log(tokens):
    """FlashMLA's log-sum-exp, which the call site's sink correction reads, is the natural-log
    sum of the scaled scores ``[R, H, S]`` (read through the interface as the call site calls it).
    """
    _require_flash_mla_on_hopper()
    heads = 16
    module = _mla_attention(heads)
    lengths = [n for n in LENGTHS if n >= tokens]
    inputs, batch, live = _latent_batch(lengths, tokens, heads, padding=0)
    q = inputs["q"].view(len(lengths), tokens, heads, LATENT + ROPE)
    metadata, num_splits = attention_module.get_mla_metadata(inputs["seqlens_k"], tokens * heads, 1)
    with torch.no_grad():
        _, lse = attention_module.flash_mla_with_kvcache(
            q,
            inputs["k"].unsqueeze(-2),
            inputs["block_table"],
            inputs["seqlens_k"],
            LATENT,
            metadata,
            num_splits,
            softmax_scale=module.softmax_scale,
            causal=True,
        )
    _, ref_lse = ar.paged_attention_reference(
        inputs["q"].squeeze(1), batch, scale=module.softmax_scale
    )
    # Trusted: torch's fp32 logsumexp of the fp32 scores of the same bf16 inputs.
    trusted, start = [], 0
    for r, length in enumerate(lengths):
        k = ar.gather_request_kv(batch.k, batch.block_table, r, length)[:, 0].float()
        scores = torch.einsum("shd,ld->shl", q[r].float(), k) * module.softmax_scale
        allowed = ar.allowed_keys(tokens, length, None, "cuda")
        trusted.append(torch.logsumexp(scores.masked_fill(~allowed[:, None], float("-inf")), -1))
    trusted = torch.cat(trusted)  # [R * S, H]
    got = lse.permute(0, 2, 1).reshape(-1, heads)  # [R, H, S] -> [R * S, H]
    ou.assert_close_to_reference(got, ref_lse, trusted, what="FlashMLA lse")
