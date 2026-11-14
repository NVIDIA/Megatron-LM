# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Oracle for ``_append_kv_cache_kernel``
(megatron/core/inference/contexts/fused_kv_append_kernel.py, launcher
``triton_append_key_value_cache``), the paged KV-cache write of every MHA/GQA layer (and the MTP
draft's KV slot) under dynamic inference.

**Contract** (from ``DynamicInferenceContext.append_key_value_cache``, whose torch fallback
defines the same write, and the launcher's docstring). For each of the first
``n = padded_active_token_count`` tokens ``t`` and each KV head ``h``,

    memory_buffer[0, layer, block[t], pos[t], h, :] = key[t, 0, h, :]
    memory_buffer[1, layer, block[t], pos[t], h, :] = value[t, 0, h, :]

bit for bit, where ``block = token_to_block_idx`` and ``pos =
token_to_local_position_within_kv_block``, except that tokens routed to the dummy block
(``block[t] == dummy_block_idx``: CUDA-graph padding and recomputed prefix tokens) write nothing,
so the dummy block keeps its contents. Everything else in ``memory_buffer`` (other layers,
blocks, positions), the rows of ``key``/``value`` and the index entries past ``n``, and all
inputs are untouched. ``layer`` is the 0-based attention-layer slot (the last slot is the MTP
draft's when it has one). Inputs: ``key``/``value`` ``[T >= n, 1, H, D]`` CUDA, **any strides**
(production: views of the fused QKV projection split along its last dimension; the key is a
fresh tensor after RoPE); ``memory_buffer [2, L, P, B, H, D]`` in the same dtype; ``block`` int64
(``gpu_view.py:162``; the kernel also compiles for int32), ``pos`` int32. Preconditions
(unchecked): unique ``(block, pos)`` among the live tokens, indices in range, value shaped like
key. ``n = 0`` does nothing. Tolerance class: copy, **exact**.

**Oracle.** Torch advanced-index assignment on a clone of the cache
(``expected[0, layer][block[live], pos[live]] = key[live, 0]``).

**Inputs from the kernel's structure.** Grid ``(n, H)``, one program per token and head copying
``D`` elements in one ``BLOCK_SIZE_H = next_pow2(D)`` tile; ``n``, ``H``, ``D`` and the dummy
index are int32 runtime arguments, the 10 strides runtime ints (Triton specializes 1 and
multiples of 16, and pointer alignment). Factors, combined pairwise: head dims 63/64/65,
127/128/129 and 80, 96, 256 (masked tiles); 1, 2, 8, 16, 17 heads; 1, 15/16/17, 64, 129 tokens;
bf16, fp16, fp32; key and value contiguous, split from a fused QKV row (strided, with a base
offset that is unaligned for odd ``D``), key contiguous with the value split (after RoPE), or with
inner strides 2 and 3; int64 and int32 block indices, int32 and int64 positions; the first,
a middle and the last (MTP) layer slot; no dummy tokens, trailing padding, recomputed prefix
tokens plus padding; 16-, 64- and 256-token blocks; 0 or 3 rows (and index entries) past ``n``,
poisoned. Also ``n = 0``, **CUDA-graph replay** with new data and new block, position and dummy
routing (the call is captured in layer and block graphs), a production size (8192 tokens of a
Llama-3-8B-shaped GQA layer at TP 1), and int32 limits (``large``).

**Memory.** The cache is guard-banded (one block of guard) and starts as random bits with the
dummy block NaN-poisoned; the whole cache must equal the expected one. Inputs and indices checked
unchanged. One small case runs under compute-sanitizer.

**Hardware.** H100. ``large``: cache offsets past 2**31 through int64 block indices pass. A token
whose key/value row starts past 2**31 fails on main, finding O6-M2
(``token_idx * stride_key_token`` and ``token_idx * stride_value_token`` are int32,
``fused_kv_append_kernel.py:75-76``). #9307 (fix/context-kernels-int64-offsets) fixes it and
carries that case as its regression test
(``test_fused_kv_append_kernel.py::TestKVAppendFusedQKVViews::test_token_offsets_past_int32``);
the token-offset sizes here stop at the one whose last row crosses 2**31.
"""

import math

import pytest
import torch

from megatron.core.inference.contexts.fused_kv_append_kernel import triton_append_key_value_cache
from tests.unit_tests.inference.oracles import oracle_utils as ou
from tests.unit_tests.inference.oracles.context import copy_buffers as cb

DEVICE = "cuda"

LAYERS = 3
"""Attention-layer slots in the cache; the last one stands for the MTP draft's slot."""
LAYER_SLOTS = {"first": 0, "middle": 1, "mtp_slot": LAYERS - 1}

HEAD_DIMS = ou.boundary_sizes(64, 128, multiples=(), degenerate=False, production=(80, 96, 256))
QUERY_HEADS_PER_GROUP = 2
"""Query heads per KV head in the fused-QKV layouts of the factor cases."""

CASES = ou.pairwise_cases(
    {
        "head_dim": HEAD_DIMS,
        "tokens": (1, 15, 16, 17, 64, 129),
        "layout": ("contiguous", "qkv_split", "rope_key", "hdim_strided"),
        "heads": (1, 2, 8, 16, 17),
        "dtype": (torch.bfloat16, torch.float16, torch.float32),
        "dummy": ("none", "padding", "prefix_and_padding"),
        "layer": tuple(LAYER_SLOTS),
        "block_tokens": (16, 64, 256),
        "block_dtype": (torch.int64, torch.int32),
        "pos_dtype": (torch.int32, torch.int64),
        "extra": (0, 3),
    }
)


def _case_id(c):
    dtype = str(c["dtype"]).replace("torch.", "")
    blk = str(c["block_dtype"]).replace("torch.", "")
    pos = str(c["pos_dtype"]).replace("torch.", "")
    return (
        f"n{c['tokens']}+{c['extra']}-H{c['heads']}-D{c['head_dim']}-{dtype}-{c['layout']}"
        f"-{c['dummy']}-{c['layer']}-blk{c['block_tokens']}-{blk}-pos_{pos}"
    )


def _key_value(layout, rows, tokens, heads, head_dim, dtype, g, group=QUERY_HEADS_PER_GROUP):
    """``(key, value, owners)``: ``[rows, 1, heads, head_dim]`` views in ``layout`` and the
    tensors that hold them (rows past ``tokens`` NaN-poisoned: the kernel must not read them)."""
    owners = []

    def owner(width):
        t = cb.random_bits((rows, 1, heads, width), dtype, g)
        ou.poison_(t[tokens:])
        owners.append(t)
        return t

    if layout == "contiguous":
        key, value = owner(head_dim), owner(head_dim)
    elif layout in ("qkv_split", "rope_key"):
        mixed = owner((group + 2) * head_dim)
        key = mixed[..., group * head_dim : (group + 1) * head_dim]
        value = mixed[..., (group + 1) * head_dim :]
        if layout == "rope_key":
            key = owner(head_dim)
    elif layout == "hdim_strided":
        key = owner(2 * head_dim)[..., ::2]
        value = owner(3 * head_dim)[..., 1::3]
    else:
        raise ValueError(layout)
    return key, value, owners


def _slots(tokens, extra, pool, block_tokens, dummy_kind, g):
    """``(block, pos)`` for ``tokens + extra`` entries: unique slots outside the dummy block
    (the last one), then the dummy routing of the live tokens."""
    dummy = pool - 1
    slots = torch.randperm(dummy * block_tokens, generator=g)[: tokens + extra]
    block, pos = slots // block_tokens, slots % block_tokens
    if dummy_kind == "padding":
        block[tokens - max(1, tokens // 4) : tokens] = dummy
    elif dummy_kind == "prefix_and_padding":
        block[1 : min(3, tokens)] = dummy
        block[tokens - 1] = dummy
    return block, pos


def _reference_(expected, layer, key, value, block, pos, tokens, dummy):
    """In place on ``expected`` (a clone of the cache)."""
    block, pos = block[:tokens].cpu().long(), pos[:tokens].cpu().long()
    live = torch.nonzero(block != dummy).reshape(-1)
    b, p = block[live].to(expected.device), pos[live].to(expected.device)
    live = live.to(key.device)
    expected[0, layer][b, p] = key[live, 0]
    expected[1, layer][b, p] = value[live, 0]


def _append_and_check(
    key, value, owners, layer, tokens, block, pos, pool, block_tokens, dtype, g, what
):
    heads, head_dim = key.shape[-2:]
    guard = block_tokens * heads * head_dim * dtype.itemsize
    shape = (2, LAYERS, pool, block_tokens, heads, head_dim)
    mem = ou.guarded_like(cb.random_bits(shape, dtype, g), guard_bytes=guard)
    dummy = pool - 1
    ou.poison_(mem.tensor[:, :, dummy])
    expected = mem.tensor.clone()
    _reference_(expected, layer, key, value, block, pos, tokens, dummy)
    owners_before = [o.clone() for o in owners]
    block_dev, pos_dev = block.to(DEVICE), pos.to(DEVICE)

    triton_append_key_value_cache(
        layer_number=layer,
        key=key,
        value=value,
        memory_buffer=mem.tensor,
        padded_active_token_count=tokens,
        token_to_block_idx=block_dev,
        token_to_local_position_within_kv_block=pos_dev,
        dummy_block_idx=dummy,
    )
    torch.cuda.synchronize()

    ou.check_guards(cache=mem)
    ou.assert_exact(mem.tensor, expected, what=what)
    for i, (o, before) in enumerate(zip(owners, owners_before)):
        ou.assert_exact(o, before, what=f"key/value storage {i} is read-only")
    ou.assert_exact(block_dev, block, what="block indices are read-only")
    ou.assert_exact(pos_dev, pos, what="positions are read-only")


SANITIZER_CASE = dict(
    head_dim=80,
    tokens=17,
    layout="qkv_split",
    heads=2,
    dtype=torch.bfloat16,
    dummy="prefix_and_padding",
    layer="mtp_slot",
    block_tokens=16,
    block_dtype=torch.int64,
    pos_dtype=torch.int32,
    extra=3,
)


@ou.requires_sm(ou.HOPPER)
@pytest.mark.parametrize(
    "case",
    [pytest.param(c, id=_case_id(c)) for c in CASES]
    + [
        pytest.param(
            SANITIZER_CASE, id="sanitizer-" + _case_id(SANITIZER_CASE), marks=pytest.mark.sanitizer
        )
    ],
)
def test_append_matches_contract(case):
    g = ou.make_generator(device=DEVICE)
    g_cpu = ou.make_generator()
    tokens, extra, bt = case["tokens"], case["extra"], case["block_tokens"]
    key, value, owners = _key_value(
        case["layout"], tokens + extra, tokens, case["heads"], case["head_dim"], case["dtype"], g
    )
    pool = math.ceil((tokens + extra) / bt) + 2  # a spare block and the dummy block
    block, pos = _slots(tokens, extra, pool, bt, case["dummy"], g_cpu)
    _append_and_check(
        key,
        value,
        owners,
        LAYER_SLOTS[case["layer"]],
        tokens,
        block.to(case["block_dtype"]),
        pos.to(case["pos_dtype"]),
        pool,
        bt,
        case["dtype"],
        g,
        what=_case_id(case),
    )


@ou.requires_sm(ou.HOPPER)
def test_no_tokens_is_a_no_op():
    """``padded_active_token_count = 0``: the launcher returns; the cache is unchanged."""
    g = ou.make_generator(device=DEVICE)
    key, value, owners = _key_value("qkv_split", 4, 0, 8, 128, torch.bfloat16, g)
    block = torch.zeros(4, dtype=torch.int64)
    pos = torch.arange(4, dtype=torch.int32)
    _append_and_check(key, value, owners, 1, 0, block, pos, 3, 16, torch.bfloat16, g, what="n = 0")


@ou.requires_sm(ou.HOPPER)
def test_append_at_production_size():
    """8192 tokens of a Llama-3-8B-shaped layer at TP 1 (32 query heads, 8 KV heads, D 128,
    bf16): the key after RoPE, the value a view of the fused QKV row; 64-token blocks, the last
    10% of the tokens CUDA-graph padding routed to the dummy block; the MTP slot."""
    tokens, heads, head_dim, bt, dtype = 8192, 8, 128, 64, torch.bfloat16
    g = ou.make_generator(device=DEVICE)
    g_cpu = ou.make_generator()
    key, value, owners = _key_value("rope_key", tokens, tokens, heads, head_dim, dtype, g, group=4)
    pool = tokens * 3 // (2 * bt) + 1
    dummy = pool - 1
    slots = torch.randperm(dummy * bt, generator=g_cpu)[:tokens]
    block, pos = slots // bt, (slots % bt).to(torch.int32)
    block[-tokens // 10 :] = dummy
    _append_and_check(
        key, value, owners, LAYERS - 1, tokens, block, pos, pool, bt, dtype, g, what="production"
    )


@ou.requires_sm(ou.HOPPER)
def test_cuda_graph_replay_follows_the_indices():
    """Captured once with a fixed token count (the grid), replayed with new key/value data and
    new block, position and dummy routing in the same index buffers, as on every replay of a
    layer graph; the cache accumulates across replays."""
    tokens, rows, heads, head_dim, bt, dtype = 37, 48, 8, 128, 16, torch.bfloat16
    layer, pool = 1, 9
    dummy = pool - 1
    g = ou.make_generator(device=DEVICE)
    g_cpu = ou.make_generator()
    key, value, owners = _key_value("qkv_split", rows, rows, heads, head_dim, dtype, g, group=4)
    (mixed,) = owners
    shape = (2, LAYERS, pool, bt, heads, head_dim)
    mem = ou.guarded_like(cb.random_bits(shape, dtype, g), guard_bytes=bt * heads * head_dim * 2)
    ou.poison_(mem.tensor[:, :, dummy])
    slots = torch.randperm(dummy * bt, generator=g_cpu)[:rows]
    block = (slots // bt).to(DEVICE)
    pos = (slots % bt).to(torch.int32).to(DEVICE)

    def append():
        triton_append_key_value_cache(
            layer, key, value, mem.tensor, tokens, block, pos, dummy_block_idx=dummy
        )

    expected = mem.tensor.clone()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        append()  # compile outside capture
    torch.cuda.current_stream().wait_stream(stream)
    _reference_(expected, layer, key, value, block, pos, tokens, dummy)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        append()

    for padding, prefix in ((0, ()), (5, ()), (tokens, ()), (1, (2, 3))):
        mixed.copy_(cb.random_bits(mixed.shape, dtype, g))
        slots = torch.randperm(dummy * bt, generator=g_cpu)[:rows]
        step_block, step_pos = slots // bt, (slots % bt).to(torch.int32)
        step_block[tokens - padding : tokens] = dummy
        step_block[list(prefix)] = dummy
        block.copy_(step_block)
        pos.copy_(step_pos)
        _reference_(expected, layer, key, value, step_block, step_pos, tokens, dummy)
        graph.replay()
        torch.cuda.synchronize()
        ou.check_guards(cache=mem)
        ou.assert_exact(mem.tensor, expected, what=f"replay, {padding} padding, prefix {prefix}")


@ou.requires_sm(ou.HOPPER)
@pytest.mark.parametrize("blocks", ou.int32_overflow_params(64 * 8 * 128))
def test_append_cache_offsets_past_int32(blocks):
    """Cache offsets ``block * 65536`` (64-token blocks of 8 heads x 128) at and past 2**31:
    the written block ``blocks - 1`` ends at 2**31 - 1 or starts at 2**31; int64 block indices
    (production) keep the offset in 64 bits, so this should pass. 8.6 GB."""
    bt, heads, head_dim, dtype = 64, 8, 128, torch.bfloat16
    pool = blocks + 1
    dummy = pool - 1
    ou.require_free_memory(2 * pool * bt * heads * head_dim * dtype.itemsize + 2**30)
    g = ou.make_generator(device=DEVICE)
    tokens = bt + 2
    key = cb.random_bits((tokens, 1, heads, head_dim), dtype, g)
    value = cb.random_bits((tokens, 1, heads, head_dim), dtype, g)
    block = torch.tensor([blocks - 1] * bt + [0, 0], dtype=torch.int64, device=DEVICE)
    pos = torch.tensor(list(range(bt)) + [0, 1], dtype=torch.int32, device=DEVICE)
    mem = torch.zeros((2, 1, pool, bt, heads, head_dim), dtype=dtype, device=DEVICE)

    triton_append_key_value_cache(0, key, value, mem, tokens, block, pos, dummy)
    torch.cuda.synchronize()

    ou.assert_exact(mem[0, 0, blocks - 1], key[:bt, 0], what=f"key block {blocks - 1}")
    ou.assert_exact(mem[1, 0, blocks - 1], value[:bt, 0], what=f"value block {blocks - 1}")
    ou.assert_exact(mem[0, 0, 0, :2], key[bt:, 0], what="key block 0")
    ou.assert_exact(mem[1, 0, 0, :2], value[bt:, 0], what="value block 0")
    written = int(torch.count_nonzero(mem[:, :, 1 : blocks - 1])) + int(
        torch.count_nonzero(mem[:, :, 0, 2:])
    )
    assert written == 0, f"{written} elements outside the written slots changed"


TOKEN_STRIDE = 8 * (16 + 2) * 128
"""Elements per token of a fused QKV row with 8 KV heads, 16 query heads per KV head and D 128
(a 128-query-head GQA layer at TP 1): the key's and value's token stride."""


@ou.requires_sm(ou.HOPPER)
@pytest.mark.parametrize(
    "tokens",
    [p for p in ou.int32_overflow_params(TOKEN_STRIDE) if (p.values[0] - 1) * TOKEN_STRIDE < 2**31],
)
def test_append_key_offsets_past_int32(tokens):
    """Key and value token offsets ``t * 18432`` up to and past 2**31: key and value are views
    of a fused QKV tensor behind a 2**31-element poisoned front guard, so a wrapped int32 offset
    reads poison instead of faulting. A token whose row starts past 2**31 fails on main (finding
    O6-M2) and is #9307's regression case. Unreachable at production sizes: it needs
    ``max_tokens`` times the per-rank fused QKV width to reach 2**31 (about 350k tokens for an 8B
    GQA model at TP 1). About 14 GB."""
    heads, head_dim, group, bt, dtype = 8, 128, 16, 64, torch.bfloat16
    width = (group + 2) * head_dim
    ou.require_free_memory(3 * tokens * TOKEN_STRIDE * dtype.itemsize + 2**31)
    mixed, guard = cb.behind_front_guard((tokens, 1, heads, width), dtype, device=DEVICE)
    g = ou.make_generator(device=DEVICE)
    mixed.copy_(cb.random_bits(mixed.shape, dtype, g))
    key = mixed[..., group * head_dim : (group + 1) * head_dim]
    value = mixed[..., (group + 1) * head_dim :]
    pool = math.ceil(tokens / bt) + 1
    slot = torch.arange(tokens, device=DEVICE)
    block, pos = slot // bt, (slot % bt).to(torch.int32)
    mem = torch.zeros((2, 1, pool, bt, heads, head_dim), dtype=dtype, device=DEVICE)

    triton_append_key_value_cache(0, key, value, mem, tokens, block, pos, pool - 1)
    torch.cuda.synchronize()

    assert cb.changed(guard) == 0, "the front guard was written"
    flat = mem.view(2, pool * bt, heads, head_dim)
    ou.assert_exact(flat[0, :tokens], key[:, 0], what=f"keys of {tokens} tokens")
    ou.assert_exact(flat[1, :tokens], value[:, 0], what=f"values of {tokens} tokens")
