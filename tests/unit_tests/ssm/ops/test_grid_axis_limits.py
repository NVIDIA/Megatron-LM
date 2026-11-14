# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""SSM kernel launches at CUDA's grid-axis limit.

CUDA caps grid axes 1 and 2 at 65535 programs (axis 0 at 2**31 - 1). Each case launches a
kernel with exactly 65535 programs, and with one more, along the dimension that grows with the
batch: requests x heads for the decode recurrence, sequences x heads for the chunk-state
recurrence, chunks for the chunk output, token blocks for the varlen causal conv. The last
request's results must be bitwise those of the same request launched alone: the arithmetic is
identical, and only the program that computes it differs.

The ``large`` cases (opt-in, ``-m large``; 9 to 26 GiB each) cover offsets that pass 2**31 at
sizes the axis-0 grids now launch: the decode's token rows, its dense state rows, and the conv's
token index, each with a control one size smaller.
"""

import pytest
import torch

from megatron.core.ssm.ops.common import causal_conv1d_varlen
from megatron.core.ssm.ops.gdp.chunk_h import chunk_gated_delta_product_fwd_h
from megatron.core.ssm.ops.gdp.chunk_o import chunk_gated_delta_product_fwd_o
from megatron.core.ssm.ops.gdp.common import CHUNK_SIZE
from megatron.core.ssm.ops.gdp.fused_recurrent import fused_recurrent_gated_delta_rule_update

AXIS_LIMIT = 65535
HEADS = 32
"""Heads per rank of GDN's default shape (value heads) and of GDP 800m, at TP 1."""
TAIL = CHUNK_SIZE + 1
"""Tokens of the checked prefill sequence: a full chunk and a one-token chunk."""
AT_LIMIT = [
    pytest.param(AXIS_LIMIT // HEADS, id="at-limit"),
    pytest.param(AXIS_LIMIT // HEADS + 1, id="past-limit"),
]


def _requires_cuda():
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")


def _randn(gen, *shape, dtype=torch.bfloat16, scale=0.1):
    return (torch.randn(*shape, generator=gen, device="cuda") * scale).to(dtype)


def _cu_seqlens(lengths):
    cu = torch.tensor([0] + lengths, dtype=torch.int64).cumsum(0)
    return cu.to(dtype=torch.int32, device="cuda")


def _embed(x, rows):
    """``x`` (``[1, n, ...]``) as the last ``n`` rows of zeros ``[1, rows, ...]``."""
    out = x.new_zeros(1, rows, *x.shape[2:])
    out[:, rows - x.shape[1] :] = x
    return out


@pytest.mark.parametrize("requests", AT_LIMIT)
def test_decode_requests_times_heads(requests):
    """GDP/GDN decode: one program per request, value head and value tile (2047 requests of 32
    heads launch 65504 (request, head) programs, 2048 launch 65536)."""
    _requires_cuda()
    gen = torch.Generator(device="cuda").manual_seed(1234)
    H, K, V = 16, 16, 16
    q, k = _randn(gen, requests, 1, H, K), _randn(gen, requests, 1, H, K)
    v = _randn(gen, requests, 1, HEADS, V)
    g = -torch.rand(requests, 1, HEADS, generator=gen, device="cuda")
    beta = torch.rand(requests, 1, HEADS, generator=gen, device="cuda").bfloat16()
    initial = _randn(gen, requests + 1, HEADS, K, V, dtype=torch.float32, scale=1.0)
    slots = torch.randperm(requests + 1, generator=gen, device="cuda")[:requests].int()

    def decode(rows):
        state = initial.clone()
        out, _ = fused_recurrent_gated_delta_rule_update(
            q[rows],
            k[rows],
            v[rows],
            g=g[rows],
            beta=beta[rows],
            state=state,
            state_indices=slots[rows],
            use_qk_l2norm_in_kernel=True,
        )
        return out, state

    out, state = decode(slice(None))
    for request in (0, requests - 1):
        alone_out, alone_state = decode(slice(request, request + 1))
        slot = int(slots[request])
        assert torch.equal(out[request], alone_out[0]), f"output of request {request} differs"
        assert torch.equal(state[slot], alone_state[slot]), f"state of request {request} differs"


@pytest.mark.parametrize("sequences", AT_LIMIT)
def test_prefill_sequences_times_heads(sequences):
    """GDP/GDN prefill chunk states: one program per sequence, head and value tile; the checked
    two-chunk sequence comes after one-token sequences."""
    _requires_cuda()
    gen = torch.Generator(device="cuda").manual_seed(1234)
    K, V = 32, 16
    k, w = _randn(gen, 1, TAIL, HEADS, K), _randn(gen, 1, TAIL, HEADS, K)
    u = _randn(gen, 1, TAIL, HEADS, V)
    g = -torch.rand(1, TAIL, HEADS, generator=gen, device="cuda").mul(0.05)
    g[:, :CHUNK_SIZE] = g[:, :CHUNK_SIZE].cumsum(1)
    h0 = _randn(gen, 1, HEADS, K, V, scale=1.0)

    def run(k, w, u, g, h0, lengths):
        state = torch.full((2, HEADS, K, V), float("nan"), device="cuda")
        slots = [-1] * (len(lengths) - 1) + [1]
        h, v_new, _ = chunk_gated_delta_product_fwd_h(
            k=k,
            w=w,
            u=u,
            g=g,
            initial_state=h0,
            cu_seqlens=_cu_seqlens(lengths),
            state=state,
            state_indices=torch.tensor(slots, dtype=torch.int32, device="cuda"),
        )
        return h[0, -2:], v_new[0, -TAIL:], state[1]

    # Alone first, so the autotuner's choice for this shape is the one the batch reuses.
    expected = run(k, w, u, g, h0, [TAIL])
    lengths = [1] * (sequences - 1) + [TAIL]
    tokens = sum(lengths)
    k, w, u, g = (_embed(x, tokens) for x in (k, w, u, g))
    h0 = torch.cat([h0.new_zeros(sequences - 1, HEADS, K, V), h0])
    actual = run(k, w, u, g, h0, lengths)
    for name, a, e in zip(("chunk states", "corrected values", "final state"), actual, expected):
        assert torch.equal(a, e), f"{name} of the last of {sequences} sequences differ"


@pytest.mark.parametrize("chunks", [AXIS_LIMIT, AXIS_LIMIT + 1])
def test_prefill_output_chunks(chunks):
    """GDP/GDN prefill output: one program per chunk and value tile; the checked two-chunk
    sequence comes after a sequence of zero chunks (4.2M tokens of one small head)."""
    _requires_cuda()
    gen = torch.Generator(device="cuda").manual_seed(1234)
    K = V = 16
    q, k, v = _randn(gen, 1, TAIL, 1, K), _randn(gen, 1, TAIL, 1, K), _randn(gen, 1, TAIL, 1, V)
    h = _randn(gen, 1, 2, 1, K, V)
    g = -torch.rand(1, TAIL, 1, generator=gen, device="cuda").mul(0.05)
    g[:, :CHUNK_SIZE] = g[:, :CHUNK_SIZE].cumsum(1)

    def run(q, k, v, h, g, lengths):
        o = chunk_gated_delta_product_fwd_o(
            q=q, k=k, v=v, h=h, g=g, scale=K**-0.5, cu_seqlens=_cu_seqlens(lengths)
        )
        return o[0, -TAIL:]

    expected = run(q, k, v, h, g, [TAIL])
    lead = chunks - 2
    lengths = [lead * CHUNK_SIZE, TAIL]
    q, k, v, g = (_embed(x, sum(lengths)) for x in (q, k, v, g))
    h = torch.cat([h.new_zeros(1, lead, 1, K, V), h], dim=1)  # the chunk states of the zeros
    actual = run(q, k, v, h, g, lengths)
    assert torch.equal(actual, expected), f"outputs of the last sequence of {chunks} chunks differ"


@pytest.fixture
def conv_smallest_token_block(monkeypatch):
    """Pin the varlen conv's autotuner to its config with the fewest tokens per program, the
    one with the most token blocks per step, and return that BLOCK_T."""
    tuner = causal_conv1d_varlen._causal_conv1d_varlen_kernel
    config = min(tuner.configs, key=lambda c: c.kwargs["BLOCK_T"])
    monkeypatch.setattr(tuner, "configs", [config])
    monkeypatch.setattr(tuner, "cache", {})
    return config.kwargs["BLOCK_T"]


@pytest.mark.parametrize("blocks", [AXIS_LIMIT, AXIS_LIMIT + 1])
def test_varlen_conv_token_blocks(conv_smallest_token_block, blocks):
    """Varlen causal conv (Mamba, GDN and GDP prefill): one program per channel block and token
    block; at BLOCK_T 8 a step of 524281 tokens has 65536 token blocks. The checked request comes
    after one long request."""
    _requires_cuda()
    gen = torch.Generator(device="cuda").manual_seed(1234)
    conv_dim, width = 64, 4
    tokens = blocks * conv_smallest_token_block
    lengths = [tokens - TAIL, TAIL]
    x = _randn(gen, tokens, conv_dim, scale=1.0)
    weight, bias = _randn(gen, conv_dim, width, scale=1.0), _randn(gen, conv_dim, scale=1.0)
    states = _randn(gen, 2, conv_dim, width - 1, scale=1.0)

    out = causal_conv1d_varlen.causal_conv1d_varlen_fn(
        x, weight, bias, _cu_seqlens(lengths), states
    )
    alone = causal_conv1d_varlen.causal_conv1d_varlen_fn(
        x[-TAIL:].contiguous(), weight, bias, _cu_seqlens([TAIL]), states[1:].contiguous()
    )
    assert torch.equal(out[-TAIL:], alone), f"conv of the last request of {tokens} tokens differs"


# Offsets that pass 2**31 at sizes the axis-0 grids now launch (`large`, opt-in with -m large;
# each case with a control one size smaller).


def _requires_free_memory(nbytes):
    free = torch.cuda.mem_get_info()[0]
    if free < nbytes:
        pytest.skip(f"needs {nbytes / 2**30:.1f} GiB of free GPU memory, {free / 2**30:.1f} free")


@pytest.mark.large
@pytest.mark.parametrize(
    "requests",
    [
        pytest.param(2**31 // (HEADS * 128), id="fits"),
        pytest.param(2**31 // (HEADS * 128) + 1, id="past-int32"),
    ],
)
def test_decode_token_offsets_past_int32(requests):
    """GDP/GDN decode with a slot-indexed cache: the last request's query and key rows start at
    (request * T * H + head) * K, which passes 2**31 from 524289 requests of one token at 32 heads
    of 128 (from 4855 requests of 27 steps at 64 heads of 256). Every other request is padding."""
    _requires_cuda()
    H, K, V = HEADS, 128, 8
    _requires_free_memory(requests * H * (4 * K + 4 * V + 8) + (1 << 30))
    gen = torch.Generator(device="cuda").manual_seed(1234)
    q = torch.zeros(requests, 1, H, K, device="cuda", dtype=torch.bfloat16)
    k, v = torch.zeros_like(q), q.new_zeros(requests, 1, H, V)
    g = q.new_zeros(requests, 1, H, dtype=torch.float32)
    beta = q.new_zeros(requests, 1, H)
    for x in (q, k, v):
        x[-1] = _randn(gen, *x.shape[1:])
    g[-1] = -torch.rand(*g.shape[1:], generator=gen, device="cuda")
    beta[-1] = torch.rand(*beta.shape[1:], generator=gen, device="cuda")
    initial = _randn(gen, 2, H, K, V, dtype=torch.float32, scale=1.0)
    slots = torch.full((requests,), -1, dtype=torch.int32, device="cuda")
    slots[-1] = 1

    def decode(rows):
        state = initial.clone()
        out, _ = fused_recurrent_gated_delta_rule_update(
            q[rows],
            k[rows],
            v[rows],
            g=g[rows],
            beta=beta[rows],
            state=state,
            state_indices=slots[rows],
            use_qk_l2norm_in_kernel=True,
        )
        return out[-1], state[1]

    expected = decode(slice(requests - 1, requests))
    actual = decode(slice(None))
    for name, a, e in zip(("output", "state"), actual, expected):
        assert torch.equal(a, e), f"{name} of the last of {requests} requests differs"


@pytest.mark.large
@pytest.mark.parametrize(
    "sequences",
    [
        pytest.param(2**31 // (HEADS * 128 * 128), id="fits"),
        pytest.param(2**31 // (HEADS * 128 * 128) + 1, id="past-int32"),
    ],
)
def test_dense_state_offsets_past_int32(sequences):
    """The recurrence with a dense state (``state`` without slot indices, one row per sequence):
    the last sequence's state starts at (sequence * HV + head) * K * V, which passes 2**31 from
    4097 sequences of 32 heads of 128 x 128 (8.6 GB of fp32 state)."""
    _requires_cuda()
    H, K, V = HEADS, 128, 128
    _requires_free_memory(2 * sequences * H * K * V * 4 + (1 << 30))
    gen = torch.Generator(device="cuda").manual_seed(1234)
    q, k = _randn(gen, sequences, 1, H, K), _randn(gen, sequences, 1, H, K)
    v = _randn(gen, sequences, 1, H, V)
    g = -torch.rand(sequences, 1, H, generator=gen, device="cuda")
    beta = torch.rand(sequences, 1, H, generator=gen, device="cuda").bfloat16()
    state = torch.zeros(sequences, H, K, V, device="cuda")
    state[-1] = _randn(gen, H, K, V, dtype=torch.float32, scale=1.0)
    alone = state[-1:].clone()
    last = slice(sequences - 1, sequences)
    alone_out, _ = fused_recurrent_gated_delta_rule_update(
        q[last],
        k[last],
        v[last],
        g=g[last],
        beta=beta[last],
        state=alone,
        use_qk_l2norm_in_kernel=True,
    )
    out, _ = fused_recurrent_gated_delta_rule_update(
        q, k, v, g=g, beta=beta, state=state, use_qk_l2norm_in_kernel=True
    )
    assert torch.equal(out[-1], alone_out[0]), f"output of the last of {sequences} differs"
    assert torch.equal(state[-1], alone[0]), f"state of the last of {sequences} differs"


@pytest.mark.large
@pytest.mark.parametrize(
    "tokens", [pytest.param(2**31 - 64, id="fits"), pytest.param(2**31 + 8, id="past-int32")]
)
def test_varlen_conv_tokens_past_int32(tokens):
    """Varlen causal conv over one channel: the last request's tokens run past index 2**31, which
    the axis-0 grid now launches (26 GiB with the per-token request metadata)."""
    _requires_cuda()
    _requires_free_memory(tokens * 12 + (1 << 30))
    gen = torch.Generator(device="cuda").manual_seed(1234)
    conv_dim, width = 1, 4
    weight, bias = _randn(gen, conv_dim, width, scale=1.0), _randn(gen, conv_dim, scale=1.0)
    states = _randn(gen, 2, conv_dim, width - 1, scale=1.0)
    tail = _randn(gen, TAIL, conv_dim, scale=1.0)
    # Alone first, so the autotuner's choice for this conv_dim is the one the step reuses.
    alone = causal_conv1d_varlen.causal_conv1d_varlen_fn(
        tail, weight, bias, _cu_seqlens([TAIL]), states[1:].contiguous()
    )
    x = torch.zeros(tokens, conv_dim, device="cuda", dtype=torch.bfloat16)
    x[-TAIL:] = tail
    seq_idx = torch.zeros(tokens, device="cuda", dtype=torch.int32)
    seq_idx[-TAIL:] = 1
    seq_start = torch.zeros_like(seq_idx)
    seq_start[-TAIL:] = tokens - TAIL
    out = causal_conv1d_varlen.causal_conv1d_varlen_fn(
        x,
        weight,
        bias,
        # Only its length is read with precomputed metadata; int64, since tokens pass 2**31.
        torch.tensor([0, tokens - TAIL, tokens], device="cuda"),
        states,
        precomputed_seq_idx=seq_idx,
        precomputed_seq_start=seq_start,
    )
    assert torch.equal(out[-TAIL:], alone), f"conv of the last request of {tokens} tokens differs"
