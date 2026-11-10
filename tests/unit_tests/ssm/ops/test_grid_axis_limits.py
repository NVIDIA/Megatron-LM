# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""SSM kernel launches at CUDA's grid-axis limit.

CUDA caps grid axes 1 and 2 at 65535 programs (axis 0 at 2**31 - 1). Each case launches a
kernel with exactly 65535 programs, and with one more, along the dimension that grows with the
batch: requests x heads for the decode recurrence, sequences x heads for the chunk-state
recurrence, chunks for the chunk output, token blocks for the varlen causal conv. The last
request's results must be bitwise those of the same request launched alone: the arithmetic is
identical, and only the program that computes it differs.
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
