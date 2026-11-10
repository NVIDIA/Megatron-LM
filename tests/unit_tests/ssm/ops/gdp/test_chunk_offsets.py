# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Base offsets of the GDP/GDN prefill's chunk-state and output kernels.

A packed prefill step addresses each sequence at its first token and first chunk-state row, so
those base offsets grow with the whole step. Each case runs a two-chunk sequence after ``lead``
zero chunks of a sequence before it, after ``lead`` one-token sequences, or after ``lead`` zero
chunks of its own (from a zero initial state, which zero tokens leave unchanged). Its chunk
states, corrected values, final state and outputs must be bitwise those of the same sequence
launched alone: the arithmetic is identical, and only the addresses differ.

The ``large`` cases (opt-in, ``-m large``; 8-12 GiB each) choose ``lead`` so that one base offset
reaches 2**31 elements; a dynamic-batching prefill step reaches them from 174763 tokens (GDP
800m's expanded keys) or 262144 tokens (GDN's chunk states).
"""

import pytest
import torch

from megatron.core.ssm.ops.gdp.chunk_h import chunk_gated_delta_product_fwd_h
from megatron.core.ssm.ops.gdp.chunk_o import chunk_gated_delta_product_fwd_o
from megatron.core.ssm.ops.gdp.common import CHUNK_SIZE

TAIL = CHUNK_SIZE + 1
"""Tokens of the checked sequence: a full chunk and a one-token chunk."""

# name: (K, V, lead, layout). Layouts: "after" puts the sequence after one sequence of `lead`
# zero chunks, "within" after `lead` zero chunks of its own, "many" after `lead` one-token
# sequences. The large leads put one offset at 2**31 elements:
CHUNK_H_LARGE = {
    # the chunk-state base row lead * K * V
    "states": (256, 256, 2**31 // (256 * 256), "after"),
    # the chunk-state row within the sequence
    "states-within": (256, 256, 2**31 // (256 * 256), "within"),
    # the key and w base token lead * CHUNK_SIZE * K
    "keys": (256, 16, 2**31 // (CHUNK_SIZE * 256), "after"),
    # the value and corrected-value base token lead * CHUNK_SIZE * V
    "values": (32, 256, 2**31 // (CHUNK_SIZE * 256), "after"),
    # the initial-state row lead * K * V
    "initial-states": (256, 256, 2**31 // (256 * 256), "many"),
}
CHUNK_O_LARGE = {
    # the query and key base token lead * CHUNK_SIZE * K
    "queries": (256, 16, 2**31 // (CHUNK_SIZE * 256)),
    # the value and output base token lead * CHUNK_SIZE * V
    "values": (32, 256, 2**31 // (CHUNK_SIZE * 256)),
}


def _requires_cuda():
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")


def _requires_free_memory(nbytes):
    free = torch.cuda.mem_get_info()[0]
    if free < nbytes:
        pytest.skip(f"needs {nbytes / 2**30:.1f} GiB of free GPU memory, {free / 2**30:.1f} free")


def _lengths(lead, layout):
    """Sequence lengths of the step; the checked sequence is the last one."""
    if layout == "after":
        return [lead * CHUNK_SIZE, TAIL]
    if layout == "within":
        return [lead * CHUNK_SIZE + TAIL]
    return [1] * lead + [TAIL]


def _cu_seqlens(lengths):
    cu = torch.tensor([0] + lengths, dtype=torch.int64).cumsum(0)
    return cu.to(dtype=torch.int32, device="cuda")


def _embed(x, tokens):
    """``x`` (``[1, TAIL, ...]``) as the last ``TAIL`` tokens of zeros ``[1, tokens, ...]``."""
    out = x.new_zeros(1, tokens, *x.shape[2:])
    out[:, tokens - TAIL :] = x
    return out


def _run_chunk_h(K, V, lead, layout):
    g0 = torch.Generator(device="cuda").manual_seed(1234)

    def randn(*shape, scale=0.1):
        return torch.randn(*shape, generator=g0, device="cuda") * scale

    k, w = randn(1, TAIL, 1, K).bfloat16(), randn(1, TAIL, 1, K).bfloat16()
    u = randn(1, TAIL, 1, V).bfloat16()
    # Within-chunk cumulative base-2 log decays.
    g = -torch.rand(1, TAIL, 1, generator=g0, device="cuda").mul(0.05)
    g[:, :CHUNK_SIZE] = g[:, :CHUNK_SIZE].cumsum(1)
    h0 = randn(1, 1, K, V, scale=1.0).bfloat16()
    if layout == "within":
        # Zero tokens leave a zero state unchanged, whatever their decay rounds to.
        h0.zero_()

    def run(k, w, u, g, h0, lengths, slot_indices):
        state = torch.full((2, 1, K, V), float("nan"), device="cuda")
        h, v_new, _ = chunk_gated_delta_product_fwd_h(
            k=k,
            w=w,
            u=u,
            g=g,
            initial_state=h0,
            cu_seqlens=_cu_seqlens(lengths),
            state=state,
            state_indices=torch.tensor(slot_indices, dtype=torch.int32, device="cuda"),
        )
        return h[0, -2:], v_new[0, -TAIL:], state[1]

    # Alone first, so the autotuner's choice for this shape is the one the step reuses.
    expected = run(k, w, u, g, h0, [TAIL], [1])

    lengths = _lengths(lead, layout)
    tokens = sum(lengths)
    k, w, u, g = (_embed(x, tokens) for x in (k, w, u, g))
    h0 = torch.cat([h0.new_zeros(len(lengths) - 1, 1, K, V), h0]) if len(lengths) > 1 else h0
    actual = run(k, w, u, g, h0, lengths, [-1] * (len(lengths) - 1) + [1])
    for name, a, e in zip(("chunk states", "corrected values", "final state"), actual, expected):
        assert torch.equal(a, e), f"{name} of the sequence after {lead} {layout} differ"


def _run_chunk_o(K, V, lead):
    g0 = torch.Generator(device="cuda").manual_seed(1234)

    def randn(*shape):
        return (torch.randn(*shape, generator=g0, device="cuda") * 0.1).bfloat16()

    q, k, v = randn(1, TAIL, 1, K), randn(1, TAIL, 1, K), randn(1, TAIL, 1, V)
    h = randn(1, 2, 1, K, V)
    g = -torch.rand(1, TAIL, 1, generator=g0, device="cuda").mul(0.05)
    g[:, :CHUNK_SIZE] = g[:, :CHUNK_SIZE].cumsum(1)

    def run(q, k, v, h, g, lengths):
        o = chunk_gated_delta_product_fwd_o(
            q=q, k=k, v=v, h=h, g=g, scale=K**-0.5, cu_seqlens=_cu_seqlens(lengths)
        )
        return o[0, -TAIL:]

    expected = run(q, k, v, h, g, [TAIL])

    lengths = _lengths(lead, "after")
    tokens = sum(lengths)
    q, k, v, g = (_embed(x, tokens) for x in (q, k, v, g))
    h = torch.cat([h.new_zeros(1, lead, 1, K, V), h], dim=1)  # the chunk states of the zeros
    actual = run(q, k, v, h, g, lengths)
    assert torch.equal(actual, expected), f"outputs of the sequence after {lead} chunks differ"


@pytest.mark.parametrize("layout", ["after", "within", "many"])
def test_chunk_states_after_other_tokens(layout):
    _requires_cuda()
    _run_chunk_h(K=64, V=32, lead=3, layout=layout)


def test_outputs_after_other_tokens():
    _requires_cuda()
    _run_chunk_o(K=64, V=32, lead=3)


@pytest.mark.large
@pytest.mark.parametrize("case", list(CHUNK_H_LARGE))
def test_chunk_states_past_int32_offsets(case):
    _requires_cuda()
    K, V, lead, layout = CHUNK_H_LARGE[case]
    lengths = _lengths(lead, layout)
    # bf16 k, w, u, v_new, chunk states (lead + 2 rows) and initial states, and temporaries.
    states = lead + 2 + len(lengths)
    _requires_free_memory(4 * sum(lengths) * (K + V) + 2 * states * K * V + (1 << 30))
    _run_chunk_h(K, V, lead, layout)


@pytest.mark.large
@pytest.mark.parametrize("case", list(CHUNK_O_LARGE))
def test_outputs_past_int32_offsets(case):
    _requires_cuda()
    K, V, lead = CHUNK_O_LARGE[case]
    tokens = lead * CHUNK_SIZE + TAIL
    # bf16 q, k, v, o and chunk states, and temporaries.
    _requires_free_memory(4 * tokens * (K + V) + 2 * (lead + 2) * K * V + (1 << 30))
    _run_chunk_o(K, V, lead)
