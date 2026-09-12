# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Exercise teacher-LSE addressing beyond the signed int32 element boundary."""

import pytest
import torch

from megatron.core.transformer.experimental_attention_variant.csa_utils import (
    csa2_indexer,
    csa_teacher_lse,
)

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")


@pytest.fixture(scope="module")
def long_teacher_query():
    pytest.importorskip("triton")
    if not torch.cuda.is_bf16_supported():
        pytest.skip("BF16 is not supported")
    heads, dim = 64, 512
    rows = 65538
    required_bytes = rows * heads * dim * 2 + 512 * 1024**2
    if torch.cuda.mem_get_info()[0] < required_bytes:
        pytest.skip("Large-offset teacher LSE regression requires 4.5 GiB of free GPU memory")

    # Keep one real allocation across cases; only sampled rows need random values.
    # Two sequences of 32769 rows also cover overflow caused by batch flattening.
    query = torch.zeros((rows, heads, dim), device="cuda", dtype=torch.bfloat16)
    sample_rows = torch.tensor([0, 32768, 32769, 65535, 65536, 65537], device="cuda")
    generator = torch.Generator(device="cuda").manual_seed(421)
    samples = torch.randn(
        (sample_rows.numel(), heads, dim), device="cuda", dtype=query.dtype, generator=generator
    )
    query.index_copy_(0, sample_rows, samples)
    keys = torch.randn((4, dim), device="cuda", dtype=query.dtype, generator=generator)
    sink = torch.linspace(-1, 1, heads, device="cuda")
    return query, sample_rows, samples.float(), keys, sink


@pytest.mark.parametrize("mode", [0, 1], ids=["dense", "candidates"])
def test_csa2_teacher_lse_large_query_offsets(long_teacher_query, mode):
    query, sample_rows, samples, keys, sink = long_teacher_query
    rows, heads, dim = query.shape
    scale = dim**-0.5
    window = torch.zeros((rows, 1), device="cuda", dtype=torch.int32)
    non_compressed = csa_teacher_lse.fused_csa_window_lse(query, keys, sink, window, scale)
    window_logits = (samples @ keys[0].float()) * scale
    expected_window = torch.logaddexp(window_logits, sink[None, :])
    torch.testing.assert_close(non_compressed[sample_rows], expected_window, atol=2e-3, rtol=2e-3)

    starts = torch.zeros(rows, device="cuda", dtype=torch.int32)
    visible = torch.full_like(starts, 4)
    # Candidate mode selects the final block of two keys, independently of dense mode.
    ids = torch.ones((rows, 1), device="cuda", dtype=torch.int32)
    counts = torch.ones_like(starts)
    actual = torch.empty_like(non_compressed)
    width, block_size = (4, 1) if mode == 0 else (2, 2)
    csa2_indexer._teacher_lse_kernel[(rows, heads // 16)](
        query,
        keys,
        non_compressed,
        ids,
        counts,
        starts,
        visible,
        actual,
        heads,
        dim,
        width,
        ids.shape[-1],
        mode,
        block_size,
        scale,
        16,
        64,
        num_stages=1,
    )

    selected_keys = keys if mode == 0 else keys[2:]
    logits = (samples @ selected_keys.float().T) * scale
    expected = torch.logaddexp(expected_window, logits.logsumexp(-1))
    torch.testing.assert_close(actual[sample_rows], expected, atol=2e-3, rtol=2e-3)


@pytest.mark.parametrize("layout", ["sbhd", "thd"])
def test_shared_teacher_lse_large_query_offsets(long_teacher_query, layout):
    query, sample_rows, samples, keys, sink = long_teacher_query
    rows, heads, dim = query.shape
    seq_len, batch, ratio = rows // 2, 2, 16384
    scale = dim**-0.5
    window = torch.zeros((rows, 1), device="cuda", dtype=torch.int32)
    if layout == "sbhd":
        compressed = keys.reshape(batch, 2, dim)
        metadata = dict(batch_size=batch, seqlen_q=seq_len)
        sample_sequences = sample_rows % batch
        sample_positions = sample_rows // batch
    else:
        compressed = keys
        metadata = dict(
            cu_seqlens_q=torch.tensor([0, seq_len, rows], device="cuda", dtype=torch.int32),
            cu_seqlens_k=torch.tensor([0, 2, 4], device="cuda", dtype=torch.int32),
            max_seqlen_q=seq_len,
            max_seqlen_k=2,
        )
        sample_sequences = sample_rows // seq_len
        sample_positions = sample_rows % seq_len

    actual = csa_teacher_lse.fused_csa_teacher_lse(
        query, keys, compressed, sink, window, scale, ratio, **metadata
    )
    if layout == "sbhd":
        actual = actual[sample_sequences, sample_positions]
    else:
        actual = actual[sample_rows]

    sample_keys = keys.reshape(batch, 2, dim)[sample_sequences].float()
    logits = torch.einsum("rhd,rkd->rhk", samples, sample_keys) * scale
    visible = (sample_positions + 1) // ratio
    logits.masked_fill_(
        torch.arange(2, device="cuda")[None, None, :] >= visible[:, None, None], float("-inf")
    )
    window_logits = (samples @ keys[0].float()) * scale
    expected = torch.logaddexp(torch.logaddexp(window_logits, sink[None, :]), logits.logsumexp(-1))
    torch.testing.assert_close(actual, expected, atol=2e-3, rtol=2e-3)
