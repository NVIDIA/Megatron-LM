# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
"""Layout check for the chunked vocabulary wgrad reduce-scatter of a GTP vocabulary weight.

Each rank holds a different micro-batch (grad_output_r, input_r); the gtp_remat wgrad of a
[V, H] weight is sum_r grad_output_r^T @ input_r, and rank r owns rows [r*S, (r+1)*S) of it.
``fill_wgrad_chunk`` + one reduce-scatter per chunk must reproduce that shard exactly, padding
rows included.
"""

import pytest
import torch
import torch.distributed as dist

from megatron.core.tensor_parallel.generalized_tensor_parallelism import fill_wgrad_chunk
from tests.unit_tests.test_utilities import Utils


def _gemm(out, grad_output, total_input):
    out.copy_(grad_output.t().float() @ total_input.float())


@pytest.mark.parametrize("vocab,chunks", [(128, 4), (120, 4), (128, 1), (96, 8)])
def test_chunked_reduce_scatter_matches_full_shard(vocab, chunks):
    Utils.initialize_model_parallel(tensor_model_parallel_size=1)
    try:
        group = dist.group.WORLD
        N, rank = group.size(), group.rank()
        T, H = 64, 32
        padded = ((vocab + N - 1) // N) * N  # GTP pads the tail to a multiple of N rows
        S = padded // N
        if S % chunks:
            pytest.skip("shard rows must divide into chunks")
        b = S // chunks
        gen = torch.Generator(device="cuda").manual_seed(1234 + rank)
        go = torch.randn(T, vocab, device="cuda", dtype=torch.bfloat16, generator=gen)
        x = torch.randn(T, H, device="cuda", dtype=torch.bfloat16, generator=gen)
        # Reference: full wgrad (padded with zero rows), summed over ranks, this rank's rows.
        full = torch.zeros(padded, H, device="cuda", dtype=torch.float32)
        full[:vocab] = go.t().float() @ x.float()
        dist.all_reduce(full, group=group)
        expected = full[rank * S : (rank + 1) * S]
        out = torch.empty(S, H, device="cuda", dtype=torch.float32)
        buf = torch.empty(N * b, H, device="cuda", dtype=torch.float32)
        cols = None
        for k in range(chunks):
            _, cols = fill_wgrad_chunk(buf, go, x, N, S, k, vocab, _gemm, cols_buf=cols)
            dist.reduce_scatter_tensor(out[k * b : (k + 1) * b], buf, group=group)
        torch.testing.assert_close(out, expected, rtol=1e-5, atol=1e-4)
    finally:
        Utils.destroy_model_parallel()
