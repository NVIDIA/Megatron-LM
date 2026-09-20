# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Single-GPU synthetic selected-ID QSA forward/backward latency and memory probe.

Run with a free GPU, for example ``CUDA_VISIBLE_DEVICES=1 python .../bench_qsa_id_sparse.py``.
The ``recent`` route is optimistic for KV locality; the ``strided`` route
scatters the same number of unique selected blocks over the visible prefix.
"""

import gc
import time

import torch

from megatron.core.transformer.experimental_attention_variant.qsa_id_sparse import (
    qsa_sparse_attention_id,
    validate_qsa_block_ids,
)


def measure(seq_len, route, topk=512, dim=64, hq=4, hk=2, ratio=4):
    device = "cuda"
    q = torch.randn(1, hq, seq_len, dim, device=device, dtype=torch.bfloat16, requires_grad=True)
    k = torch.randn(1, hk, seq_len, dim, device=device, dtype=torch.bfloat16, requires_grad=True)
    v = torch.randn_like(k, requires_grad=True)
    positions = torch.arange(seq_len, device=device, dtype=torch.int32).unsqueeze(0)
    visible = (positions + 1) // ratio
    slots = torch.arange(topk, device=device, dtype=torch.int32)
    if route == "recent":
        ids = visible[:, :, None] - 1 - slots
        ids = ids.clamp_min(-1)
    else:
        step = torch.where(torch.gcd(visible, torch.full_like(visible, 97)) == 1, 97, 1)
        ids = (slots * step[:, :, None] + positions[:, :, None] * 131) % visible.clamp_min(1)[
            :, :, None
        ]
        ids = ids.masked_fill(slots >= visible[:, :, None], -1)
    ids = ids.to(torch.int32).contiguous()
    validate_qsa_block_ids(ids, positions, ratio)  # outside timed section
    grad = torch.randn_like(q)

    def step():
        out = qsa_sparse_attention_id(q, k, v, ids, positions, ratio=ratio)
        gradients = torch.autograd.grad(out, (q, k, v), grad)
        return out, gradients

    step()  # compile and warm up both kernels
    torch.cuda.synchronize()
    gc.collect()
    torch.cuda.empty_cache()
    baseline = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    out, gradients = step()
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    peak = torch.cuda.max_memory_allocated() - baseline
    print(
        f"length={seq_len} route={route} topk={topk} ratio={ratio} dim={dim} hq={hq} hk={hk} "
        f"fwd_bwd_s={elapsed:.4f} extra_peak_mib={peak / 2**20:.2f} "
        f"ids_mib={ids.numel() * ids.element_size() / 2**20:.2f} "
        f"finite={bool(torch.isfinite(out).all() and all(torch.isfinite(x).all() for x in gradients))}",
        flush=True,
    )


if __name__ == "__main__":
    for length in (4096, 8192, 16384):
        for pattern in ("recent", "strided"):
            measure(length, pattern)
