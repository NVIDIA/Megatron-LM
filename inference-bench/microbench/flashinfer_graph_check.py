"""Checks FlashInfer's paged decode wrapper for the mcore integration, against FA4 on the same batch:

1. zero-length (padding) requests: is the output finite, and are the real rows unaffected?
2. use_cuda_graph=True: capture run() once, re-plan with different lengths (same batch size),
   replay, compare with FA4 on the new lengths.

    python3 microbench/flashinfer_graph_check.py
"""

import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fa4_decode_splits_bench import attn, build  # noqa: E402

import flashinfer  # noqa: E402


class Args:
    q_heads, kv_heads, head_dim, page_size = 8, 1, 128, 256


def page_lists(table, lengths, page_size):
    """indptr, indices, last_page_len (CPU int32) from a padded CPU block table; a zero-length
    request gets 0 pages and last_page_len 0."""
    lengths = torch.tensor(lengths, dtype=torch.int32)
    pages = (lengths + page_size - 1) // page_size
    indptr = torch.zeros(len(lengths) + 1, dtype=torch.int32)
    indptr[1:] = torch.cumsum(pages, 0)
    cols = torch.arange(table.shape[1], dtype=torch.int32)
    indices = table[cols[None, :] < pages[:, None]].to(torch.int32)
    last = torch.where(pages > 0, lengths - (pages - 1) * page_size, torch.zeros_like(lengths))
    return indptr, indices, last.to(torch.int32)


def main():
    args, gen = Args(), torch.Generator().manual_seed(0)
    scale = args.head_dim ** -0.5
    workspace = torch.zeros(256 * 2**20, dtype=torch.uint8, device="cuda")

    def plan(w, table, lengths):
        indptr, indices, last = page_lists(table.cpu(), lengths, args.page_size)
        w.plan(indptr, indices, last, args.q_heads, args.kv_heads, args.head_dim, args.page_size,
               q_data_type=torch.bfloat16, kv_data_type=torch.bfloat16, sm_scale=scale)

    # 1. zero-length padding rows (eager wrapper).
    real = [3000, 70000, 512, 9000]
    inputs = build(real + [1, 1], args, gen)  # 2 padding rows; their lengths are set to 0 below
    q, k, v, cu_q, seqused, table = inputs
    lengths = real + [0, 0]
    w = flashinfer.BatchDecodeWithPagedKVCacheWrapper(workspace, "NHD", use_tensor_cores=True,
                                                      backend="fa2")
    plan(w, table, lengths)
    out = w.run(q, (k, v))
    seqused.copy_(torch.tensor(lengths, dtype=torch.int32))
    ref = attn(inputs, max(lengths), 1)
    n = len(real)
    print("1. padding rows: finite =", bool(torch.isfinite(out[n:]).all()),
          "| padding max |out| =", out[n:].abs().max().item(),
          "| real rows max err vs FA4 =", (out[:n].float() - ref[:n].float()).abs().max().item())

    # 2. CUDA graph: capture once, re-plan, replay.
    bs, max_pages = 128, 1024
    g1 = [int(x) for x in torch.randint(100, 20000, (bs,), generator=gen)]
    g2 = [int(x) for x in torch.randint(100, 200000, (bs,), generator=gen)]
    g2[:5] = [0] * 5  # some padding rows in the second plan
    inputs = build([max(a, b, 1) for a, b in zip(g1, g2)], args, gen)
    q, k, v, cu_q, seqused, table = inputs
    wg = flashinfer.BatchDecodeWithPagedKVCacheWrapper(
        workspace, "NHD", use_cuda_graph=True, use_tensor_cores=True, backend="fa2",
        paged_kv_indptr_buffer=torch.zeros(bs + 1, dtype=torch.int32, device="cuda"),
        paged_kv_indices_buffer=torch.zeros(bs * max_pages, dtype=torch.int32, device="cuda"),
        paged_kv_last_page_len_buffer=torch.zeros(bs, dtype=torch.int32, device="cuda"))
    out = torch.empty_like(q)
    plan(wg, table, g1)
    wg.run(q, (k, v), out=out)  # warm up outside capture
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        wg.run(q, (k, v), out=out)
    for name, lengths in (("plan 1", g1), ("plan 2 (longer, 5 empty rows)", g2)):
        plan(wg, table, lengths)
        graph.replay()
        torch.cuda.synchronize()
        seqused.copy_(torch.tensor(lengths, dtype=torch.int32))
        ref = attn(inputs, max(lengths), 1)
        live = torch.tensor([x > 0 for x in lengths], device="cuda")
        err = (out[live].float() - ref[live].float()).abs().max().item()
        print(f"2. graph replay after {name}: max err vs FA4 = {err:.2e}, "
              f"finite = {bool(torch.isfinite(out).all())}")


if __name__ == "__main__":
    main()
