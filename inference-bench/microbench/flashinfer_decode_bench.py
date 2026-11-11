"""FlashInfer paged decode vs FA4 on real per-rank KV-length snapshots (same batches, shapes and
timing as fa4_decode_splits_bench.py: one attention layer, CUDA-graph timed, L2 flushed per call).

Kernels:
  fa4_1split    FA4 with num_splits=0 (its heuristic: 1 split at batch 128)
  fa4_8split    FA4 with num_splits=8 (mcore's floor)
  fi_fa2        FlashInfer BatchDecodeWithPagedKVCacheWrapper, backend fa2 (load-balanced plan())
  fi_fa2_tc     the same with use_tensor_cores=True (prefill-style kernel for GQA decode)
  fi_trtllm     flashinfer.decode.trtllm_batch_decode_with_kv_cache (trtllm-gen; dense block table,
                no plan())
plan() runs outside the timed graph, as in a server; its host time is reported as plan_us.

    python3 microbench/flashinfer_decode_bench.py --snapshots <kv_snapshots.json> --out <results.jsonl>
"""

import argparse
import json
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fa4_decode_splits_bench import attn, build, time_graph  # noqa: E402

import flashinfer  # noqa: E402
from flashinfer.decode import trtllm_batch_decode_with_kv_cache  # noqa: E402


def csr_from_table(table, lengths, page_size):
    """FlashInfer's page lists (indptr, indices, last_page_len) from a padded block table."""
    pages = [(n + page_size - 1) // page_size for n in lengths]
    indptr = torch.zeros(len(lengths) + 1, dtype=torch.int32)
    indptr[1:] = torch.cumsum(torch.tensor(pages, dtype=torch.int32), 0)
    indices = torch.cat([table[i, :p] for i, p in enumerate(pages)]).to(torch.int32)
    last = torch.tensor([n - (p - 1) * page_size for n, p in zip(lengths, pages)], dtype=torch.int32)
    return indptr.cuda(), indices, last.cuda()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--snapshots", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--only", nargs="*", default=None, help="snapshot names to run (default all)")
    ap.add_argument("--kernels", nargs="*", default=None, help="kernel labels to time (default all)")
    ap.add_argument("--variants", nargs="+", default=["real", "balanced"], choices=["real", "balanced"])
    ap.add_argument("--q-heads", type=int, default=8)
    ap.add_argument("--kv-heads", type=int, default=1)
    ap.add_argument("--head-dim", type=int, default=128)
    ap.add_argument("--page-size", type=int, default=256)
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--reps", type=int, default=5)
    args = ap.parse_args()

    snapshots = json.load(open(args.snapshots))
    gen = torch.Generator().manual_seed(0)
    flush = torch.empty(512 * 2**20, dtype=torch.uint8, device="cuda")
    flush_us = time_graph(lambda: flush.zero_(), args.iters, args.reps)
    workspace = torch.zeros(256 * 2**20, dtype=torch.uint8, device="cuda")
    trt_workspace = torch.zeros(256 * 2**20, dtype=torch.uint8, device="cuda")
    scale = args.head_dim ** -0.5
    print(f"flashinfer {flashinfer.__version__}; L2 flush {flush_us:.1f} us (subtracted)", flush=True)
    out = open(args.out, "w")
    for name, lengths in snapshots.items():
        if args.only and name not in args.only:
            continue
        variants = (("real", lengths), ("balanced", [round(sum(lengths) / len(lengths))] * len(lengths)))
        for variant, lens in variants:
            if variant not in args.variants:
                continue
            inputs = build(lens, args, gen)
            q, k, v, cu_q, seqused, table = inputs
            max_len = max(lens)
            ref = attn(inputs, max_len, 1)
            indptr, indices, last = csr_from_table(table.cpu(), lens, args.page_size)

            kernels = {
                "fa4_1split": (lambda: attn(inputs, max_len, 0), None),
                "fa4_8split": (lambda: attn(inputs, max_len, 8), None),
            }
            for label, tc in (("fi_fa2", False), ("fi_fa2_tc", True)):
                wrapper = flashinfer.BatchDecodeWithPagedKVCacheWrapper(
                    workspace, "NHD", use_tensor_cores=tc, backend="fa2")
                t0 = time.perf_counter()
                wrapper.plan(indptr, indices, last, args.q_heads, args.kv_heads, args.head_dim,
                             args.page_size, q_data_type=torch.bfloat16, kv_data_type=torch.bfloat16,
                             sm_scale=scale)
                torch.cuda.synchronize()
                plan_us = 1e6 * (time.perf_counter() - t0)
                kernels[label] = ((lambda w=wrapper: w.run(q, (k, v))), plan_us)
            seq_u = seqused.to(torch.int32)
            kernels["fi_trtllm"] = ((lambda: trtllm_batch_decode_with_kv_cache(
                q, (k, v), trt_workspace, table, seq_u, max_len, bmm1_scale=scale, bmm2_scale=1.0,
                kv_layout="NHD")), None)

            for label, (fn, plan_us) in kernels.items():
                if args.kernels and label not in args.kernels:
                    continue
                rec = {"snapshot": name, "variant": variant, "kernel": label, "max_len": max_len,
                       "mean_len": round(sum(lens) / len(lens)), "plan_us": plan_us}
                try:
                    err = (fn().reshape(ref.shape).float() - ref.float()).abs().max().item()

                    def timed(fn=fn):
                        flush.zero_()
                        fn()

                    rec.update(us=time_graph(timed, args.iters, args.reps) - flush_us, max_abs_err=err)
                except Exception as e:  # noqa: BLE001
                    rec.update(us=None, error=f"{type(e).__name__}: {str(e)[:300]}")
                out.write(json.dumps(rec) + "\n")
                out.flush()
                msg = (f"{rec['us']:8.1f} us  err {rec['max_abs_err']:.1e}" if rec.get("us") is not None
                       else rec["error"])
                plan = f"  plan {plan_us:.0f} us (host)" if plan_us else ""
                print(f"{name:28s} {variant:8s} {label:11s} {msg}{plan}", flush=True)
            del inputs, q, k, v
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
