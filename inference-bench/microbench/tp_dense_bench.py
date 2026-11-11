"""Does wide TP help decode? Proxy of every dense GEMM in one Nemotron 3 Ultra decode step, per GPU.

Fixed load: c tokens per GPU (sequence parallel). TP-sharded GEMMs run on the TP group's gathered
M = c * TP tokens with the weight dimension sharded by TP; the MoE router and latent projections run
on the local c tokens with full (duplicated) weights, as in Megatron's MoE layer. Per step:

  48 Mamba    in_proj  [M, 8192] x [8192, 35072/TP]      out_proj [M, 16384/TP] x [16384/TP, 8192]
  12 attn     qkv      [M, 8192] x [8192, 8704/TP]       proj     [M, 8192/TP]  x [8192/TP, 8192]
  48 MoE      router   [c, 8192] x [8192, 512] (fp32)    latent fc1 [c, 8192] x [8192, 2048]
              latent fc2 [c, 2048] x [2048, 8192]        shared fc1 [M, 8192] x [8192, 10240/TP]
              shared fc2 [M, 10240/TP] x [10240/TP, 8192]
  1  LM head  [M, 8192] x [8192, 131072/TP]

torch.matmul(x, W.t()) like Megatron's inference layers, a distinct weight per layer (no L2 reuse),
the whole step in one CUDA graph (launch gaps included). Prints per-step time, a breakdown by GEMM
type, and achieved weight bandwidth. TP communication (all-gather / reduce-scatter of c*TP rows per
layer, growing with TP) is not included. TP > 8 is hypothetical for Ultra's Mamba (8 groups).

    python inference-bench/microbench/tp_dense_bench.py [--tp 1 2 4 8 16 32 64] [--tokens-per-gpu 1 2 4 ...]
"""

import argparse
import csv
import statistics

import torch

H, MAMBA_IN, MAMBA_INNER, QKV, ATTN_INNER = 8192, 35072, 16384, 8704, 8192
LATENT, SHARED, VOCAB = 2048, 10240, 131072
N_MAMBA, N_ATTN, N_MOE = 48, 12, 48
PEAK_TBS = 8.0


def gemm_specs(tp, c):
    """(name, count, M, K, N, dtype) per GEMM type for one decode step on one GPU."""
    m = c * tp
    bf, f32 = torch.bfloat16, torch.float32
    return [
        ("mamba_in_proj", N_MAMBA, m, H, MAMBA_IN // tp, bf),
        ("mamba_out_proj", N_MAMBA, m, MAMBA_INNER // tp, H, bf),
        ("attn_qkv", N_ATTN, m, H, QKV // tp, bf),
        ("attn_proj", N_ATTN, m, ATTN_INNER // tp, H, bf),
        ("router (fp32, local)", N_MOE, c, H, 512, f32),
        ("latent_fc1 (local, dup)", N_MOE, c, H, LATENT, bf),
        ("latent_fc2 (local, dup)", N_MOE, c, LATENT, H, bf),
        ("shared_fc1", N_MOE, m, H, SHARED // tp, bf),
        ("shared_fc2", N_MOE, m, SHARED // tp, H, bf),
        ("lm_head", 1, m, H, VOCAB // tp, bf),
    ]


def build(specs):
    """Per-layer weights and inputs. Values do not affect GEMM timing, so tensors are uninitialized."""
    ops = []
    for name, count, m, k, n, dt in specs:
        x = torch.randn(m, k, device="cuda", dtype=dt)
        for _ in range(count):
            w = torch.empty(n, k, device="cuda", dtype=dt).normal_()
            ops.append((name, x, w, torch.empty(m, n, device="cuda", dtype=dt)))
    return ops


def time_graph(fn, reps=20):
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        fn()
        fn()
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        fn()
    g.replay()
    torch.cuda.synchronize()
    ts = []
    for _ in range(reps):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record()
        g.replay()
        b.record()
        b.synchronize()
        ts.append(a.elapsed_time(b) * 1000)
    return statistics.median(ts)


def run_ops(ops):
    for _, x, w, out in ops:
        torch.matmul(x, w.t(), out=out)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tp", type=int, nargs="+", default=[1, 2, 4, 8, 16, 32, 64])
    ap.add_argument("--tokens-per-gpu", type=int, nargs="+", default=[1, 2, 4, 8, 16, 32, 64])
    ap.add_argument("--csv", default="inference-bench/microbench/tp_dense_bench.csv")
    args = ap.parse_args()

    torch.backends.cuda.matmul.allow_tf32 = False
    print(f"GPU {torch.cuda.get_device_name()}; dense GEMMs of one Ultra decode step per GPU, "
          f"fixed c tokens per GPU (M = c x TP for TP-sharded GEMMs)")
    results = []
    for tp in args.tp:
        for c in args.tokens_per_gpu:
            specs = gemm_specs(tp, c)
            try:
                ops = build(specs)
            except torch.cuda.OutOfMemoryError:
                print(f"TP{tp} c={c}: out of memory, skipped")
                continue
            wbytes = sum(w.numel() * w.element_size() for _, _, w, _ in ops)
            with torch.inference_mode():
                total = time_graph(lambda: run_ops(ops))
                parts = {}
                for name, *_ in specs:
                    sub = [o for o in ops if o[0] == name]
                    parts[name] = time_graph(lambda sub=sub: run_ops(sub))
            tbs = wbytes / (total * 1e-6) / 1e12
            print(f"\nTP{tp:<2} c={c:<3} (M={c * tp}): step {total / 1000:7.2f} ms, weights {wbytes / 1e9:6.2f} GB/GPU, "
                  f"{tbs:4.2f} TB/s ({100 * tbs / PEAK_TBS:3.0f}% of peak)")
            print("   " + ", ".join(f"{k} {v / 1000:.2f}" for k, v in parts.items()) + " (ms)")
            results.append({"tp": tp, "tokens_per_gpu": c, "m": c * tp, "step_ms": round(total / 1000, 3),
                            "weight_gb": round(wbytes / 1e9, 3), "tbs": round(tbs, 3),
                            **{f"{k}_ms": round(v / 1000, 3) for k, v in parts.items()}})
            del ops
            torch.cuda.empty_cache()

    with open(args.csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(results[0]))
        w.writeheader()
        w.writerows(results)

    print("\nDense GEMM time per decode step per GPU (ms), fixed tokens per GPU:")
    print(f"{'c/GPU':>5} | " + " | ".join(f"{'TP' + str(t):>7}" for t in args.tp))
    for c in args.tokens_per_gpu:
        cells = []
        for t in args.tp:
            r = [x for x in results if x["tp"] == t and x["tokens_per_gpu"] == c]
            cells.append(f"{r[0]['step_ms']:7.2f}" if r else f"{'-':>7}")
        print(f"{c:5d} | " + " | ".join(cells))
    print(f"\ncsv -> {args.csv}")


if __name__ == "__main__":
    main()
