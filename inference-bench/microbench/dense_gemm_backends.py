"""Dense decode GEMMs of Ultra v3 at TP8: default cuBLAS vs TunableOp-tuned cuBLAS vs max-autotune.

Megatron's inference layers call torch.matmul(x, W.t()) (cuBLAS heuristic; `nvjet_*` kernels,
often split-K with a separate reduce at decode sizes). This times every dense GEMM of a decode step
at the per-GPU shapes of TP8 (weights sharded by TP; router and latent projections duplicated on
the local tokens), for c local tokens per GPU (M = c for local GEMMs, c x 8 for TP-gathered ones):

  default     torch.matmul -> cuBLAS heuristic (what runs today)
  autotune    torch.compile(mode="max-autotune-no-cudagraphs"): Inductor benchmarks its Triton GEMM
              templates against ATen per shape and keeps the faster
  tunableop   PyTorch TunableOp: benchmarks cuBLAS / cuBLASLt algorithms per shape (tuned eagerly
              first, then captured with tuning off, as a server would load a tuning file)
  fi_<b>      flashinfer.gemm.mm_bf16(x, W.t(), backend=<b>) for b in cudnn, cutlass, cublaslt,
              tinygemm, cute-dsl, tgv, auto; each warmed up once under flashinfer's autotune()
              before capture. bf16 only (the fp32 router is skipped).

Every variant runs inside a CUDA graph cycling through distinct weight copies (> L2), so each call
streams its weights from HBM. Reports us per call, % of HBM peak (weight bytes / time / 8 TB/s),
and the per-step total (x layer count).

    python inference-bench/microbench/dense_gemm_backends.py [--local-tokens 1 2 4 16] [--csv ...]
"""

import argparse
import csv
import os
import statistics

import torch

TP = 8
PEAK_TBS = 8.0
L2_BYTES = 256 * 2**20  # cycle through > L2 worth of weights

# (name, layers per step, M is TP-gathered, K, N, dtype)
SHAPES = [
    ("mamba_in_proj", 48, True, 8192, 35072 // TP, torch.bfloat16),
    ("mamba_out_proj", 48, True, 16384 // TP, 8192, torch.bfloat16),
    ("attn_qkv", 12, True, 8192, 8704 // TP, torch.bfloat16),
    ("attn_proj", 12, True, 8192 // TP, 8192, torch.bfloat16),
    ("router_fp32", 48, False, 8192, 512, torch.float32),
    ("latent_fc1", 48, False, 8192, 2048, torch.bfloat16),
    ("latent_fc2", 48, False, 2048, 8192, torch.bfloat16),
    ("shared_fc1", 48, True, 8192, 10240 // TP, torch.bfloat16),
    ("shared_fc2", 48, True, 10240 // TP, 8192, torch.bfloat16),
    ("lm_head", 1, True, 8192, 131072 // TP, torch.bfloat16),
]


def graph_time_us(fn, n_calls, reps=50):
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
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
        ts.append(a.elapsed_time(b) * 1000 / n_calls)
    return statistics.median(ts)


def build(m, k, n, dtype):
    wbytes = k * n * torch.tensor([], dtype=dtype).element_size()
    copies = max(2, -(-L2_BYTES // wbytes))
    x = torch.randn(m, k, device="cuda", dtype=dtype)
    ws = [torch.randn(n, k, device="cuda", dtype=dtype) * k**-0.5 for _ in range(copies)]
    outs = [torch.empty(m, n, device="cuda", dtype=dtype) for _ in range(copies)]
    return x, ws, outs, wbytes


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--local-tokens", type=int, nargs="+", default=[1, 2, 4, 16])
    ap.add_argument("--csv", default="inference-bench/microbench/dense_gemm_backends.csv")
    ap.add_argument("--variants", default="default,autotune,tunableop,fi_cudnn,fi_cutlass,"
                    "fi_cublaslt,fi_tinygemm,fi_cute-dsl,fi_tgv,fi_auto")
    args = ap.parse_args()

    print(f"GPU {torch.cuda.get_device_name()}, torch {torch.__version__}; Ultra v3 dense GEMMs at TP{TP}; "
          f"TF32 for fp32 matmul: {torch.backends.cuda.matmul.allow_tf32}")
    configs = []
    for c in args.local_tokens:
        for name, layers, gathered, k, n, dt in SHAPES:
            configs.append((c, name, layers, c * TP if gathered else c, k, n, dt))

    results = {}  # (c, name) -> dict
    bufs = {}
    for c, name, layers, m, k, n, dt in configs:
        bufs[(c, name)] = build(m, k, n, dt)

    def runner(mm, key):
        x, ws, outs, _ = bufs[key]

        # The compiled fn allocates its output (from the graph pool); no copy kernel to time.
        def fn():
            for w, o in zip(ws, outs):
                if mm is None:
                    torch.matmul(x, w.t(), out=o)
                else:
                    mm(x, w)
        return fn

    with torch.inference_mode():
        # 1. default cuBLAS (exactly what the inference layers call).
        for c, name, layers, m, k, n, dt in configs:
            key = (c, name)
            results[key] = {"c": c, "gemm": name, "layers": layers, "M": m, "K": k, "N": n,
                            "dtype": str(dt).split(".")[-1], "weight_mb": bufs[key][3] / 2**20}
            results[key]["default_us"] = graph_time_us(runner(None, key), len(bufs[key][1]))
            print(f"default   c={c:<3} {name:15s} M={m:<4} {results[key]['default_us']:7.2f} us", flush=True)

        variants = ["default_us"]
        wanted = args.variants.split(",")

        # FlashInfer mm_bf16 backends.
        fi_backends = [v[3:] for v in wanted if v.startswith("fi_")]
        if fi_backends:
            import flashinfer
            from flashinfer.autotuner import autotune

        for b in fi_backends:
            col = f"fi_{b}_us"
            variants.append(col)
            for c, name, layers, m, k, n, dt in configs:
                key = (c, name)
                if dt != torch.bfloat16:
                    continue
                x, ws, outs, _ = bufs[key]

                def mm(a, w, o, b=b):
                    return flashinfer.gemm.mm_bf16(a, w.t(), out=o, backend=b)

                def fn(x=x, ws=ws, outs=outs, mm=mm):
                    for w, o in zip(ws, outs):
                        mm(x, w, o)

                try:
                    with autotune(True):
                        mm(x, ws[0], outs[0])
                    torch.cuda.synchronize()
                    results[key][col] = graph_time_us(fn, len(ws))
                except Exception as e:  # noqa: BLE001
                    print(f"fi_{b:10s} c={c} {name}: {type(e).__name__}: {str(e)[:140]}")
                    results[key][col] = float("nan")
                print(f"fi_{b:9s} c={c:<3} {name:15s} M={m:<4} {results[key][col]:7.2f} us", flush=True)

        # 2. Inductor max-autotune (Triton templates vs ATen), one compile per shape.
        if "autotune" in wanted:
            variants.append("autotune_us")
            import torch._inductor.config as icfg
            icfg.max_autotune_gemm_backends = "TRITON,ATEN"
            for c, name, layers, m, k, n, dt in configs:
                key = (c, name)
                f = torch.compile(lambda a, b: a @ b.t(), mode="max-autotune-no-cudagraphs", dynamic=False)
                try:
                    f(bufs[key][0], bufs[key][1][0])  # compile + autotune outside capture
                    results[key]["autotune_us"] = graph_time_us(runner(f, key), len(bufs[key][1]))
                except Exception as e:  # noqa: BLE001
                    print(f"autotune  c={c} {name}: {type(e).__name__}: {str(e)[:160]}")
                    results[key]["autotune_us"] = float("nan")
                print(f"autotune  c={c:<3} {name:15s} M={m:<4} {results[key]['autotune_us']:7.2f} us", flush=True)
                torch._dynamo.reset()

    # 3. TunableOp: tune each shape eagerly, then capture with tuning off. Last, since enabling
    # TunableOp changes what plain torch.matmul does for the rest of the process.
    if "tunableop" in wanted:
        variants.append("tunableop_us")
        import torch.cuda.tunable as tunable

        with torch.inference_mode():
            tunable.enable(True)
            tunable.tuning_enable(True)
            tunable.set_max_tuning_duration(200)
            for c, name, layers, m, k, n, dt in configs:
                x, ws, outs, _ = bufs[(c, name)]
                torch.matmul(x, ws[0].t(), out=outs[0])
            torch.cuda.synchronize()
            tunable.tuning_enable(False)
            for c, name, layers, m, k, n, dt in configs:
                key = (c, name)
                results[key]["tunableop_us"] = graph_time_us(runner(None, key), len(bufs[key][1]))
                print(f"tunableop c={c:<3} {name:15s} M={m:<4} {results[key]['tunableop_us']:7.2f} us",
                      flush=True)

    rows = list(results.values())
    for r in rows:
        for v in variants:
            if v in r:
                r[v.replace("_us", "_pct_peak")] = 100 * r["weight_mb"] * 2**20 / (r[v] * 1e-6) / 1e12 / PEAK_TBS
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with open(args.csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, restval="")
        w.writeheader()
        w.writerows(rows)

    for c in args.local_tokens:
        print(f"\nlocal tokens per GPU c={c} (M = {c} local / {c * TP} TP-gathered); us per call (% of peak)")
        print(f"{'gemm':15s} {'M':>4} {'MB':>6} | " + " | ".join(f"{v[:-3]:>16}" for v in variants))
        print("  (per-step totals below include only GEMMs a variant ran; fi_* skip the fp32 router)")
        tot = {v: 0.0 for v in variants}
        for r in (r for r in rows if r["c"] == c):
            cells = []
            for v in variants:
                if v in r and r[v] == r[v]:
                    cells.append(f"{r[v]:7.2f} ({r[v.replace('_us', '_pct_peak')]:3.0f}%)")
                    tot[v] += r[v] * r["layers"]
                else:
                    cells.append(f"{'-':>16}")
            print(f"{r['gemm']:15s} {r['M']:4d} {r['weight_mb']:6.1f} | " + " | ".join(f"{x:>16}" for x in cells))
        print(f"{'per step (us)':15s} {'':>4} {'':>6} | " + " | ".join(f"{tot[v]:16.0f}" for v in variants))
    print(f"\ncsv -> {args.csv}")


if __name__ == "__main__":
    main()
