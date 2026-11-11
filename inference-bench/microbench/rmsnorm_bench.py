"""RMSNorm kernels at decode-sized token counts: TE vs FlashInfer vs Triton vs torch.compile.

The inference path calls TE's rmsnorm_fwd ~108 times per decode step (Mamba in_proj, attention
qkv, MoE pre-norm, final norm). On Ultra (hidden 8192) the trace shows TE's untuned
`rmsnorm_fwd_general_kernel` at ~8.2 us per call for ~2 rows. This measures the alternatives the
way they run in the engine: many calls replayed from one CUDA graph, so the per-call time includes
the in-graph launch gap but no CPU launch overhead.

    python inference-bench/microbench/rmsnorm_bench.py [--hidden 8192 4096 2688] [--rows 1 2 4 ...]
"""

import argparse
import statistics

import torch
import triton
import triton.language as tl

EPS = 1e-5


# ── Candidates ────────────────────────────────────────────────────────────────


def make_te_fn():
    import transformer_engine.pytorch.cpp_extensions as tex
    from transformer_engine.pytorch.constants import TE_DType

    def te_fn(x, w, out):
        # Same call as megatron/core/tensor_parallel/inference_layers.py:_rms_norm_kernel (its TE fallback).
        y, _, _ = tex.rmsnorm_fwd(x, w, EPS, None, None, TE_DType[x.dtype], 16, False)
        return y

    return te_fn


def make_flashinfer_fn():
    import flashinfer

    def fi_fn(x, w, out):
        return flashinfer.norm.rmsnorm(x, w, eps=EPS, out=out)

    return fi_fn


@triton.jit
def _rmsnorm_kernel(x_ptr, w_ptr, y_ptr, stride_x, stride_y, N, eps, BLOCK_N: tl.constexpr):
    # One program per row; the whole row lives in registers (fp32 accumulation).
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_N)
    mask = cols < N
    x = tl.load(x_ptr + row * stride_x + cols, mask=mask, other=0.0).to(tl.float32)
    rstd = tl.rsqrt(tl.sum(x * x, axis=0) / N + eps)
    w = tl.load(w_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    tl.store(y_ptr + row * stride_y + cols, (x * rstd * w).to(y_ptr.dtype.element_ty), mask=mask)


def make_triton_fn(num_warps):
    def tr_fn(x, w, out):
        M, N = x.shape
        _rmsnorm_kernel[(M,)](
            x, w, out, x.stride(0), out.stride(0), N, EPS,
            BLOCK_N=triton.next_power_of_2(N), num_warps=num_warps,
        )
        return out

    return tr_fn


def reference(x, w):
    xf = x.float()
    return (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + EPS) * w.float()).to(x.dtype)


def make_compile_fn():
    compiled = torch.compile(reference, dynamic=False, fullgraph=True)
    return lambda x, w, out: compiled(x, w)


# ── Timing ────────────────────────────────────────────────────────────────────


def time_in_graph(fn, M, N, calls, reps):
    """Per-call time when `calls` norms on distinct buffers are replayed from one CUDA graph."""
    # Distinct inputs/weights per call (like distinct layers); inputs are hot in L2, as they are
    # right after the producing kernel in the model.
    xs = [torch.randn(M, N, device="cuda", dtype=torch.bfloat16) for _ in range(calls)]
    ws = [torch.rand(N, device="cuda", dtype=torch.bfloat16) + 0.5 for _ in range(calls)]
    outs = [torch.empty(M, N, device="cuda", dtype=torch.bfloat16) for _ in range(calls)]
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for i in range(3):  # warmup (compile / autotune / lazy init) outside capture
            fn(xs[i], ws[i], outs[i])
    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for i in range(calls):
            fn(xs[i], ws[i], outs[i])
    graph.replay()
    torch.cuda.synchronize()
    times = []
    for _ in range(reps):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        graph.replay()
        end.record()
        end.synchronize()
        times.append(start.elapsed_time(end) * 1000 / calls)
    return statistics.median(times)


def check(fn, M, N):
    x = torch.randn(M, N, device="cuda", dtype=torch.bfloat16) * 3
    w = torch.rand(N, device="cuda", dtype=torch.bfloat16) + 0.5
    out = torch.empty_like(x)
    y = fn(x, w, out)
    ref = reference(x, w)
    return (y.float() - ref.float()).abs().max().item()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hidden", type=int, nargs="+", default=[8192, 4096, 2688])
    parser.add_argument("--rows", type=int, nargs="+",
                        default=[1, 2, 4, 8, 16, 32, 64, 128, 512, 2048])
    parser.add_argument("--calls", type=int, default=108, help="norm calls per graph (per step)")
    parser.add_argument("--reps", type=int, default=50)
    args = parser.parse_args()

    candidates = {"te": make_te_fn}
    for name, maker in (("flashinfer", make_flashinfer_fn), ("compile", make_compile_fn)):
        candidates[name] = maker
    for w in (4, 8, 16, 32):
        candidates[f"triton_w{w}"] = (lambda w=w: make_triton_fn(w))

    fns = {}
    for name, maker in candidates.items():
        try:
            fns[name] = maker()
        except Exception as e:  # noqa: BLE001 - report and skip unavailable backends
            print(f"skip {name}: {type(e).__name__}: {e}")

    print(f"GPU: {torch.cuda.get_device_name()}, {args.calls} calls per graph, median of "
          f"{args.reps} replays; per-call us (max |err| vs fp32 reference at the largest M)")
    for N in args.hidden:
        print(f"\nhidden {N}")
        header = f"{'rows':>6} | " + " ".join(f"{n:>11}" for n in fns)
        print(header)
        print("-" * len(header))
        for M in args.rows:
            cells = []
            for name, fn in fns.items():
                try:
                    cells.append(f"{time_in_graph(fn, M, N, args.calls, args.reps):11.2f}")
                except Exception as e:  # noqa: BLE001
                    cells.append(f"{'err':>11}")
                    print(f"  {name} M={M}: {type(e).__name__}: {e}")
            print(f"{M:6d} | " + " ".join(cells))
        errs = []
        for name, fn in fns.items():
            try:
                errs.append(f"{name}={check(fn, max(args.rows), N):.3g}")
            except Exception as e:  # noqa: BLE001
                errs.append(f"{name}=err({type(e).__name__})")
        print("max |err|: " + ", ".join(errs))


if __name__ == "__main__":
    main()
