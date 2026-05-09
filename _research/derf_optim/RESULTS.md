# DyT/Derf throughput optimisation

When `--normalization` is `DyT` or `Derf`, the spec must unfuse from TE's
`LayerNormColumnParallelLinear` (TE's fused norm-linear kernel only knows
LayerNorm and RMSNorm). At our 350M / bf16 / TP=1 / Aurora @ 1e-2 leader
shape this costs ~93 TFLOP/s/GPU.

`option1_compile.py` recovers most of it.

## Numbers

Steady-state TFLOP/s/GPU averaged over iter 50–200 of a 200-iter run.

| variant | TFLOP/s/GPU | Δ vs unfused | Δ vs RMSNorm baseline | wall job |
| --- | ---: | ---: | ---: | --- |
| RMSNorm baseline (TE pipeline) | **310** | +93 | 0 | (Aurora rank-1 leaderboard) |
| Unfused Derf reference | 217 | 0 | -93 | 2075996 (full 1B-token) |
| **Option 1 — torch.compile (Derf + linear)** | **271** | **+54** | **-39** | 2076258 |
| **Option 1 — torch.compile (DyT + linear)** | **268** | **+51** | **-42** | 2078500 |

Loss matches the unfused reference within bf16 single-rounding noise.
Gated by `_research/derf_optim/test_correctness.py` (forward + d/dx + all
parameter grads, fp32 + bf16).

## How Option 1 works

A TP=1 composite module wraps `(DyT|Derf)(x)` then `F.linear(...)` in a
single `torch.compile(fullgraph=True)` region. Inductor folds the
elementwise norm into a Triton-generated matmul prologue, so the
post-norm activation never lands in DRAM. cuBLAS does the heavy GEMM via
aten dispatch.

DyT and Derf share the implementation (templated `_compiled_dyt_linear` /
`_compiled_derf_linear` functions); the spec selects the right class at
build time via `make_qkv_class(config.normalization)`.

Wired via `APERTUS_DERF_OPTIM=compile`. See
`megatron/core/models/gpt/gpt_layer_specs.py` for the dispatch.

## Why this is the practical ceiling

Six approaches were measured before settling on Option 1. The full
comparison is in commit history (branch `feat/dyt-derf-norm`); the key
takeaways:

- **Inductor's prologue fusion is the actual lever.** Explicit
  Triton-norm-then-cuBLAS (Option 4, 230 TFLOP/s) lands well below
  Option 1 because the post-norm activation has to materialise in DRAM.
- **Kernel quality is not the lever.** Hand-tuned CUDA elementwise norm
  with LDG.128 vectorised loads (Option 6, 226 TFLOP/s) ties with Triton
  (230) — the compiler-generated kernel was already near-optimal.
- **The gemm dispatch is not the lever.** `F.linear` (Option 4: 230) and
  TE's `general_gemm` (Option 5: 231) hit the same cuBLAS path with the
  same workspace.
- **A from-scratch Triton fused matmul + Derf prologue underperforms cuBLAS**
  at this shape (Option 2 tuned: 202). Closing that needs Hopper-native TMA
  + persistent kernel + warp specialisation — days of work.

The remaining 39 TFLOP/s gap from Option 1 (271) to TE's RMSNorm baseline
(310) is **TE's specific kernel pipelining** — workspace reuse, activation
buffer caching across the norm/GEMM boundary, async stream scheduling.
Recovering it requires landing Derf inside TE's `ln_fwd_kernels.cuh`
template (build pipeline confirmed, ~2-3 days of focused surgery + build
cycles). Not pursued — Option 1's 271 is sufficient for production use.

## Reproducing

```bash
# Local correctness gate (CPU; fp32 + bf16 numerical match):
python -m _research.derf_optim.test_correctness --dtype fp32
python -m _research.derf_optim.test_correctness --dtype bf16

# Cluster throughput (200-iter Aurora recipe):
sbatch _research/derf_optim/runs/o1-derf-compile-throughput.sbatch  # --normalization Derf
sbatch _research/derf_optim/runs/o1-dyt-compile-throughput.sbatch   # --normalization DyT
```
