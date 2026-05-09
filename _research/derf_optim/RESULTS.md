# Derf throughput optimization — results

Three approaches to recover the throughput lost when `--normalization Derf`
forces the TE LayerNormColumnParallelLinear kernel to unfuse (TE's fused
norm-linear only knows LayerNorm and RMSNorm). Tested on the Aurora @ 1e-2
recipe at the 350M-ablation shape (24L / 1024H, seq 4096, GBS 128, MBS 16,
TP=1, bf16). Steady-state TFLOP/s/GPU averaged over iter 50–200 of a
200-iter run.

## Numbers

| variant | TFLOP/s/GPU | Δ vs unfused | Δ vs RMSNorm baseline | wall job |
| --- | ---: | ---: | ---: | --- |
| RMSNorm baseline (TE fused norm-linear) | **310** | +93 | 0 | (Aurora rank-1 leaderboard) |
| Unfused Derf (reference) | 217 | 0 | -93 | 2075996 (full 1B-token) |
| **Option 1** — torch.compile fused (Derf + nn.Linear) | **271** | **+54** | **-39** | 2076258 |
| Option 2 — Triton fused (Derf + matmul) | 170 | -47 | -140 | 2076274 |
| Option 3 — hand-rolled CUDA fused (Derf + matmul) | (timed out) | — | — | 2076308 |

Loss curves of all three options match the unfused reference within bf16
single-rounding noise (numerical correctness gated via
`_research/derf_optim/test_correctness.py`, both fp32 and bf16).

## Why each option performs the way it does

### Option 1 — torch.compile  (winner, +54 TFLOP/s)
A tiny TP=1 composite module wraps `Derf(x)` then `F.linear(...)` in a single
`torch.compile(fullgraph=True)` region. Inductor fuses the elementwise norm
into the matmul prologue (eliminating the duplicated activation read) but
crucially keeps **cuBLAS for the matmul itself**, so the heavy lifting still
runs on the well-tuned vendor kernel. Recovers ~58 % of the 93 TFLOP/s lost
from unfusing.

### Option 2 — Triton fused matmul + Derf prologue  (loses)
A from-scratch Triton kernel reads `x` once, applies Derf in registers, and
accumulates the matmul. Backward in PyTorch with post-norm recompute (saves
only `x`, matching the TE RMSNorm memory profile). Numerically clean.
Throughput collapses: my hand-rolled Triton matmul (BLOCK_M=BLOCK_N=BLOCK_K=64,
no Tensor Cores, no warp-specialized scheduling) runs at roughly half cuBLAS
speed for these shapes. The fusion savings can't compensate. Lesson: a
Triton norm-linear only wins if the Triton matmul is competitive with cuBLAS,
which on Hopper requires Tensor Core + careful tiling that is days of work.

### Option 3 — hand-rolled CUDA fused matmul + prologue  (did not finish)
`_research/derf_optim/cuda/derf_linear_kernel.cu`. First pass used SMEM-tiled
matmul with `dim3 block(64, 32) = 2048 threads/block` — **exceeds the 1024
threads/block hard limit** on every NVIDIA GPU since Kepler; kernel
launches were silently failing or hanging. Rewrote as a naive
thread-per-output-element kernel (128 threads/block, each thread loops over
K). Compiles, launches, but runs at <10 TFLOP/s — uncached global loads
of `x`, `W`, `gamma`, `beta` per K element times millions of output elements
exceeds the 20-minute SLURM wall by ~10×. To produce a real number we'd
need CUTLASS or a hand-written Tensor Core mainloop with proper tile
quantisation, which is multi-day scope and clearly out of band for an
autonomous session.

## Recommendation

**Ship Option 1.** It's a ~50-line composite module gated by
`APERTUS_DERF_OPTIM=compile`, recovers the bulk of the lost throughput,
keeps the spec wiring in pure Python, and inherits whatever future cuBLAS
improvements the container's PyTorch picks up.

The other two are only worth the engineering cost if (a) we move to a
shape where cuBLAS itself becomes the bottleneck (long-context or larger
hidden), or (b) a CUTLASS-based DerfLinear with TC support is available.
The right place to do the latter is upstream (TE), not in this fork.

## Reproducing

```bash
# Local correctness gate (CPU; fp32 + bf16 numerical match):
python -m _research.derf_optim.test_correctness --option 1 --dtype fp32
python -m _research.derf_optim.test_correctness --option 1 --dtype bf16
# Options 2 and 3 require CUDA, run on cluster instead.

# Cluster throughput (200-iter Aurora recipe, --normalization Derf):
sbatch _research/derf_optim/runs/o1-derf-compile-throughput.sbatch
sbatch _research/derf_optim/runs/o2-derf-triton-throughput.sbatch
sbatch _research/derf_optim/runs/o3-derf-cuda-throughput.sbatch  # incomplete impl
```

The spec wiring switches via `APERTUS_DERF_OPTIM=compile|triton|cuda`
(see `megatron/core/models/gpt/gpt_layer_specs.py:309-340`).
