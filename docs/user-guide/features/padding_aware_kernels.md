<!---
   Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
   NVIDIA CORPORATION and its licensors retain all intellectual property
   and proprietary rights in and to this software, related documentation
   and any modifications thereto. Any use, reproduction, disclosure or
   distribution of this software and related documentation without an express
   license agreement from NVIDIA CORPORATION is strictly prohibited.
-->

# Padding-Aware Kernels for Packed Sequences

Packed-sequence (THD) training with a sequence packing scheduler pads every per-rank buffer up to a
fixed capacity (`--max-seqlen-per-dp-cp-rank`), so that CUDA graphs and fixed-shape pipeline
communication see static shapes. With long, variable-length samples a large part of that buffer is
padding: on a 256K-token DeepSeek-V4 recipe with one sample per packing window, 34% of the rows are
padding. The attention kernels already skip those rows through `cu_seqlens`, but the dense GEMMs,
the norms and the cross entropy still process the whole buffer.

Padding-aware kernels keep the static buffer and skip the padding tail inside the kernels. A
**live-row bound**, the index of the last real row plus one rounded up to 128, is derived on the
device from the padding mask when a layer's attention or MLP helper is entered (one small kernel
each) and published to the kernels through a context. Because the bound is a device scalar computed
inside the (captured) forward, it changes per microbatch without host synchronization and works
under CUDA graph replay. Outputs and gradients of the padding rows are zero-filled, or, where noted
below, left as finite pass-through values, so the rest of the model never reads uninitialized memory.

## User Guide

### Basic Usage

```bash
# Row-bounded LM head (and MTP head) GEMMs. Default: off.
--padding-aware-lm-head

# Row-bounded Transformer Engine linears inside the transformer layers (and the DeepSeek-V4 grouped
# output projection), decided per module from the model shape. Default: off.
--padding-aware-dense-linear

# Padding fraction the static per-module cost model assumes. Default: 0.2.
# Set it to the recipe's expected tail-padding fraction.
--padding-aware-assumed-padding 0.34

# Single-pass Triton cross entropy over bf16 logits; skips the padding rows when a bound is
# published. Needs --cross-entropy-loss-fusion; has no effect without it.
--cross-entropy-loss-fusion
--cross-entropy-fusion-impl triton
```

The two boolean flags default to off; `--padding-aware-assumed-padding` defaults to 0.2. The pair
`--cross-entropy-loss-fusion --cross-entropy-fusion-impl triton` is also useful independently of the
padding-aware flags, on any THD or fixed-length recipe: it replaces the several fp32 passes of the
native fused cross entropy with one pass over the bf16 logits (5-7x fewer bytes, numerically equal to
the native path up to fp32 rounding) and removes the fp32 logits transients from the last pipeline
stage.

### What Is Covered

| Path | Enabled by | Mechanism |
|---|---|---|
| LM head, MTP head (`LinearCrossEntropyModule`) | `--padding-aware-lm-head` | One-group Transformer Engine grouped GEMM whose row count is a device scalar (`GroupedTensorStorage` with `first_dims`). Forward and input gradient over the live rows; the weight gradient is live-row-bounded (grouped GEMM into an fp32 scratch, then one add into `main_grad`) from 8192 rows per rank, dense over the full buffer below that (`MCORE_PADDING_AWARE_WGRAD` overrides). |
| `TELinear`, `TEColumnParallelLinear`, `TERowParallelLinear`, `TELayerNormColumnParallelLinear` | `--padding-aware-dense-linear` | Same one-group grouped GEMM through TE's grouped-linear path; the norm of the fused norm+linear still runs over the buffer |
| DeepSeek-V4 grouped output projection (`o_groups`) | `--padding-aware-dense-linear` | Grouped GEMM over strided members of the `[rows, groups, cols]` buffer (TE grouped tensor with a `leading_dim`), replacing the einsum |
| Cross entropy | `--cross-entropy-fusion-impl triton` (rows skipped when `--padding-aware-lm-head` publishes a bound) | Online-softmax Triton kernel, one program per row; tensor parallelism through all-gathered row statistics |
| mHC hyper-connection kernels, RoPE fusion | bound published by `--padding-aware-dense-linear` | Triton kernels are row-bounded (the padding rows of the residual streams pass through unchanged); of the cuTile kernels, the projection-RMS/compute-h forward and the h-aggregate backward are bounded, the cuTile sinkhorn, h-aggregate forward and h-post-BDA (selected only with `mhc_fused_backend=cutile`) are not |
| RMSNorm (`TENorm`), three-way gradient add of the hyper-connection, MTP arg-max | bound published by `--padding-aware-dense-linear` / `--padding-aware-lm-head`; off by default (see Kill Switches) | Row-bounded Triton kernels |

Not covered: the MoE experts (the token dispatcher already routes only real tokens, except inside the
MoE megakernel, which executes the shared experts for every local token), the op-fuser MLPs
(`--use-transformer-engine-op-fuser`), and `--cross-entropy-fusion-impl linear`.

### The Static Policy

Which `TELinear` runs padding-aware is decided **once, from the model shape**: the weight shape,
`--max-seqlen-per-dp-cp-rank`, whether FP8 is on and `--padding-aware-assumed-padding` feed a cost
model that compares the predicted GEMM saving against the fixed overhead of the grouped path
(`MCORE_PADDING_AWARE_MARGIN`, default 1.3). The decision never depends on the runtime row count, so
a captured graph always contains the same kernels. Each decision is logged once at initialization,
for example `padding-aware policy: TELinear [4096 x 8192] -> padding-aware (ok)` or
`-> dense (predicted saving ... < 1.3 x overhead ...)`.

A layer is eligible when it is not an expert linear (its rows are routed tokens), has no bias, no
GTP-sharded weight, no tensor-parallel communication inside the layer (TP=1 or duplicated weights),
no symmetric all-reduce, no userbuffers TP-communication overlap (`--tp-comm-overlap`), no delayed
weight-gradient compute, weight dimensions that are multiples of 32 and a buffer of at least 4096
rows. At call time the grouped path additionally requires bf16/fp16
activations or the MXFP8 recipe; with another FP8 recipe, under CPU offloading, FP8 calibration or
quantized inputs/outputs the call falls back to the dense TE forward (a once-per-module log line
reports the first fallback).

Only CUDA-graph runs (`--cuda-graph-impl` other than `none`) enable the dense-linear policy: the
grouped path costs about twice the CPU launch time of a dense linear, which eager steps pay on the
critical path while graph replay hides it. The LM head, the cross entropy and the bounded mHC/RoPE
kernels are not behind this gate.

### Kill Switches and Environment Variables

| Variable | Values | Default | Effect |
|---|---|---|---|
| `MCORE_PADDING_AWARE_SKIP` | comma list of `ce`, `mhc`, `rope`, `norm`, `add`, `argmax`, `arena` | unset = `norm,add,argmax` | Turns the named padding-aware paths off. The default off-set holds the row-bounded RMSNorm, the bounded hyper-connection gradient add and the bounded MTP arg-max, measured net-negative end to end (the bound bookkeeping costs more than the skipped rows save). The empty string engages everything; an explicit list is taken literally. |
| `MCORE_PADDING_AWARE_MARGIN` | float | `1.3` | Margin of the static cost model (predicted saving must exceed margin x overhead). |
| `MCORE_PADDING_AWARE_EAGER` | `1` | unset | Enables the dense-linear policy in eager (non-graph) runs. |
| `MCORE_PADDING_AWARE_FORCE` | `1` | unset | Enables every eligible layer regardless of the cost model (tests, attribution). |
| `MCORE_PADDING_AWARE_WGRAD` | `grouped` / `dense` (LM head), `scratch` / `accumulate` (TE linears) | automatic | Overrides the weight-gradient strategy. |
| `MCORE_PADDING_AWARE_QNORM` | `1` | unset | Opt-in row-bounded DeepSeek-V4 query RMSNorm (slower than the compiled norm on the 576-wide view; not governed by `MCORE_PADDING_AWARE_SKIP`). |
| `MCORE_PADDING_AWARE_LOG` | `1` | unset | Logs the first bounds (synchronizes; debugging only). |

### Requirements and Limitations

- Triton is required by every padding-aware path (the GEMM paths use it for the tail zero-fill and
  the bound); the mHC cuTile kernels need the cuTile compiler (SM100).
- Transformer Engine: `--padding-aware-lm-head` needs the grouped-tensor GEMM
  (`GroupedTensorStorage` + `general_grouped_gemm_for_grouped_tensor`, device-side `first_dims`);
  `--padding-aware-dense-linear` additionally needs TE's grouped-linear path
  (`_GroupedLinear._forward_grouped_tensor` with the current argument list; a release with a
  different one keeps the dense kernels); the DeepSeek-V4 grouped output projection additionally
  needs grouped tensors with a `leading_dim` (strided members). Each path falls back to the dense
  kernels when its API is missing; the dense-linear fallback is reported in the per-module policy
  log line. Without the TE grouped path no live-row bound is published, so the bounded kernels and
  the cross-entropy row skipping run over the full buffer (the Triton cross entropy keeps its
  single-pass saving).
- Sequence parallelism and deferred embedding weight gradients are not supported on the LM head
  path.
- Under contiguous context-parallel partitioning the padding tail lands on the last CP ranks; the
  fully live ranks that set the per-layer time save nothing. The kernel-level gain is therefore
  largest with per-rank windows (CP=1 or zigzag) and modest with contiguous CP. In that regime,
  shrinking `--max-seqlen-per-dp-cp-rank` (and `--seq-length`) to the longest sample is the larger
  lever and combines with these kernels.

### Measurements

GB200, DeepSeek-V4 proxy (16K rows per rank, CP1, lognormal 2K-16K samples, ~30% padding, chunk CUDA
graphs): -10.9% step time, -15.7 GiB peak memory. GB200, 256K DeepSeek-V4 recipe on 128 GPUs (CP16
contiguous, TP1, PP2, EP64, MXFP8, chunk CUDA graphs): -2.0% step time and -15.5 GiB on the last
pipeline stage with the flags above; the matched packing window mentioned under Requirements adds
-10.4% on its own.
