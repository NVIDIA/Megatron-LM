<!---
   Copyright (c) 2022-2026, NVIDIA CORPORATION. All rights reserved.
   NVIDIA CORPORATION and its licensors retain all intellectual property
   and proprietary rights in and to this software, related documentation
   and any modifications thereto. Any use, reproduction, disclosure or
   distribution of this software and related documentation without an express
   license agreement from NVIDIA CORPORATION is strictly prohibited.
-->

# Low-Precision Training Guide

This guide explains how to train with sub-BF16 precisions (FP8 and FP4) in
Megatron Core, summarizes the trade-offs between the available recipes, and
provides ready-to-use configurations for the most common scenarios.

Low-precision support in Megatron Core is built on
[Transformer Engine (TE)](https://github.com/NVIDIA/TransformerEngine). Megatron
exposes TE recipes through `TransformerConfig` fields (and matching CLI flags)
so you can switch recipes without touching model code.

## When to Use Low Precision

Compared to a BF16 baseline, low precision delivers:

- **Higher training throughput** — TE FP8/FP4 GEMMs on Hopper, Ada, and
  Blackwell hit substantially higher TFLOPS than BF16.
- **Lower activation and parameter memory** — Activations and (optionally)
  parameters are stored in 8 or 4 bits.
- **Less DP communication bandwidth** — With `--fp8-param-gather` /
  `--fp4-param-gather`, the parameter all-gather is performed in the quantized
  dtype.

The trade-off is **numerics**: not every model converges identically to BF16
under every recipe. The recipe and configuration choices below let you balance
throughput, memory, and stability.

## Hardware and Recipe Compatibility

| Recipe (`--fp8-recipe` / `--fp4-recipe`) | Format flag | Required architecture | Minimum TE | Typical use |
| --- | --- | --- | --- | --- |
| `delayed` (default) | `--fp8-format {e4m3,hybrid}` | Hopper, Ada, Blackwell | 1.0 | Stable baseline FP8, supports CPU offload |
| `tensorwise` | `--fp8-format {e4m3,hybrid}` | Hopper, Ada, Blackwell | 1.11 | Current-scaling FP8, no amax history |
| `mxfp8` | `--fp8-format e4m3` | Blackwell | 2.0 | Block-scaled E4M3, fastest on B200/GB200 |
| `blockwise` | `--fp8-format {e4m3,hybrid}` | Hopper, Blackwell | 2.0 | 1×128 / 128×128 block scaling (DeepSeek-style) |
| `nvfp4` | `--fp4-format e2m1` | Blackwell | 2.7.0.dev0 | NVFP4 block-scaling for max throughput / memory |
| `custom` | `--fp8-format` or `--fp4-format` | n/a | n/a | User-supplied quantizer via `--fp8-quantizer-factory` / `--fp4-quantizer-factory` |

`--fp8-format` and `--fp4-format` are mutually exclusive; you can only enable
one of FP8 or FP4 per run.

## Recipe Reference

Each recipe is selected by passing both the format flag (which turns the recipe
on) and the recipe name (which picks the scaling strategy).

### `delayed` — Delayed scaling FP8

The historical default. The scaling factor is computed from a rolling amax
history.

Minimum config:

```bash
--fp8-format hybrid \
--fp8-recipe delayed \
--fp8-amax-history-len 1024 \
--fp8-amax-compute-algo max
```

Optional knobs: `--fp8-margin` (default `0`), `--fp8-param-gather`,
`--fp8-wgrad` (set to `False` to keep `wgrad` in higher precision).

When to use:

- You need the broadest hardware support (Hopper and newer).
- You require **FP8 + optimizer CPU offload** (see
  [optimizer_cpu_offload.md](optimizer_cpu_offload.md)) — only `delayed` is
  supported in that mode.
- You want the most validated recipe across community-trained models.

### `tensorwise` — Per-tensor current scaling FP8

Computes the scale from the current tensor, removing the amax history buffer
and the warmup-window dependency of `delayed`.

Minimum config:

```bash
--fp8-format hybrid \
--fp8-recipe tensorwise \
--fp8-param-gather
```

When to use:

- You want a simpler, history-free FP8 recipe on Hopper/Ada/Blackwell.
- You see instability during the first hundred steps with `delayed` (typical
  amax-warmup behavior).
- Production RL / SFT workloads. See
  [examples/rl/model_configs/nemotron5_56b.sh](../../../examples/rl/model_configs/nemotron5_56b.sh)
  for a complete recipe.

### `mxfp8` — Block-scaled E4M3 (Blackwell)

Microscaled FP8: a power-of-two scale per 1×32 block. Native to Blackwell
Tensor Cores.

Minimum config:

```bash
--fp8-format e4m3 \
--fp8-recipe mxfp8 \
--fp8-param-gather \
--reuse-grad-buf-for-mxfp8-param-ag
```

`--reuse-grad-buf-for-mxfp8-param-ag` reuses the gradient buffer for the
parameter all-gather and is strongly recommended whenever you combine `mxfp8`
with `--fp8-param-gather` — Megatron will warn otherwise (see
[`OptimizerConfig`](../../../megatron/core/optimizer/optimizer_config.py)).

Optional: `--keep-fp8-transpose-cache` keeps the row-wise / column-wise FP8
data on the device to avoid recomputation under fine-grained activation
offloading.

When to use:

- Blackwell hardware (B200, GB200).
- You want the best raw throughput on Blackwell for both dense and MoE models.

See [examples/gptoss/02_train.sh](../../../examples/gptoss/02_train.sh) for a
worked end-to-end example.

### `blockwise` — DeepSeek-style block scaling

1×128 activation blocks paired with 128×128 weight blocks. Originally
popularized by the DeepSeek-V3 recipe.

Minimum config:

```bash
--fp8-format e4m3 \
--fp8-recipe blockwise \
--fp8-param-gather
```

When to use:

- You are reproducing or fine-tuning a model trained with the DeepSeek FP8
  recipe.
- You need better numerical headroom for outlier channels than `tensorwise` /
  `mxfp8` while staying in FP8.

See [examples/rl/model_configs/nemotron5p5_12b_H.sh](../../../examples/rl/model_configs/nemotron5p5_12b_H.sh).

### `nvfp4` — NVFP4 block scaling (Blackwell)

FP4 with NVFP4 block scaling. Provides additional memory and bandwidth
savings on top of FP8.

Minimum config:

```bash
--fp4-format e2m1 \
--fp4-param-gather
```

Requires Transformer Engine ≥ `2.7.0.dev0`.

When to use:

- Blackwell hardware where FP4 GEMM kernels are available.
- You have validated your model under FP4 QAT or are willing to apply
  `--first-last-layers-bf16` and additional stability mitigations (see below).

NVFP4 is an actively evolving recipe in Megatron Core. Track the
[2026 Q2 roadmap](https://github.com/NVIDIA/Megatron-LM/issues/4997) for the
latest fixes (e.g. NVFP4 param gather, MXFP8 / NVFP4 mixed-precision work).

### `custom` — User-supplied quantizer

Set `--fp8-recipe custom` (or `--fp4-recipe custom`) and provide a Python
import path to a quantizer factory via `--fp8-quantizer-factory` or
`--fp4-quantizer-factory`. Use this when you need per-module overrides — also
see `--te-precision-config-file` for declarative per-module precision control.

## Numerical Stability Knobs

Beyond picking a recipe, the following options are the main levers when a run
diverges or shows higher loss than the BF16 baseline.

| Flag | Effect | When to enable |
| --- | --- | --- |
| `--first-last-layers-bf16` | Keep the first / last `N` transformer blocks in BF16. | First defense against early-training instability under any FP8/FP4 recipe. |
| `--num-layers-at-start-in-bf16 N` / `--num-layers-at-end-in-bf16 N` | Controls how many layers `--first-last-layers-bf16` covers (default `1` each). | Increase to 2–4 for FP4 or very deep models. |
| `--fp8-margin M` | Adds `M` bits of headroom to the scaling factor. | Loss spikes despite recipe / first-last mitigation. |
| `--fp8-amax-history-len` / `--fp8-amax-compute-algo` | Controls `delayed` recipe warm-up behavior. | Tune when `delayed` is required (e.g. for CPU offload). |
| `--fp8-wgrad` (set `False`) | Keeps `wgrad` GEMM in higher precision. | Final fallback when MoE / long-context runs show gradient noise. |
| `--tp-only-amax-red` | Restricts amax all-reduce to the TP/CP domain. | Large DP scales where global amax reduction is expensive. |

## Memory Knobs

Once a recipe is stable, the following options reduce memory further:

- **`--fp8-param-gather` / `--fp4-param-gather`** — Keep parameters quantized
  and all-gather them in the quantized dtype. Requires the distributed
  optimizer, Torch FSDP2, Megatron-FSDP, or inference mode.
- **`--reuse-grad-buf-for-mxfp8-param-ag`** — Required companion for
  `mxfp8 + --fp8-param-gather` (see above).
- **`--keep-fp8-transpose-cache`** — Trades memory for compute by avoiding
  recomputation of the FP8 transpose under fine-grained activation
  offloading. See
  [fine_grained_activation_offloading.md](fine_grained_activation_offloading.md).
- **Precision-aware optimizer** — `--use-precision-aware-optimizer` enables
  the TE `FusedAdam` path and lets you store optimizer state in lower
  precision:
  - `--main-grads-dtype {fp32, bf16}`
  - `--exp-avg-dtype {fp32, fp16, bf16, fp8}`
  - `--exp-avg-sq-dtype {fp32, fp16, bf16, fp8}`

## Configuration Cookbook

The following minimal recipes are starting points. All assume a BF16 model
init plus the distributed optimizer.

### Hopper / H100 — fastest stable FP8

```bash
--bf16 \
--fp8-format hybrid \
--fp8-recipe tensorwise \
--fp8-param-gather \
--use-distributed-optimizer
```

Use `delayed` instead of `tensorwise` if you also need optimizer CPU offload.

### Blackwell / B200 — fastest FP8

```bash
--bf16 \
--fp8-format e4m3 \
--fp8-recipe mxfp8 \
--fp8-param-gather \
--reuse-grad-buf-for-mxfp8-param-ag \
--use-distributed-optimizer
```

### Blackwell / B200 — maximum memory / bandwidth savings (FP4)

```bash
--bf16 \
--fp4-format e2m1 \
--fp4-param-gather \
--first-last-layers-bf16 \
--num-layers-at-start-in-bf16 2 \
--num-layers-at-end-in-bf16 2 \
--use-distributed-optimizer
```

### FP8 with optimizer CPU offload

```bash
--bf16 \
--fp8-format hybrid \
--fp8-recipe delayed \
--fp8-param-gather \
--use-distributed-optimizer \
--optimizer-cpu-offload
```

Only `delayed` is supported with CPU offload — Megatron enforces this in
`arguments.py`.

### FP8 with Megatron-FSDP

Use the FP8 recipe of your choice and the FSDP precision policy described in
[megatron_fsdp.md](megatron_fsdp.md#mixed-precision). The relevant FSDP-side
flags are `--megatron-fsdp-main-params-dtype`,
`--megatron-fsdp-main-grads-dtype`, and `--megatron-fsdp-grad-comm-dtype`.

See [examples/megatron_fsdp/train_llama3_8b_fsdp_h100_fp8.sh](../../../examples/megatron_fsdp/train_llama3_8b_fsdp_h100_fp8.sh).

### Minimum-memory optimizer

Layer on top of any recipe above:

```bash
--use-precision-aware-optimizer \
--main-grads-dtype bf16 \
--exp-avg-dtype fp8 \
--exp-avg-sq-dtype fp8
```

## Feature Compatibility Matrix

| Recipe | Distributed optimizer | Megatron-FSDP | Pipeline parallel | CUDA Graph | Optimizer CPU offload | Inference |
| --- | --- | --- | --- | --- | --- | --- |
| `delayed` | ✓ | ✓ | ✓ | ✓ | ✓ (required) | ✓ |
| `tensorwise` | ✓ | ✓ | ✓ | ✓ | ✗ | ✓ |
| `mxfp8` | ✓ (recommended) | ✓ | ✓ | ✓ (`--inference-cuda-graph-scope=block` is mxfp8-only) | ✗ | ✓ |
| `blockwise` | ✓ | ✓ | ✓ | ✓ | ✗ | ✓ |
| `nvfp4` | ✓ | ✓ | ✓ | partial — see roadmap | ✗ | partial |

`--fp8-param-gather` / `--fp4-param-gather` additionally require the
distributed optimizer, Torch FSDP2, Megatron-FSDP, or inference mode.

## End-to-End Examples

| Example | Recipe | Notes |
| --- | --- | --- |
| [examples/llama/train_llama3_8b_h100_fp8.sh](../../../examples/llama/train_llama3_8b_h100_fp8.sh) | `delayed`, hybrid | Reference H100 single-node FP8 run. |
| [examples/megatron_fsdp/train_llama3_8b_fsdp_h100_fp8.sh](../../../examples/megatron_fsdp/train_llama3_8b_fsdp_h100_fp8.sh) | `delayed`, hybrid | FP8 + Megatron-FSDP. |
| [examples/gptoss/02_train.sh](../../../examples/gptoss/02_train.sh) | hybrid + `--fp8-param-gather` | GPT-OSS Megatron Bridge example. |
| [examples/rl/model_configs/nemotron5_56b.sh](../../../examples/rl/model_configs/nemotron5_56b.sh) | `tensorwise`, hybrid | RL workload, large MoE. |
| [examples/rl/model_configs/nemotron5p5_12b_H.sh](../../../examples/rl/model_configs/nemotron5p5_12b_H.sh) | `blockwise`, e4m3 | DeepSeek-style blockwise FP8. |

## Troubleshooting

- **Loss spikes early in training** — Enable `--first-last-layers-bf16`,
  bump `--num-layers-at-{start,end}-in-bf16`, and/or raise `--fp8-margin`.
  Switching from `tensorwise` / `mxfp8` to `delayed` provides a more
  conservative baseline.
- **Slow convergence vs. BF16** — Verify your recipe matches the hardware
  table above; mismatched recipes silently fall back to slower paths.
- **OOM at the optimizer step** — Enable `--use-precision-aware-optimizer`
  and lower `--exp-avg-dtype` / `--exp-avg-sq-dtype`. If still tight, enable
  optimizer CPU offload (requires `--fp8-recipe delayed`).
- **`--fp8-param-gather` rejected at startup** — Add
  `--use-distributed-optimizer` (or run with Torch FSDP2 / Megatron-FSDP).
- **MXFP8 + param-gather warning** — Add
  `--reuse-grad-buf-for-mxfp8-param-ag`; this is the supported MXFP8
  param-gather path.
- **`--fp4-format requires Transformer Engine >= 2.7.0.dev0`** — Upgrade TE or
  fall back to an FP8 recipe.
- **MXFP8 + inference CUDA Graph errors** — `--inference-cuda-graph-scope=block`
  is only supported with `--fp8-recipe mxfp8`; for other recipes use a
  different graph scope.

## See Also

- [Megatron-FSDP](megatron_fsdp.md) — Mixed-precision policy, quantized
  parameter handling.
- [Distributed Optimizer](dist_optimizer.md) — Required for
  `--fp8-param-gather` / `--fp4-param-gather`.
- [Optimizer CPU Offload](optimizer_cpu_offload.md) — Constraints on FP8
  recipe choice.
- [Fine-Grained Activation Offloading](fine_grained_activation_offloading.md)
  — Interaction with `--keep-fp8-transpose-cache`.
- [Transformer Engine documentation](https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/index.html)
  — Authoritative reference for the underlying recipes.
