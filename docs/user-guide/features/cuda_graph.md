<!---
   Copyright (c) 2022-2026, NVIDIA CORPORATION. All rights reserved.
   NVIDIA CORPORATION and its licensors retain all intellectual property
   and proprietary rights in and to this software, related documentation
   and any modifications thereto. Any use, reproduction, disclosure or
   distribution of this software or related documentation without an express
   license agreement from NVIDIA CORPORATION is strictly prohibited.
-->

# CUDA Graph

CUDA Graphs reduce kernel-launch overhead by recording GPU operations once and replaying the recording on subsequent iterations. Megatron-LM provides three CUDA graph implementations controlled by `--cuda-graph-impl`.

For implementation background and design details, see NVIDIA's
[Transformer Engine and Megatron-LM CUDA Graph Support](https://docs.nvidia.com/dl-cuda-graph/torch-cuda-graph/te-megatron-cuda-graphs.html).
That article is a useful conceptual reference, but some examples there still use older flags such as
`--enable-cuda-graph` or `--cuda-graph-scope full_iteration`; in this repository, prefer
`--cuda-graph-impl local|transformer_engine|full_iteration` as documented below.

## Overview

CUDA graph behavior is set by three orthogonal flags:

| Flag | Values | Purpose |
|---|---|---|
| `--cuda-graph-impl` | `none` / `local` / `transformer_engine` / `full_iteration` | Which capture backend or strategy to use |
| `--cuda-graph-modules` | `attn` / `mlp` / `moe` / `moe_router` / `moe_preprocess` / `mamba` | Per-layer **training** capture coverage; multi-valued and only meaningful for `local` and `transformer_engine` |
| `--inference-cuda-graph-scope` | `none` / `layer` / `block` | Granularity of CUDA graphs during **inference**; only `local` supports non-`none` values |

Supported combinations:

| `--cuda-graph-impl` | Backend | Training capture | Inference capture |
|---|---|---|---|
| `none` | — | off | off |
| `local` | MCore `CudaGraphManager` | per-layer, controlled by `--cuda-graph-modules` | `layer` (default) or `block`, controlled by `--inference-cuda-graph-scope` |
| `transformer_engine` | TE `make_graphed_callables()` | per-layer, controlled by `--cuda-graph-modules` | not supported (`none` only) |
| `full_iteration` | MCore `FullCudaGraphWrapper` | one graph per training iteration; `--cuda-graph-modules` must be empty | not supported (`none` only) |

---

## CUDA Graph — Local Implementation (`--cuda-graph-impl local`)

Uses MCore's built-in `CudaGraphManager`. During training, this is a per-layer mode:
leaving `--cuda-graph-modules` unset captures the whole Transformer layer, while specifying
modules restricts capture to selected sub-regions. During inference, `local` can instead attach
graphs at either the layer boundary or the enclosing block boundary, as controlled by
`--inference-cuda-graph-scope`.

Operationally, this path is tightly integrated into MCore training and inference:

- graphable modules create and own their `CudaGraphManager` instances automatically
- the existing training schedules drive warmup/capture/replay automatically
- users select the mode through config flags only; there is no separate helper API to
  wire into a custom training loop or a separate need to handle static input buffers

### Usage

```bash
--cuda-graph-impl local
```

### `--cuda-graph-modules` options

| Module | What is captured |
|---|---|
| *(empty / not set)* | Entire Transformer layer (default) |
| `attn` | `TransformerLayer._forward_attention()` |
| `mlp` | `TransformerLayer._forward_mlp()` for dense layers |
| `moe` | `TransformerLayer._forward_mlp()` for MoE layers (drop-and-pad only) |
| `moe_router` | MoE router + shared experts (if not EP-comm-overlapped) |
| `moe_preprocess` | `MoELayer.preprocess()` — must be paired with `moe_router` |
| `mamba` | Mamba SSM layer |

**Example — MoE model, capture attention and router:**
```bash
--cuda-graph-impl local \
# Optionally restrict captured modules (default: capture whole layer, but not working for MoE dynamic shapes)
--cuda-graph-modules attn moe_router moe_preprocess
```

---

## CUDA Graph — Transformer Engine Implementation (`--cuda-graph-impl transformer_engine`)

Uses Transformer Engine's `make_graphed_callables()` path. In Megatron-LM's CLI, this has the
same training granularity as `local`: leaving `--cuda-graph-modules` unset captures the whole
Transformer layer, while specifying modules restricts capture to selected sub-regions. The main difference from
`local` is the backend implementation and feature compatibility. Unlike `local`, this path does
not support inference CUDA graphs.

Compared to `local`, this path exposes a more general and self-contained API via TE's
`make_graphed_callables()`, giving users greater flexibility and control over how CUDA graphs are
wired into custom training loops. The trade-off is that it requires more manual setup:

- the training loop must instantiate `TECudaGraphHelper`
- the training loop must call helper methods such as `create_cudagraphs()` and
  `cuda_graph_set_manual_hooks()` at the correct points

Megatron-LM's stock training loop already wires these calls in `megatron/training/training.py`,
but custom training scripts must do the same work themselves.

### Usage

```bash
--cuda-graph-impl transformer_engine \
--cuda-graph-modules attn moe_router moe_preprocess
```

The same training `--cuda-graph-modules` options apply as for `local`, and the default is likewise
whole-layer training capture when the flag is omitted.

---

## Full-Iteration Training CUDA Graph (`--cuda-graph-impl full_iteration`)

Captures the entire training iteration (excluding optimizer) as a single CUDA graph. The same
wrapper is also used for training-loop validation/eval in forward-only mode. This provides the
largest training/validation latency reduction.

This implementation does not create inference CUDA graphs. For inference, use
`--cuda-graph-impl local --inference-cuda-graph-scope layer|block`.

### Requirements

- `--no-check-for-nan-in-loss-and-grad` is required: NaN checks involve CPU-GPU synchronization
  which cannot run inside a CUDA graph.
- `--cuda-graph-modules` must be omitted (or left empty): per-module selection has no meaning
  when the entire iteration is captured as a single graph.

### Example

```bash
--cuda-graph-impl full_iteration \
--no-check-for-nan-in-loss-and-grad
```

---

## Common Configuration Examples

### mHC training

`--enable-mhc-connections` supports Transformer Engine training graphs and
full-iteration graphs with fixed sequence lengths and microbatch shapes. The
existing mHC parameter names, checkpoint keys, fused-kernel selection, and
residual-stream contraction are shared with eager training.

For `HybridModel`, the graph boundary is the mHC wrapper. Capture includes its
stream aggregation and mixing, and static inputs have width
`hidden_size * mhc_num_residual_streams`. Hybrid MTP stacks participate in TE
capture through their individual wrapped layers. A custom GPT layer specification
using `HyperConnectionTransformerLayer` follows the same TE protocol.

| Training path | mHC capture coverage |
|---|---|
| TE, dense Hybrid layers | Attention, dense MLP, GDN, and Mamba, selected by the applicable module scopes |
| TE, MoE | Router and optional preprocessing; expert dispatch and computation continue eagerly |
| Full iteration | Forward, backward, and gradient synchronization; optimizer steps remain outside the graph |

For a Hybrid attention/MoE model, including a Hybrid MTP pattern, use:

```bash
--enable-mhc-connections \
--cuda-graph-impl transformer_engine \
--cuda-graph-modules attn moe_router moe_preprocess
```

The partial MoE graph returns the mHC residual and mixing tensors alongside the
router tensors. This preserves the backward path from the eager expert output
through the captured mHC prefix. Whole-layer and whole-MoE TE capture are not
supported for mHC MoE models. Hybrid partial MoE capture requires separate attention and
MoE layers, as provided by the standard Hybrid layer specs. TE graphs require independent
MTP depths (`mtp_use_repeated_layer=False`); this restriction is specific to TE graphs.
TE mHC capture/replay also requires `padding_mask=None`, since the per-layer static
samples do not transport the MoE token padding mask.

Full-iteration MoE capture requires fixed-capacity, padded expert inputs. The
eager reference must use the same capacity and padding settings, since capacity
changes can change token routing:

```bash
--enable-mhc-connections \
--cuda-graph-impl full_iteration \
--no-check-for-nan-in-loss-and-grad \
--moe-token-dispatcher-type alltoall \
--moe-expert-capacity-factor 1.0 \
--moe-pad-expert-input-to-capacity
```

Selective mHC recomputation, activation offloading, and expert-parallel
communication overlap are rejected for these mHC graph modes. Packed THD input,
dynamic context parallelism, and inference capture are outside this training
support. The existing pipeline-parallel and MTP placement requirements still
apply. Nonzero dropout requires the graph-safe RNG implementation used by the
selected backend; it is not subject to a blanket mHC dropout restriction.

When combined with PP/VPP support, full-iteration dropout can expose a native
allocator capture error in NGC PyTorch 26.08 (`4fdf77b940`). Lazy RNG state
initialization can query device-wide allocator events while another stream is
capturing. This configuration requires a PyTorch allocator capture-safety fix;
the MCore graph support does not repair that dependency error.

The full-iteration loader returns a fresh batch dictionary for each consumer
while retaining the captured tensor storage. This allows pipeline stages to
discard unused batch fields without corrupting subsequent refills. Batched P2P
communication retains its eager synchronization fence and omits that device-wide
fence only while the CUDA stream is being captured.

The mHC functional regression recipes enable `--deterministic-mode` and set
`NVTE_ALLOW_NONDETERMINISTIC_ALGO=0` to request exact golden verification, including
the number of zero gradients. A passing repeat is still required to establish
reproducibility. GDN uses its existing deterministic implementation under this
setting and allocates its initial recurrent state directly on the input device
so capture does not perform a CPU-to-CUDA copy. Fused mHC remains enabled in the
recipes.

### Dense Model Training

All three implementations work for dense models:

```bash
# Per-layer (local)
--cuda-graph-impl local
# equivalent: --cuda-graph-impl local --cuda-graph-modules attn mlp

# Per-layer (TE)
--cuda-graph-impl transformer_engine
# equivalent: --cuda-graph-impl transformer_engine --cuda-graph-modules attn mlp

# Full-iteration
--cuda-graph-impl full_iteration \
--no-check-for-nan-in-loss-and-grad
```

### MoE Model Training

MoE expert dispatch involves dynamic shapes and cannot be captured. `--cuda-graph-modules` is used
to capture only the static parts (attention, router, preprocess) while leaving expert compute in
eager mode. Example using `transformer_engine` (`local` works the same way):

```bash
--cuda-graph-impl transformer_engine \
--cuda-graph-modules attn moe_router moe_preprocess
```

With paged stash (currently available only on `dev`; see
`docs/user-guide/features/paged_stash.md` on the `dev` branch), expert dispatch shapes become
static (pre-sized via `--moe-expert-rank-capacity-factor`), which allows full-iteration CUDA
graphs to be used on MoE models as well:

```bash
--cuda-graph-impl full_iteration \
--no-check-for-nan-in-loss-and-grad \
--moe-flex-dispatcher-backend hybridep \
--use-transformer-engine-op-fuser \
--moe-expert-rank-capacity-factor <float> \
--moe-paged-stash
```

---

## Additional Notes

- `--cuda-graph-warmup-steps` (default: 3) controls how many warmup steps run before CUDA graph
  capture. Setting it to 0 is not recommended: some operations rely on the first few iterations
  for lazy initialization or autotuning, and capturing too early may produce incorrect or
  suboptimal graphs.
- Inference CUDA graphs (serving or RL rollout) currently require
  `--cuda-graph-impl local`. Use `--inference-cuda-graph-scope layer|block` with
  `local`; all other implementations must set `--inference-cuda-graph-scope none`,
  meaning inference runs in eager mode.
- Background reference: [Transformer Engine and Megatron-LM CUDA Graph Support](https://docs.nvidia.com/dl-cuda-graph/torch-cuda-graph/te-megatron-cuda-graphs.html),
  which also covers PyTorch CUDA Graph best practices and lessons learned.

---

## Migration Guide

Legacy configurations (including `--enable-cuda-graph`, `--external-cuda-graph`, the renamed
`--cuda-graph-scope` flag (now `--cuda-graph-modules`), and deprecated module values such as
`full_iteration` and `full_iteration_inference`) are still accepted and automatically migrated
at runtime, but we encourage updating your configs to the new forms:

| Old command | New command |
|---|---|
| `--enable-cuda-graph` | `--cuda-graph-impl local` |
| `--external-cuda-graph` | `--cuda-graph-impl transformer_engine` |
| `--cuda-graph-scope <modules>` | `--cuda-graph-modules <modules>` |
| `--cuda-graph-impl local --cuda-graph-scope full_iteration` | `--cuda-graph-impl full_iteration` |
| `--cuda-graph-impl local --cuda-graph-scope full_iteration_inference` | `--cuda-graph-impl local --inference-cuda-graph-scope block` |
| `--cuda-graph-impl local --cuda-graph-scope attn moe_router moe_preprocess full_iteration_inference` | `--cuda-graph-impl local --cuda-graph-modules attn moe_router moe_preprocess --inference-cuda-graph-scope block` |
