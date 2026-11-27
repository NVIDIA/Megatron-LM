{/*
   Copyright (c) 2022-2026, NVIDIA CORPORATION. All rights reserved.
   NVIDIA CORPORATION and its licensors retain all intellectual property
   and proprietary rights in and to this software, related documentation
   and any modifications thereto. Any use, reproduction, disclosure or
   distribution of this software or related documentation without an express
   license agreement from NVIDIA CORPORATION is strictly prohibited.
*/}

# CUDA Graph

CUDA Graphs reduce kernel-launch overhead by recording GPU operations once and replaying the recording on subsequent iterations. Megatron-LM provides three CUDA graph implementations controlled by `--cuda-graph-impl`.

For implementation background and design details, see NVIDIA's
[Transformer Engine and Megatron-LM CUDA Graph Support](https://docs.nvidia.com/dl-cuda-graph/latest/torch-cuda-graph/te-megatron-cuda-graphs.html).
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

For scheduler-produced THD batches, use `--use-varlen-dataset`,
`--sequence-packing-scheduler dp_balanced`, `--cuda-graph-modules attn`,
`--pad-packed-seq-alignment max`, and an explicit `--cuda-graph-thd-max-packed-sequences`. The metadata capacity includes one
reserved dummy document for token padding. `--max-seqlen-per-dp-cp-rank` sets the
fixed local token capacity; the source global batch size and micro-batch size 1
must stay fixed. PP/VPP automatically keeps graph inputs alive until their
corresponding backward even when packing reduces the number of microbatches.
Source-batch ramp-up is unsupported: `--step-batch-size-schedule`
is rejected for this path, and the deprecated `--rampup-batch-size` is ignored by
the microbatch calculator. Increasing the source batch would require recapturing
the graphs with a new capacity. Dynamic CP, EP communication overlap, delayed
wgrad, and paged stash are outside this path. mHC wrappers and attention graph
splitting require their separate graph integrations; these packing options do
not enable mHC capture by themselves.

### Usage

```bash
--cuda-graph-impl transformer_engine \
--cuda-graph-modules attn moe_router moe_preprocess
```

The same training `--cuda-graph-modules` options apply as for `local`, and the default is likewise
whole-layer training capture when the flag is omitted.

### Hybrid models: one graph per mixer and feed-forward pair

`HybridModel` runs each block as two layers: a mixer (attention, GDN, and similar) without a
feed-forward branch, followed by an MoE or MLP layer without an attention branch. With per-layer
graphs, every block therefore launches two forward and two backward graphs.
`--cuda-graph-pair-hybrid-layers` captures each mixer together with the MoE or MLP layer that
follows it as one graph, which halves the graph launches per block:

```bash
--cuda-graph-impl transformer_engine \
--cuda-graph-modules attn moe_router moe_preprocess \
--cuda-graph-pair-hybrid-layers
```

- The pair's graph runs the same operations as the two per-layer graphs. The mixer launches it,
  and the MoE or MLP layer then runs only its work outside the graph, such as the MoE token
  dispatch, experts and combine. With `--fp32-residual-connection`, the pair rounds the mixer's
  FP32 output to the BF16 graph input of the MoE or MLP layer, as the per-layer graphs do.
- The pair is not part of the model, so parameter names and checkpoints do not change.
- A layer keeps its own graph when it cannot be paired: Mamba layers, layers whose neighbor is not
  graphed, pairs split by a pipeline stage boundary, hash-routed MoE layers, and pairs whose two
  layers differ in their FP8/FP4 first/last-layer BF16 setting.
- An MoE layer whose graph contains the MoE megakernel (`--moe-megakernel-backend`) keeps its own
  graph, with a warning: the megakernel's weight views are refreshed eagerly right before that
  graph replays.
- A layer whose own config in a `hybrid_layer_config_list` sets a `cuda_graph_impl` other than
  `transformer_engine` is not paired and runs as it does with per-layer graphs; its neighbor keeps
  its own graph.
- The layers of Hybrid MTP blocks, which per-layer graphs leave eager, are graphed too, paired the
  same way. They stay eager, with a warning:
  - with `--mtp-use-repeated-layer` and several MTP depths;
  - with `--keep-mtp-in-bf16` in FP8 or FP4 training;
  - when the MTP layer configs of a `hybrid_layer_config_list` select another FP8 or FP4 recipe
    than the model (format, recipe, recipe options or amax reduction group), since one capture
    uses the model's recipe for all layers;
  - with `--fp32-residual-connection`, because the BF16 graph inputs would round the FP32
    residual stream that the eager MTP layers keep.
- The MoE layers of Hybrid MTP blocks also stay eager, with a warning, and the mixers before them
  keep their own graphs, when an MoE layer's own config sets up a graph scope that pairing does
  not capture:
  - `--mlp-chunks-for-training` greater than 1 with `moe_router`;
  - `moe` or whole-layer capture with `--moe-token-dispatcher-type allgather`;
  - `moe` or whole-layer capture with neither drop-and-pad MoE (`--moe-expert-capacity-factor`
    with `--moe-pad-expert-input-to-capacity`) nor `--moe-virtual-expert-load-balance`, which the
    `moe` scope requires for decoder layers too.
- Pairing is skipped, with a warning, with `--overlap-moe-expert-parallel-comm`,
  `--delay-wgrad-compute`, `--fine-grained-activation-offloading`, `--sequence-packing-scheduler`,
  `--enable-mhc-connections` or `--context-parallel-size` greater than 1.
- Wide residual streams (`--wide-residual`) do not support Transformer Engine CUDA graphs:
  configuration validation rejects that combination, with or without pairing.
- The flag has no effect with other `--cuda-graph-impl` values.

---

## Full-Iteration Training CUDA Graph (`--cuda-graph-impl full_iteration`)

Captures the entire training iteration (excluding optimizer) as a single CUDA graph. The same
wrapper is also used for training-loop validation/eval in forward-only mode. This provides the
largest training/validation latency reduction.

With an eager optimizer, full-iteration training captures model-to-optimizer gradient conversion
in the forward/backward graph's memory pool. The distributed optimizer and the layer-wise
mixed-precision optimizer can reuse completed forward/backward scratch for FP32 gradients.
Replay order must remain forward/backward, gradient conversion, then optimizer update.
The converted values are valid until the next forward/backward replay; gradient addresses must
remain stable. Resetting the training graph also invalidates the conversion graph.

This conversion graph is disabled with `--optimizer-cuda-graph`, which captures the optimizer
step itself. CPU-offloaded and Megatron-FSDP optimizer paths are unchanged, as is the distributed
precision-aware optimizer path that already aliases model gradients.

This implementation does not create inference CUDA graphs. For inference, use
`--cuda-graph-impl local --inference-cuda-graph-scope layer|block`.

### Requirements

- Loss and gradient checks remain enabled. Values passed to the rerun state machine during
  capture are snapshotted at their validation sites. Host-side checks run after each replay,
  before the training loop decides whether to rerun or update the optimizer.
- `--cuda-graph-modules` must be omitted (or left empty): per-module selection has no meaning
  when the entire iteration is captured as a single graph.

### Example

```bash
--cuda-graph-impl full_iteration
```

---

## Common Configuration Examples

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
--cuda-graph-impl full_iteration
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
- Background reference: [Transformer Engine and Megatron-LM CUDA Graph Support](https://docs.nvidia.com/dl-cuda-graph/latest/torch-cuda-graph/te-megatron-cuda-graphs.html),
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
