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

`local` is not currently supported with Megatron-FSDP. Its backward graph can replay fused
weight-gradient writes before FSDP allocates, throttles, and validates the live `main_grad` bucket;
its eager allocators can also move graph-baked parameter addresses. This configuration fails
explicitly even when double buffering or NCCL user buffers are enabled. Use
`--cuda-graph-impl transformer_engine` for per-layer Megatron-FSDP graphs, or disable CUDA graphs.

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
Programmatic Megatron-FSDP callers must also keep the DDP allocation policy consistent with the
selected graph backend: TE capture requires both `megatron_fsdp_cuda_graph_mode=True` and
`megatron_fsdp_use_planned_allocator=True`; full-iteration capture requires the graph-mode flag but
does not use the planned allocator. The Megatron-FSDP adapter rejects inconsistent combinations.

### Usage

```bash
--cuda-graph-impl transformer_engine \
--cuda-graph-modules attn moe_router moe_preprocess
```

For the TE backend, `attn` includes every attention implementation that declares a compatible
graph input contract. This includes standard `SelfAttention` and Gated DeltaNet for fixed-shape
SBHD inputs. Packed-THD Gated DeltaNet remains eager because its forward path currently resolves
sequence metadata with host synchronization.

The same training `--cuda-graph-modules` options apply as for `local`, and the default is likewise
whole-layer training capture when the flag is omitted.

### Megatron-FSDP buffer planning

Transformer Engine graphs record absolute addresses for unsharded parameter buffers and fused
weight-gradient (`main_grad`) buffers. With Megatron-FSDP, the helper observes bucket lifetimes
during eager warmup and freezes graph-covered buckets onto pre-materialized, address-stable slots
before capture. Planned allocation is the only supported temporary-buffer policy for this TE and
Megatron-FSDP path: `--fsdp-double-buffer` is rejected. Buckets outside the frozen graph plan
continue to use their existing eager allocators. Every parameter-and-gradient buffer receives a
namespace composed of its human-readable model-chunk label and a process-local monotonic instance
ID. This keeps both virtual-pipeline chunks and separately wrapped model groups disjoint.

This path requires a Transformer Engine build that implements MCore's versioned
`capture_time_hooks` contract for all four forward/backward pre/post hook phases. Compatible builds
advertise `mcore_fsdp_capture_time_hooks_v1` through the `__mcore_cuda_graph_protocols__` marker on
`make_graphed_callables()` or its module. The repository validation loader publishes this marker
only after verifying TE commit `4251467130ce88595f584a5160e6350176333923` and the pinned
`graph.py` SHA-256 `71188f41bd37611075f520222f9408249372c2e8a2356f68760dd36429a5cfe9`.
An argument named `capture_time_hooks` alone does not establish the required semantics. Models
that pass runtime tensors through keyword arguments need TE 1.10 or newer for the public
`sample_kwargs` API and the pinned overlay's mixed positional/keyword-input fix. In particular,
model-computed rotary inputs must be observed during eager warmup and declared as graph inputs.

After capture succeeds, the planned-FSDP path freezes a local PP/VPP topology and schedule
signature. Every training and evaluation iteration must retain the captured topology, graph scope,
microbatch size, and number of microbatches; evaluation therefore uses the same microbatch size and
microbatch count as capture. A predictable mismatch is a fatal error and requires restarting the
job; automatic reset or retrace is not supported. Allocator occupancy, capacity, dtype, and pointer
checks remain authoritative for runtime changes that the boundary signature cannot predict.

The core planned-FSDP path rejects `--cuda-graph-dynamic-microbatches`, variable sequence
lengths, sequence-packing schedulers, and RL sequence packing during configuration. Supporting
those dynamic schedules requires the later all-rank retrace/recapture lifecycle; they must remain
eager in this PR.

The number of planned slots is not fixed at two. Warmup records each bucket's exact padded size,
dtype, and allocate/free lifetime. At freeze, non-overlapping graph-covered lifetimes are colored,
and each used `(color, bucket-offset)` pair is materialized at the maximum observed size of the
buckets assigned to it. A workload may therefore need one, two, or more colors according to the
conservative conflict graph inferred from observed lifetimes; greedy coloring can exceed both the
instantaneous peak-live count and the minimum coloring. These lifetime-planned slots are not FSDP
double buffers.

NCCL user buffers are a separate unsupported combination. Planned slots retain the FSDP
memory-allocation context, but the current manual NCCL registration can run before those slots are
materialized. Consequently, TE planned allocation rejects `--use-nccl-ub` until post-freeze
registration is implemented and GPU-validated. NCCL user buffers also imply FSDP double buffering,
which this path independently rejects.

The `optim_grads` sharding strategy is also not yet supported with TE per-layer graphs because it
has no per-layer FSDP pre-backward hook to claim a fused `main_grad` slot before replay.
`optim_grads_params` supplies that hook and is the supported gradient-sharding strategy;
`no_shard` and `optim` use persistent main gradients and do not need the claim. Every graph callable
with distributed weights or gradients must also be an FSDP unit module.

The allocator rejects an overlapping slot user, a request larger than the frozen capacity, a dtype
change, or a storage-address change. These are hard errors because falling back to another buffer
would let a graph silently read or write stale storage. Buckets outside the TE graph continue to use
the configured eager allocator. A frozen plan cannot later be extended with newly discovered graph
buckets. Fused `main_grad` slots are claimed and their pointers revalidated after parameter
all-gather and before TE backward graph replay. The claim verifies that the frozen slot is
materialized, unoccupied or already owned by the same bucket, and unchanged in dtype, capacity, and
device pointer. It then records occupancy so relocation or a live colormate conflict fails before
the graph can write to a capture-era address.

TE capture is one-shot. If capture fails, the stock training integration restores temporarily
disabled DDP forward-pre hooks in a `finally` block. The helper restores callable hooks and
Megatron-FSDP parameter exposure, and clears only the global capture flag and GC freeze state that
it owns, before re-raising the original failure. The same helper rejects a retry after failure.

This cleanup is intentionally not a distributed transaction. It does not reset partially
constructed graphs, provide cross-rank failure consensus, or promise exhaustive rollback of
model/optimizer, activation-offload, and RNG side effects outside the state listed above. Treat a
capture failure as fatal and restart the job after correcting the underlying problem.

### Model-computed rotary inputs

Models such as Qwen3.5-VL compute language MRoPE above the Transformer layer and pass it as
per-microbatch keyword arguments. TE capture declares the rotary tensors observed during eager
warmup as graph inputs so replay copies each microbatch's values into independent static buffers.
Observation is restricted to known rotary arguments.

If captured attention requires model-computed rotary inputs, at least one eager warmup forward is
required. Capture fails rather than producing a graph without positional embeddings when no rotary
inputs were observed or when the installed TE version cannot accept keyword graph inputs. This
applies to `position_embedding_type="mrope"` language layers. Rotary inputs are not declared when
attention remains eager under a partial capture scope.

The stock training loop creates `TECudaGraphHelper` for the language decoder only. Qwen3.5-VL
vision layers, including their per-microbatch 2-D RoPE path, remain eager. Custom multimodal loops
may explicitly construct `VisionTECudaGraphHelper`; that opt-in path is not part of the stock
training integration and requires at least one eager vision forward before capture. The helper then
declares the observed 2-D RoPE inputs or fails loudly if they are unavailable. Every rank that
participates in the helper's default-group capture barrier must invoke it in the same order,
including ranks with no vision layers; those ranks take the no-graph path but still complete the
capture phase. There is no cross-rank capture-failure consensus in this core path.
Dynamic-microbatch capture is not supported by this custom Vision helper. Without Megatron-FSDP, a
custom loop may use both helpers; each helper consumes and cleans only the decoder subtree it owns,
so one capture does not erase the other's observation. Multiple
helpers cannot currently freeze disjoint bucket subsets of the same Megatron-FSDP
parameter-and-gradient buffer: the first frozen plan is immutable and a second helper fails if it
discovers new buckets. Keep vision eager in that configuration, as the stock training loop does.

For hybrid mHC wrappers, capture discovery inspects the effective inner Transformer layer. An inner
attention implementation that opts out of TE graphs remains eager, and a partial wrapper split that
is not implemented is skipped instead of capturing a different region than requested. When the
model uses Megatron-FSDP, every `HyperConnectionHybridLayer` is deliberately kept eager: FSDP's
parameter gather/release and pre-backward hooks currently belong to the inner layer and cannot be
safely driven by a graph whose callable boundary is the outer wrapper.

One temporary environment control covers behavior that does not yet have a public config field:

- `MEGATRON_CG_SKIP_BUFFER_ADDRESS_CHECK=1` disables the module-level replay pointer check. Only
  the exact value `1` is accepted. This is a dangerous debugging escape hatch: replay may silently
  access stale storage if an address moves. Planned fused-`main_grad` claims still perform their
  allocator-level checks.

This control is internal and may be replaced by an explicit configuration field. Do not set it in
a production recipe without validating the resulting memory use and numerical behavior.

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

With [paged stash](paged_stash.md), expert dispatch shapes become static (pre-sized via
`--moe-expert-rank-capacity-factor`), which allows full-iteration CUDA graphs to be used on MoE
models as well:

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
  suboptimal graphs. It is invalid for TE attention capture with model-computed runtime rotary
  inputs such as language MRoPE because they must be observed before capture. The same restriction
  applies when a custom loop opts into `VisionTECudaGraphHelper` for vision 2-D RoPE.
- `--cuda-graph-dynamic-microbatches` assumes every rank in a pipeline group enters the same
  dynamic-slot discovery collective and that the capture topology stays fixed. Mixing ranks with
  graphable callables and empty ranks is unsupported. The planned-FSDP iteration-boundary signature
  detects later local schedule changes, but it does not provide pre-capture, cross-rank consensus
  for custom or empty-stage layouts. Those layouts must disable dynamic microbatch slots or remain
  eager. The custom Vision helper does not support dynamic-microbatch capture.
- Selective activation recomputation must not overlap a captured region when bitwise agreement with
  eager execution is required. Argument validation reports overlapping modules; in particular,
  `moe_router` capture overlaps full-`moe` recompute and any captured `shared_experts`, while
  whole-layer, `mlp`, and `moe` capture cover their respective full MLP regions.
  GDN attention capture also overlaps whole-`gdn` recompute.
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
