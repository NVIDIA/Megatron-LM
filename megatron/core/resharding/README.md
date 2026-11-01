# Resharding (Refit)

Transfer model weights between different parallelism configurations
(TP, GTP, PP, EP, DP) with optional format conversion (e.g. BF16 to MXFP8).
Used primarily in RL loops to move weights from a training model to an
inference model that may use a different parallelism layout.

## Architecture

```
refit.py            High-level API: swap_model_weights, caching, MXFP8 auto-detection
    |
planner.py          Local plan builder (every rank all-gathers metadata, replays
                    the same deterministic schedule, keeps only its own ops)
shard_planner.py    Logical-coordinate planner for TP x GTP weight shards
    |
execution.py        Submits send/recv ops to a CopyService, handles writebacks
    |
copy_services/      Pluggable transport backends
    ├── nccl         GPU-to-GPU via torch.distributed P2P
    ├── nccl_m2n     Hierarchical cross-group transfer via NCCL M2N
    ├── gloo         CPU-staged via Gloo process group
    └── nvshmem      NVSHMEM pipelined GPU-to-GPU (requires nvshmem library)

transforms.py       Format conversion hooks (MXFP8ReshardTransform)
utils.py            Data structures (TransferOp, ReshardPlan, ParameterMetadata)
```

## Quick Start

### Basic usage (collocated, same ranks hold both models)

```python
from megatron.core.resharding import swap_model_weights

swap_model_weights(
    src_model=training_model,
    target_model=inference_model,
    # Other collocated backends: "gloo", "nvshmem", or "nixl".
    # "nccl_m2n" is supported only for non-collocated refits.
    refit_method="nccl",
)
```

### With MXFP8 inference model

Call `prepare_swap_model_weights` once during initialization while the
target model's parameters are still in BF16.  This quantizes the target
decoder weights to persistent MXFP8Tensor buffers (whose device pointers
are later captured by CUDA graphs) and caches the transform on the plan.
Subsequent `swap_model_weights` calls pick it up automatically.

```python
from megatron.core.resharding import prepare_swap_model_weights, swap_model_weights

# During init (BF16 params still visible):
prepare_swap_model_weights(src_model=train_model, target_model=infer_model)

# In the RL loop (called repeatedly):
swap_model_weights(train_model, infer_model, refit_method="nccl")
# MXFP8 transform is auto-resolved from the cached plan.
```

### Multiple destination pools

Disaggregated inference can refit the same training model into multiple
disjoint destination shards. Every rank participates in one collective pass
per pool and exposes its target model only during the pass identified by
`dst_pool_index`. Use the same pool arguments for preparation and transfer so
each plan, including any MXFP8 transform, is ready before CUDA graph capture:

```python
prepare_swap_model_weights(
    src_model=train_model,
    target_model=local_inference_model,
    num_dst_pools=2,
    dst_pool_index=local_pool_index,
)
swap_model_weights(
    train_model,
    local_inference_model,
    refit_method="nccl",
    num_dst_pools=2,
    dst_pool_index=local_pool_index,
)
```

### Non-collocated (training and inference on disjoint ranks)

```python
# Source ranks:
swap_model_weights(train_model, None, "nccl",
                   src_rank_offset=0, dst_rank_offset=src_world)

# Destination ranks:
swap_model_weights(None, infer_model, "nccl",
                   src_rank_offset=0, dst_rank_offset=src_world)

# Idle ranks (must still participate in collectives):
swap_model_weights(None, None, "nccl",
                   src_rank_offset=0, dst_rank_offset=src_world)
```

### MIMO training with ordinary LLaVA inference

The public prepare/swap API accepts the original models without caller-side
adapters or new arguments:

```python
from megatron.core.resharding.refit import prepare_swap_model_weights, swap_model_weights

# Each rank supplies its local model, or None on the opposite side.
prepare_swap_model_weights(train_model, inference_model, group=refit_group)
# After an optimizer step, collectively on all refit ranks:
swap_model_weights(train_model, inference_model, "nccl", group=refit_group)
```

Regular refit reads tensor names and one root process-group collection. MIMO
differs in two ways: its components have separate process groups, and their
storage paths differ from LLaVA's. The transfer planner already accepts both
owning groups and separate storage/matching names in `ParameterMetadata`.

Models can declare their local components through an optional `refit_modules()`
method returning `(matching_label, module, process_group_collection)` tuples.
Each module must be the original model or one of its registered descendants;
precision/DDP wrappers are allowed. An empty matching label preserves the
component's tensor names, for example when removing an inference wrapper prefix.
Refit uses its ordinary parameter extractor with each component's groups, then
prefixes the matching name with its label and the storage name with its actual
module path. Rank offsets and metadata serialization remain internal to refit.
The declarations must cover every parameter and persistent buffer exactly once
and produce unique matching names. Models without the method retain the ordinary
extraction path. An empty list describes a rank with no state.

MIMO matches its language model, image encoder, and input projector under
`language_model`, `vision_model`, and `vision_projection`. These match standard
LLaVA's registered names, so inference does not need MIMO. For example, TP2
language training can use two GPUs, TP1 vision/projector training a third,
and standard TP1 LLaVA inference a fourth.

The executor uses original tensor paths, including wrapper levels; declarations
do not rename modules or checkpoint keys. Buffer dtypes are matched by transfer
ID. The existing rank-offset arguments still apply when the refit group joins
independent worlds.

The MIMO mapping covers a core CLIP image encoder and at most one input projector.
Other encoder implementations may require their own matching declarations and
explicit ownership. MIMO configurations enabling FP8, FP4, Kitchen, or per-module
quantization recipes are rejected. Missing ownership, unmapped names, and
ambiguous destination names raise errors. Callers must use equivalent component
architectures and input preprocessing; matching tensor names and shapes alone
do not establish equivalence.

Plans are cached between optimizer steps; update tensor values in place.
Models declaring `refit_modules()` use a memoized fingerprint of the planner's
local metadata, including tensor paths, layouts and ordered group membership.
Rebuilding an equivalent model reuses its plan without retaining model identities.
Fingerprints are memoized only for live models and cleared with the plan cache.
Initialize persistent buffers to their runtime dtypes before the first swap
(for example, MoE routers promote their bias to FP32 on first forward).
A buffer dtype change that preserves the tensor object requires
`clear_plan_cache()` on all refit ranks and a new prepare call before swapping
again.
For a new layout, construct new model objects, call `clear_plan_cache()` on
**all** refit ranks, and prepare again. Replacing tensors or submodules on an
existing model is not automatically tracked by refit's tensor caches.

TP2 affine projectors currently require bias to be disabled: the projector
forward adds its local bias after gathering the output, which is incompatible
with a sharded bias at TP2. This is a model-forward limitation, independent
of refit.

## Copy Service Backends

| Backend | Transport | Best for | Notes |
|---------|-----------|----------|-------|
| `nccl` | GPU P2P via `batch_isend_irecv` | Intra-node / single cluster | Lowest latency for collocated refits; runs the plan in lockstep batches |
| `nccl_m2n` | Native NCCL M2N reshard of whole tensors | Large non-collocated source/destination groups | Requires `nccl-extensions` and NCCL 2.30.5+; disjoint, source-first rank intervals; see restrictions below |
| `gloo` | CPU-staged via Gloo PG | Cross-cluster / multi-node | Higher latency; works where NCCL cross-cluster doesn't |
| `nvshmem` | Pipelined NVSHMEM puts | High-throughput intra-node | Requires NVSHMEM; uses double-buffered kernel pipeline |
| `nixl` | GPU RDMA via NIXL (UCX), sender-initiated WRITE | Cross-cluster / non-collocated | Requires NIXL; transfers GPU memory directly (no host staging); one destination pool only |

`swap_model_weights()` has no default backend; callers pass `refit_method`
explicitly. In the built-in RL loop, `--refit-method` defaults to `gloo`.

Backends that support collocated models detect same-rank (local) transfers via
`task_id` and short-circuit them into direct `tensor.copy_()` instead of going
through the network stack. NCCL M2N is the exception because its source and
destination meshes must be disjoint.

### Bounding transient execution memory

`execution_batch_bytes` (CLI: `--refit-execution-batch-bytes`) is an optional
soft per-rank limit on transient execution staging. A single complete logical
parameter is never split, so one parameter may exceed the limit. Every slice
and replica of a parameter stays in one batch, which lets MXFP8 destinations
assemble the complete BF16 value before quantizing it once.

`nccl`, `gloo`, and `nvshmem` execute the resulting batches. `nixl` keeps one
model-wide submission because its receive address map must remain stable across
refits. `nccl_m2n` uses the rank-coordinated value as its native grouped-
submission limit instead. Ranks agree on the smallest configured non-`None`
value. When every rank uses `None`, NCCL M2N preserves its environment setting
or 256 MiB default.

Independently of the byte limit, `nccl`, `gloo`, and `nvshmem` cap each batch
at 32 logical parameters by default (`MEGATRON_REFIT_MAX_PARAMS_PER_BATCH`;
`0` removes the cap). A byte budget alone can still pack thousands of small
tensors (norm weights, biases, router `expert_bias`) into one
`batch_isend_irecv`; NCCL splits such large groups into kernel plans at
rank-dependent points, and matching sends and receives that land in different
plans deadlock ([pytorch#174288](https://github.com/pytorch/pytorch/issues/174288)).
Every rank derives the batches from the same gathered metadata and environment,
so they agree without another collective. With the cap disabled and no byte
limit, these backends submit the whole model in one batch.

### NCCL M2N backend

Build the current M2N library from
[NVIDIA/nccl-extensions](https://github.com/NVIDIA/nccl-extensions), then install
its Python package together with NCCL4Py. M2N v0.2 requires NCCL 2.30.5 or
newer. For a source checkout, follow the upstream native build instructions,
then install the bindings from outside the `python/` directory:

```bash
CUDA_HOME=/usr/local/cuda pip install -e /path/to/nccl-extensions/python
```

The package imports as `nccl.m2n` and uses `nccl.core` from NCCL4Py. A wheel
may bundle `libnccl_m2n.so`; otherwise set the loader override explicitly:

```bash
export NCCL_M2N_LIBRARY=/path/to/libnccl_m2n.so
```

Select it with `refit_method="nccl_m2n"` or `--refit-method nccl_m2n`.

#### How it executes a plan

The backend does not consume the planner's point-to-point slice operations.
From the same gathered metadata and the same DP-replica source selection, the
planner also builds backend-neutral `TensorReshardSpec`s, one per logical
parameter and destination TP mesh. A spec records the global shape, the local
shapes on the source and destination TP meshes, the shard dimension on each
side (`None` for a replicated tensor), and the two rank meshes. Packed
parameters whose components are TP-sharded independently (for example Mamba
`in_proj` and `conv1d`) produce one spec per component instead of being lowered
into point-to-point slices.

For each spec, the service wraps the local source and destination tensors as
`nccl.m2n.DistTensor`s with a `Shard(dim)` or `Replicate()` placement on a 1-D
mesh and enqueues one `reshard` call. Every distinct (source mesh, destination
mesh) pair, typically one per pipeline-stage pair, gets its own NCCL4Py
communicator, created on first use from a unique ID broadcast over the refit
group with the source ranks first, and its own CUDA stream, so stage pairs run
concurrently. Within a pair, complete parameters are submitted in
`m2n.group()` batches of at most `max_group_bytes` destination bytes to
amortize M2N's fixed per-call work. The limit is the plan's coordinated
`execution_batch_bytes` if set, otherwise `MEGATRON_NCCL_M2N_MAX_GROUP_BYTES`,
otherwise 256 MiB; a single parameter may exceed it.

Destination views that are not contiguous are received into a temporary buffer
and copied back after their group is enqueued. Model parameter storage itself
is not replaced. With an MXFP8 destination, each parameter is received as one
full BF16 buffer and quantized once on the same stream, so the group limit also
bounds the BF16 staging on each destination rank. Only receiver-side MXFP8
conversion is supported.

On first use, the service all-gathers each rank's source/destination role and
group-byte limit, validates the rank topology described below and that every
rank uses the same limit, and caches the result. Each plan's specs are checked
locally against that topology before any weight data moves. If a submission
fails, the service is marked unusable; close it and construct a new one.

#### Restrictions

This backend supports only non-collocated multi-rank layouts. The communication
group must use NCCL for CUDA tensors and must contain a contiguous source
interval starting at group rank 0, immediately followed by a contiguous
destination interval, with no overlapping or idle ranks, and
`num_dst_pools > 1` is not supported. Use a process group scoped to exactly one
source/destination pool when the application has extra ranks. Supported mesh
sizes are validated by `nccl-extensions`.

Plans that cannot be expressed as whole-tensor specs are rejected before
communication. This currently covers:

- GTP-sharded parameters (use the generic `nccl`, `gloo`, `nvshmem`, or `nixl`
  slice-transfer path instead of treating a GTP shard as a complete weight);
- strided TP layouts (`partition_stride != 1`), which includes the `fc1`
  weight of gated linear units such as SwiGLU;
- fused expert tensors that are not exposed as per-expert parameters;
- source and destination dtypes that differ.

The built-in RL loop currently creates its training and inference models on the
same ranks, so it rejects `nccl_m2n`; non-collocated launchers can use the public
API or the ReFIT benchmark.

## How the Reshard Plan Works

1. Each rank extracts parameter metadata (shape, sharding, TP/EP/PP groups).
2. Metadata is all-gathered so **every** rank has the full picture
   (`dist.all_gather_object()`) — no rank-0 bottleneck, no scatter.
3. Every rank independently replays the **same deterministic schedule**
   (`_iter_global_transfer_ops`):
   - Iterate destination ranks, then each rank's destination params in gathered
     order; for each destination param, find the matching source param(s) by name.
   - Preserve the established LCM/block-interleaved planner for non-GTP
     parameters. For a GTP parameter, map every local TP x GTP shard into
     logical global weight coordinates and intersect it with the destination
     shard. This excludes GTP alignment padding while composing with column,
     row, strided, and packed TP layouts.
   - Assign a monotonic `task_id` per sub-op.  Because the iteration order and
     counter are a pure function of the gathered metadata, the send op computed
     on the sender and the recv op computed on the receiver get the **same**
     `task_id` without any central authority.
4. Each rank keeps only the ops where it is the sender or receiver.
5. The plan is cached so repeated refits skip steps 1-4.

The deterministic schedule stays stable when a larger roster is supplied: existing
transfers keep their `task_id`s and newly appended destination ranks receive new
ones. Live process-group membership changes and their orchestration remain future
work; this module does not currently add or remove ranks from a running group.

## MXFP8 Transform

When the target model uses `transformer_impl='inference_optimized'` with
FP8 enabled and `fp8_recipe='mxfp8'`, an `MXFP8ReshardTransform` is
automatically created and attached to the cached plan.

The transform handles two scale layouts:

- **2D scale** (`scale.ndim == 2`): Each row of scales maps to one row of
  data.  Slices are independent, so received BF16 data is converted to
  MXFP8 per-slice immediately.
- **1D scale** (`scale.ndim == 1`): FlashInfer swizzled layout that encodes
  scales across the full weight tensor.  Partial updates would corrupt the
  layout, so all BF16 slices are accumulated into a single buffer and
  quantized once all slices arrive.

The transform writes directly into persistent MXFP8Tensor buffers
(via `.copy_()`) so that CUDA-graph device-pointer captures remain valid
across refits.

## Caching

| Cache | Key | Contents | Why |
|-------|-----|----------|-----|
| `_service_cache` | Backend name + process-group identity + M2N execution limit | `CopyService` instance | Avoid re-creating backend communicators and buffers |
| `_plan_cache` | Rank, source/destination config, offsets, world size, expert count, pool, execution limit | `ReshardPlan` + attached transform | Avoid collective plan rebuild on repeated refits; ordinary configs include dense/expert GTP-remat sizes; models declaring `refit_modules()` use a metadata fingerprint |
| `_model_fingerprints` | Weak model key | Metadata fingerprint | Avoid re-extracting composite metadata on warm swaps; entries expire with models and are cleared by `clear_plan_cache()` |

Call `clear_all_caches()` before destroying distributed process groups
to avoid stale references.  This also finalizes NVSHMEM resources.

## Process Group Requirements

The source and destination models must each have a `pg_collection`
attribute with the following groups:

| Field | Required | Purpose |
|-------|----------|---------|
| `tp` | Yes | Tensor parallelism sharding |
| `dp` | Yes | Data parallelism routing |
| `pp` | If PP > 1 | Pipeline stage / layer index remapping |
| `ep` | If MoE | Expert parallelism routing |
| `expt_tp` | If expert TP | Expert-specific tensor parallelism |
| `gtp_remat` | If dense GTP | Dense weight-rematerialization shards |
| `expt_gtp_remat` | If expert GTP | Expert weight-rematerialization shards |

## File Reference

| File | Role |
|------|------|
| `refit.py` | Public API, caching, MXFP8 auto-detection |
| `planner.py` | Local deterministic plan builder (metadata, LCM/block-interleaved planners) |
| `shard_planner.py` | Logical-coordinate TP x GTP shard planner |
| `execution.py` | Plan executor (send/recv submission, writeback, format conversion) |
| `transforms.py` | `ReshardTransform` base class, `MXFP8ReshardTransform` |
| `utils.py` | `TransferOp`, `ReshardPlan`, `ParameterMetadata`, `ShardingDescriptor` |
| `copy_services/nccl_copy_service.py` | NCCL backend |
| `copy_services/nccl_m2n_copy_service.py` | Hierarchical NCCL M2N backend |
| `copy_services/gloo_copy_service.py` | Gloo backend |
| `copy_services/nixl_copy_service.py` | NIXL/UCX backend |
| `copy_services/nvshmem_copy_service.py` | NVSHMEM backend (delegates to `nvshmem_copy_service/`) |
| `nvshmem_copy_service/` | Full NVSHMEM implementation (planning, memory, kernels, pipeline) |
