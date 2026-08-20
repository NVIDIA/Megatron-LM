# Resharding (Refit)

Transfer model weights between different parallelism configurations
(TP, PP, EP, DP) with optional format conversion (e.g. BF16 to MXFP8).
Used primarily in RL loops to move weights from a training model to an
inference model that may use a different parallelism layout.

## Architecture

```
refit.py            High-level API: swap_model_weights, caching, MXFP8 auto-detection
    |
planner.py          Local plan builder (every rank all-gathers metadata, replays
                    the same deterministic schedule, keeps only its own ops)
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
    refit_method="nccl",  # or "nccl_m2n", "gloo", "nvshmem", or "nixl"
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

## Copy Service Backends

| Backend | Transport | Best for | Notes |
|---------|-----------|----------|-------|
| `nccl` | GPU P2P via `batch_isend_irecv` | Intra-node / single cluster | Lowest latency; default choice |
| `nccl_m2n` | NCCL M2N copy/staging reshard | Large non-collocated source/destination groups | Requires the official `nccl-extensions` Python package and NCCL 2.30.5+; source ranks must precede destination ranks |
| `gloo` | CPU-staged via Gloo PG | Cross-cluster / multi-node | Higher latency; works where NCCL cross-cluster doesn't |
| `nvshmem` | Pipelined NVSHMEM puts | High-throughput intra-node | Requires NVSHMEM; uses double-buffered kernel pipeline |
| `nixl` | GPU RDMA via NIXL (UCX), sender-initiated WRITE | Cross-cluster / non-collocated | Requires NIXL; transfers GPU memory directly (no host staging) |

Backends that support collocated models detect same-rank (local) transfers via
`task_id` and short-circuit them into direct `tensor.copy_()` instead of going
through the network stack. NCCL M2N is the exception because its source and
destination meshes must be disjoint.

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
The backend preserves the existing ReFIT planner and packs its operations into
one logical `[source, destination, bytes]` tensor. Source ranks shard dimension
0, destination ranks shard dimension 1, and one cross-dimension
`nccl.m2n.reshard` call moves the entire batch through M2N's managed
copy/staging transport.

This backend supports only non-collocated multi-rank layouts. The communication
group must contain a contiguous source interval starting at group rank 0,
immediately followed by a contiguous destination interval, with no overlapping
or idle ranks. Use a process group scoped to exactly one source/destination pool
when the application has extra ranks. Tensor data is packed into an ordinary,
reusable CUDA tensor; model parameter storage itself is not replaced. The stock
v0.2 copy/staging path caps meshes at 16 source and 64 destination ranks. A
custom build with larger bounds can be used by constructing
`NCCLM2NCopyService` with `enforce_mesh_limits=False`.

## How the Reshard Plan Works

1. Each rank extracts parameter metadata (shape, sharding, TP/EP/PP groups).
2. Metadata is all-gathered so **every** rank has the full picture
   (`dist.all_gather_object()`) — no rank-0 bottleneck, no scatter.
3. Every rank independently replays the **same deterministic schedule**
   (`_iter_global_transfer_ops`):
   - Iterate destination ranks, then each rank's destination params in gathered
     order; for each destination param, find the matching source param(s) by name.
   - Route to a dimension-specific planner (LCM tiling for standard TP,
     block-interleaved for partitioned params like Mamba `in_proj`).
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
`fp8_recipe='mxfp8'`, an `MXFP8ReshardTransform` is automatically created
and attached to the cached plan.

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
| `_service_cache` | Backend name | `CopyService` instance | Avoid re-creating CUDA streams / NVSHMEM buffers |
| `_plan_cache` | (rank, src_config, dst_config, num_experts) | `ReshardPlan` + attached transform | Avoid collective plan rebuild on repeated refits |

Call `clear_all_caches()` before destroying distributed process groups
to avoid stale references.  This also finalizes NVSHMEM resources.

## Process Group Requirements

The source and destination models must each have a `pg_collection`
attribute with the following groups:

| Field | Required | Purpose |
|-------|----------|---------|
| `tp` | Yes | Tensor parallelism sharding |
| `dp` | Yes (auto-filled on source from `parallel_state` if missing) | Data parallelism routing |
| `pp` | If PP > 1 | Pipeline stage / layer index remapping |
| `ep` | If MoE | Expert parallelism routing |
| `expt_tp` | If expert TP | Expert-specific tensor parallelism |

## File Reference

| File | Role |
|------|------|
| `refit.py` | Public API, caching, MXFP8 auto-detection |
| `planner.py` | Local deterministic plan builder (metadata, LCM/block-interleaved planners) |
| `execution.py` | Plan executor (send/recv submission, writeback, format conversion) |
| `transforms.py` | `ReshardTransform` base class, `MXFP8ReshardTransform` |
| `utils.py` | `TransferOp`, `ReshardPlan`, `ParameterMetadata`, `ShardingDescriptor` |
| `copy_services/nccl_copy_service.py` | NCCL backend |
| `copy_services/nccl_m2n_copy_service.py` | Hierarchical NCCL M2N backend |
| `copy_services/gloo_copy_service.py` | Gloo backend |
| `copy_services/nixl_copy_service.py` | NIXL/UCX backend |
| `copy_services/nvshmem_copy_service.py` | NVSHMEM backend (delegates to `nvshmem_copy_service/`) |
| `nvshmem_copy_service/` | Full NVSHMEM implementation (planning, memory, kernels, pipeline) |
