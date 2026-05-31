# Megatron FSDP v2

> ⚠️ Prototype Implementation — Not Yet Production Ready

Megatron FSDP v2 (M-FSDP2) inherits the performance of Megatron FSDP v1 and supports API drop-in replacement for FSDP2.

## Architecture

```
v2/
├── README.md                    # This file
├── __init__.py                  # Public exports (FSDPModule, fully_shard, mixed precision policies)
├── fully_shard.py               # Public fully_shard() API entry point
├── fsdp_module.py               # FSDPModule runtime state (unshard/reshard/reduce_grad)
├── hooks.py                     # Forward/backward hook registration
├── param_group.py               # ParameterGroup — groups params with shared buffers
├── dp_buffer.py                 # DataParallelBuffer — flat buffer management
├── allocator.py                 # BucketAllocator (Temporary, StorageFreeing, TracePool)
├── mixed_precision.py           # MixedPrecisionPolicy, FP8Policy, NVFP4Policy
├── utils.py                     # Internal utilities (mesh init, backward Function)
├── design.md                    # Overlap, memory, and synchronization design
├── nvfp4_design.md              # NVFP4 primary-weights design
├── mcore_fsdp_checkpoint_design.md  # Checkpoint save/load and format conversion design
└── tp_support_design.md         # Tensor Parallelism support plan (future)
```

For the broader `megatron_fsdp` package layout, see the parent directory.

## Key Concepts

### fully_shard(module, mesh=...)

Wraps a module with FSDP sharding semantics:

1. Converts the module class to `FSDPModule` dynamically
2. Groups parameters by (device, dtype, requires_grad)
3. Creates `ParameterGroup` for each group with dedicated buffers
4. Registers forward/backward hooks for unshard/reshard/reduce
5. Replaces module parameters with `DTensor` representations

### MixedPrecisionPolicy

Controls parameter/gradient dtypes and communication precision.

| Policy | Notes |
|--------|-------|
| `MixedPrecisionPolicy` | Base policy; all fields default to ``None`` |
| `FullyShardFP8Policy` | MXFP8 rowwise/colwise quantized weights |
| `FullyShardNVFP4Policy` | NVFP4 primary weights |

**Key fields:**

| Field | Default | Purpose |
|-------|---------|---------|
| `main_params_dtype` | ``None`` | Dtype for optimizer main-weight buffer. ``None`` = no separate buffer, optimizer mutates model weights directly. Set to ``torch.float32`` for quantized models (FP8/NVFP4) so the optimizer works on high-precision copies. |
| `main_grads_dtype` | ``None`` | Dtype for optimizer main-grad buffer. When ``None`` and ``use_decoupled_grad=False``, aligns with ``main_params_dtype``. Otherwise falls back to ``param.dtype``. |
| `grad_comm_dtype` | ``None`` | Dtype for gradient reduce-scatter communication. ``None`` = use ``main_grads_dtype``. |
| `use_decoupled_grad` | ``False`` | When ``False``, ``main_grads_dtype`` is inferred from ``main_params_dtype`` so the optimizer operates in a consistent precision context. |

```python
from megatron_fsdp.v2 import fully_shard, MixedPrecisionPolicy

# No separate main buffer — optimizer mutates model params directly
mp_policy = MixedPrecisionPolicy()
fully_shard(model, mp_policy=mp_policy)

# fp32 optimizer precision for bf16 model
mp_policy = MixedPrecisionPolicy(main_params_dtype=torch.float32)
fully_shard(model, mp_policy=mp_policy)

# FP8 mixed precision — fp32 main weights auto-created by adapter
from megatron_fsdp.v2 import FullyShardFP8Policy
mp_policy = MixedPrecisionPolicy(
    main_params_dtype=torch.float32,
    fp8=FullyShardFP8Policy(enabled=True)
)
fully_shard(model, mp_policy=mp_policy)
```

### FSDPModule

Mixin class added to wrapped modules. Methods:

| Method | When Called | Purpose |
|--------|-------------|---------|
| `unshard()` | Pre-forward | All-gather params from sharded buffer |
| `reshard()` | Post-forward, post-backward | Release unsharded buffer |
| `reduce_grad()` | Post-backward / grad sync | Reduce-scatter gradients into optimizer-facing shards |

### DataParallelBuffer

Flat buffer managing (a shard of) parameter/gradient data:

- `unshard()` — all-gather to full tensor
- `reshard()` — free temporary buffer
- `reduce_grad()` — reduce-scatter gradients into optimizer-facing shards
- Uses `BufferIndex` to track parameter layout within the buffer

### ParameterGroup

Groups parameters sharing the same (device, dtype, requires_grad):

- `model_weight_buffer` — stores compute weights; replicated for ZeRO-1/2 and sharded for ZeRO-3
- `main_weight_buffer` — optional high-precision optimizer copy; sharded when optimizer state is sharded
- `main_grad_buffer` — accumulates gradients before reduce
- `dist_params` — DTensor views into the buffer

### Uneven DTensor Handling

See the parent directory `..` for `uneven_dtensor.py` which provides:

- `gather_and_compute_chunk_metadata()` — computes global offsets/sizes for each shard
- `update_uneven_dtensor_chunk_metadata()` — attaches chunk metadata to DTensor for checkpointing
- `preprocess_state_dict_for_uneven_dtensor()` — enables distributed checkpoint of uneven shards
- `split_dtensor()` — splits DTensor with proper chunk metadata preservation
- `get_state_dict()` — PyTorch DCP-compatible state dict with uneven shard support

## Sharding Strategies

| Strategy | Shard Weights | Shard Gradients | Status | Notes |
|----------|---------------|-----------------|--------|-------|
| `optim_grads_params` | Yes | Yes | **Supported** | Like ZeRO-3: full parameter/gradient/optimizer sharding |
| `no_shard` | No | No | **Not yet supported** | Like DDP: no sharding |
| `optim` | No | No | **Not yet supported** | Like ZeRO-1: shard optimizer states only |
| `optim_grads` | No | Yes | **Not yet supported** | Like ZeRO-2: shard optimizer states + gradients |

## Known Limitations

### Parallelism

- **Tensor Parallelism (TP):** Not supported. v2 currently operates on a 1D
  DP-only DeviceMesh. Parameters that are already partitioned by TP layers
  (e.g., `ColumnParallelLinear`, `RowParallelLinear`) are not correctly handled.
  See [tp_support_design.md](tp_support_design.md) for the planned design.
- **Hybrid Sharding (HSDP):** Not supported. v2 does not yet support an outer
  DP dimension for hybrid (inter-node + intra-node) sharding.

### Sharding Strategies

Only `optim_grads_params` (ZeRO-3 equivalent) is implemented. `no_shard`
(DDP-like), `optim` (ZeRO-1), and `optim_grads` (ZeRO-2) are planned but
not yet available.

### `fully_shard()` API Parameters

The following parameters are accepted in the function signature but are **not
yet implemented** (marked `TODO`):

- `reshard_after_forward` — no-op
- `shard_placement_fn` — no-op; all params use `Shard(0)` on the DP dimension
- `offload_policy` — no-op; CPU offloading is not supported

### Hardware & Platform

- **GPU only.** CUDA devices only. CPU, XPU, and ROCm are not tested or supported.
- **NVFP4** (`mixed_precision.py`): The non-distributed quantization path is
  not implemented (`FIXME` at `mixed_precision.py:686`). Distributed NVFP4
  (reduce-scatter + quantize) is functional.

### Checkpointing

- **Async checkpoint:** Not supported for the v2 path.
- **`dp_reshardable` checkpoints:** Loading from `dp_reshardable` format is
  not supported. Re-save checkpoints with `--ckpt-fully-parallel-save` first.

## Integration with Megatron

There are two ways to use Megatron FSDP v2:

### Option A: Through Megatron Core (MCore)

Set `--use-megatron-fsdp-v2` in your training arguments. The adapter
(`mcore_fsdp_adapter.py`) will automatically route to the v2 `fully_shard`
path for model sharding.

```bash
python pretrain_gpt.py \
    --use-megatron-fsdp-v2 \
    --use-megatron-fsdp \
    ...
```

This is the recommended path for Megatron-LM training workflows.

### Option B: Standalone (without Megatron-LM)

Import `fully_shard` directly from the `megatron_fsdp.v2` package:

```python
from megatron_fsdp.v2 import FSDPModule, fully_shard

model = MyModel().cuda()
fully_shard(model)
```

### Installation

**Pre-release (current):** v2 has not been released as a standalone package yet.
Clone and install via `PYTHONPATH`:

```bash
git clone -b mfsdp_refactor https://github.com/shjwudp/Megatron-LM.git
export PYTHONPATH=$PWD/Megatron-LM/megatron/core/distributed/fsdp/src:$PYTHONPATH
```

**After release** (once `megatron-fsdp` is published to PyPI):

```bash
pip install megatron-fsdp
```

The standalone import (`from megatron_fsdp.v2 import fully_shard`) will work
without any Megatron-LM dependency once the package is released.

## Toy Example

See `examples/megatron_fsdp/fsdp_toy.py` for a standalone example showing:

- Basic model wrapping with `fully_shard()`
- Training loop with gradient accumulation
- Activation checkpointing (`--activation-checkpoint`)
- Distributed checkpointing with `torch.distributed.checkpoint`

```bash
torchrun --nproc_per_node=2 examples/megatron_fsdp/fsdp_toy.py \
    --model-dim 512 --n-layers 2 --batch-size 4

# With activation checkpointing and Megatron-FSDP
torchrun --nproc_per_node=2 examples/megatron_fsdp/fsdp_toy.py \
    --model-dim 512 --n-layers 2 --batch-size 4 \
    --use-megatron-fsdp --activation-checkpoint
```

## Gotchas / Pitfalls

- **Zero-numel gradient shards and fused optimizers.** When a parameter's local shard is empty on some DP ranks (e.g., small biases on high DP counts), creating a `DTensor` gradient with `numel() == 0` and passing it to fused multi-tensor optimizers (TE `FusedAdam`) can silently corrupt updates for neighboring non-empty parameters. This manifests only as convergence divergence with no error — see [design.md § Pitfall](design.md) for details and the fix in `param_group.py`.
- **Temporary communication buckets must record their CUDA stream.** `DataParallelBuffer.reshard()` and `release_grad_buffer()` can return temporary all-gather / reduce-scatter bucket storage to the allocator while side-stream CUDA work is still queued. A CUDA event wait orders streams, but it does not tell PyTorch's caching allocator that the freed storage is still used by `ag_stream` or `rs_stream`. If the block is recycled on another stream, the communication payload can be silently corrupted. Call `record_stream()` on any temporary CUDA tensor that participates in side-stream communication before it may be freed; a full synchronization only masks the bug and hurts overlap.

## Unit Tests

```bash
# Run all v2 unit tests (requires 2 GPUs)
TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 \
torchrun --nproc_per_node=2 -m pytest \
    tests/unit_tests/distributed/megatron_fsdp/v2/ -v -x

# Run specific test files
torchrun --nproc_per_node=2 -m pytest \
    tests/unit_tests/distributed/megatron_fsdp/v2/test_fully_shard.py -v

TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 \
torchrun --nproc_per_node=2 -m pytest \
    tests/unit_tests/distributed/megatron_fsdp/v2/test_mcore_checkpoint.py -v

# Single-GPU tests
pytest tests/unit_tests/distributed/megatron_fsdp/v2/ -v \
    -k "test_double_shard_rejected or test_no_params_module or test_get_state_dict_strict"
```