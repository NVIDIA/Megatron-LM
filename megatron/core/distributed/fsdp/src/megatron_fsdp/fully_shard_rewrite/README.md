# Megatron FSDP2 (fully_shard_v2)

This directory contains the **fully_shard_v2** implementation — a PyTorch FSDP2-compatible API layer for Megatron Core.

## Architecture

```
fully_shard_rewrite/
├── README.md                    # This file
├── __init__.py                  # Public exports
├── fully_shard.py               # Core fully_shard() API + FSDPModule class
├── param_group.py               # ParameterGroup — groups params with shared buffers
├── dp_buffer.py                 # DataParallelBuffer — flat buffer management
├── allocator.py                 # TemporaryBucketAllocator — temp buffer reuse
├── utils.py                     # Internal utility functions
└── design.md                    # Detailed design documentation (overlap, memory, sync)
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

### FSDPModule

Mixin class added to wrapped modules. Methods:

| Method | When Called | Purpose |
|--------|-------------|---------|
| `unshard()` | Pre-forward | All-gather params from sharded buffer |
| `reshard()` | Post-forward, post-backward | Release unsharded buffer |
| `reduce_grad()` | Post-backward | All-reduce or reduce-scatter gradients |

### DataParallelBuffer

Flat buffer managing (a shard of) parameter/gradient data:

- `unshard()` — all-gather to full tensor
- `reshard()` — free temporary buffer
- `reduce_grad()` — all-reduce or reduce-scatter gradients
- Uses `BufferIndex` to track parameter layout within the buffer

### ParameterGroup

Groups parameters sharing the same (device, dtype, requires_grad):

- `model_weight_buffer` — stores sharded model weights
- `main_weight_buffer` — optional high-precision copy
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

| Strategy | Shard Weights | Shard Gradients | Notes |
|----------|---------------|-----------------|-------|
| `no_shard` | No | No | Like DDP: no sharding |
| `optim` | No | No | Like ZeRO-1: shard optimizer states only |
| `optim_grads` | No | Yes | Like ZeRO-2: shard optimizer states + gradients |
| `optim_grads_params` | Yes | Yes | Like ZeRO-3: full parameter/gradient/optimizer sharding |

## Integration with Megatron

`FullyShardedDataParallel` in `mcore_fsdp_adapter.py` supports two code paths:

1. **Legacy path** (`use_fully_shard_api=False`): Uses `MegatronFSDP` from the original megatron_fsdp module
2. **fully_shard path** (`use_fully_shard_api=True`): Uses PyTorch or Megatron's `fully_shard_v2`

### Using fully_shard_v2

```python
from megatron.core.distributed.fsdp.mcore_fsdp_adapter import FullyShardedDataParallel

# In your config:
ddp_config.data_parallel_sharding_strategy = "optim_grads_params"
ddp_config.use_fully_shard_api = True

# During model setup:
model = FullyShardedDataParallel(
    config=transformer_config,
    ddp_config=ddp_config,
    module=model,
)
```

## Toy Example

See `examples/megatron_fsdp/fsdp_toy.py` for a standalone example showing:

- Basic model wrapping with `fully_shard()`
- Training loop with gradient accumulation
- Distributed checkpointing with `torch.distributed.checkpoint`

```bash
torchrun --nproc_per_node=2 examples/megatron_fsdp/fsdp_toy.py \
    --model-dim 512 --n-layers 2 --batch-size 4
```

## Gotchas / Pitfalls

- **Zero-numel gradient shards and fused optimizers.** When a parameter's local shard is empty on some DP ranks (e.g., small biases on high DP counts), creating a `DTensor` gradient with `numel() == 0` and passing it to fused multi-tensor optimizers (TE `FusedAdam`) can silently corrupt updates for neighboring non-empty parameters. This manifests only as convergence divergence with no error — see [design.md § Pitfall](design.md) for details and the fix in `param_group.py`.

## Unit Tests

```bash
# Run FSDP2 API tests
pytest -xvs tests/unit_tests/distributed/megatron_fsdp/test_mcore_fsdp_fully_shard_v2_api.py
```