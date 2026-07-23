# Megatron FSDP v2

> [!WARNING]
> Megatron FSDP v2 is an experimental prototype and is not production ready.

Megatron FSDP v2 (M-FSDP2) brings a PyTorch FSDP2-style `fully_shard()` API to
Megatron Core. It uses DTensor to shard model weights, gradients, and optimizer
states while integrating with Megatron's distributed optimizer, mixed-precision
training, checkpointing, communication overlap, HSDP, and CUDA Graph execution.

M-FSDP2 currently implements the subset of PyTorch FSDP2 required by supported
Megatron workflows. See [Known Limitations](#known-limitations) for details.

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
├── buffer_index.py              # Flat-buffer item indexing and shard metadata
├── allocator.py                 # BucketAllocator (Temporary, StorageFreeing, TracePool)
├── cuda_graph_runner.py         # Per-module CUDA graph capture orchestration
├── mixed_precision.py           # MixedPrecisionPolicy, FullyShardFP8Policy, FullyShardNVFP4Policy
├── utils.py                     # Internal utilities (mesh init, backward Function)
├── te_graph_runtime/            # Vendored TE CUDA graph runtime compatibility layer
└── design/
    ├── design.md                              # Overlap, memory, and synchronization design
    ├── trace_pool_allocator_design.md         # TracePoolAllocator design
    ├── mfsdp_v2_builtin_cuda_graph_design.md  # Per-module CUDA graph design
    ├── full_iteration_cuda_graph_design.md    # Full-iteration CUDA graph design
    ├── hsdp_design.md                         # HSDP mesh, buffer layouts, and conversions
    ├── hooks_api.md                           # Hook API contracts and Q&A
    ├── lazy_grad_buffer_design.md             # Lazy main grad buffer lifecycle
    ├── mixed_precision_training_design.md     # Mixed precision training support
    └── mcore_fsdp_checkpoint_design.md        # Checkpoint save/load and conversion design
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

**Key fields:**

| Field | Default | Purpose |
|-------|---------|---------|
| `main_params_dtype` | ``None`` | Dtype for optimizer main-weight buffer. ``None`` = no separate buffer, optimizer mutates model weights directly. Set to ``torch.float32`` for quantized models (FP8/NVFP4) so the optimizer works on high-precision copies. When this equals the model-weight dtype and the sharding layout matches, the separate buffer is skipped automatically to avoid a redundant copy. |
| `main_grads_dtype` | ``None`` | Dtype for optimizer main-grad buffer. When ``None`` and ``use_decoupled_grad=False``, aligns with ``main_params_dtype``. Otherwise falls back to ``param.dtype``. |
| `grad_comm_dtype` | ``None`` | Dtype for gradient reduce-scatter communication. ``None`` = use ``main_grads_dtype``. |
| `use_decoupled_grad` | ``False`` | When ``False``, ``main_grads_dtype`` is inferred from ``main_params_dtype`` so the optimizer operates in a consistent precision context. |

**FP8 & NVFP4 recipes**

`FullyShardFP8Policy` and `FullyShardNVFP4Policy` are recipe dataclasses
that configure quantized mixed-precision behavior within `MixedPrecisionPolicy`,
passed via the ``fp8`` and ``nvfp4`` fields respectively.  They are not
standalone policies.

```python
from megatron_fsdp.v2 import (
    fully_shard, MixedPrecisionPolicy,
    FullyShardFP8Policy, FullyShardNVFP4Policy,
)

# No separate main buffer — optimizer mutates model params directly
mp_policy = MixedPrecisionPolicy()
fully_shard(model, mp_policy=mp_policy)

# fp32 optimizer precision for bf16 model
mp_policy = MixedPrecisionPolicy(main_params_dtype=torch.float32)
fully_shard(model, mp_policy=mp_policy)

# FP8 mixed precision — fp32 main weights + MXFP8 rowwise/colwise quantized compute
mp_policy = MixedPrecisionPolicy(
    main_params_dtype=torch.float32,
    fp8=FullyShardFP8Policy(enabled=True),
)
fully_shard(model, mp_policy=mp_policy)

# NVFP4 mixed precision — fp32 main weights + NVFP4 primary compute weights
mp_policy = MixedPrecisionPolicy(
    main_params_dtype=torch.float32,
    nvfp4=FullyShardNVFP4Policy(enabled=True),
)
fully_shard(model, mp_policy=mp_policy)
```

### FSDPModule

Mixin class added to wrapped modules. Methods:

| Method | When Called | Purpose |
|--------|-------------|---------|
| `unshard()` | Pre-forward | All-gather params from sharded buffer |
| `reshard()` | Post-forward, post-backward | Release unsharded buffer |
| `reduce_grad()` | Post-backward / grad sync | All-reduce no-shard grads or reduce-scatter ZeRO grads |

### CUDA Graph Capture

> **Experimental** — CUDA graph support in Megatron FSDP v2 is an experimental
> feature.  The API and behaviour may change in future releases without notice.

**Why MFSDP v2 can support CUDA graphs.**  The [`TracePoolAllocator`](allocator.py)
traces one micro-batch, assigns each FSDP temporary buffer key to a stable slot,
and returns the same cached tensor view on later micro-batches.  Those stable
addresses are the memory foundation that CUDA graph capture requires.

When using the standalone API, enable per-module capture with
``enable_cuda_graph=True``:

```python
for layer in model.layers:
    fully_shard(layer, enable_cuda_graph=True)
fully_shard(model)  # root without CUDA graph
```

**How it works:**

1. The first optimized forward pass records sample arguments for each
   eligible module.
2. After the first backward completes, a single batch call to
   `te-graph-runtime`'s `make_graphed_callables` captures forward + backward
   graphs for all modules in correct order (fwds in forward-module order,
   bwds in reverse) using the shared trace pool.
3. FSDP unshard/reshard hooks run **outside** the CUDA graph capture via
   `capture_time_hooks` — they are never graphed.  During replay they
   fire normally around the graphed forward/backward.
4. Replay runs entirely through the captured graphs — no Python hooks fire
   inside the graphed region.

**Limitation — nesting:** A parent FSDP module that contains other FSDP
modules as children **cannot** use ``enable_cuda_graph=True``.  Only leaf
FSDP modules (those without FSDP children) are eligible.  Attempting to
enable CUDA graph on a module with FSDP children raises a ``RuntimeError``.

```python
# OK — layers are leaf FSDP modules
for layer in model.layers:
    fully_shard(layer, enable_cuda_graph=True)

# NOT OK — model contains FSDP layers as children
fully_shard(model, enable_cuda_graph=True)   # raises RuntimeError
```

See [`design/mfsdp_v2_builtin_cuda_graph_design.md`](design/mfsdp_v2_builtin_cuda_graph_design.md)
for the full per-module architecture.

### DataParallelBuffer

Flat buffer managing (a shard of) parameter/gradient data:

- `unshard()` — all-gather to full tensor
- `reshard()` — free temporary buffer
- `reduce_grad()` — all-reduce no-shard grads or reduce-scatter ZeRO grads
- Uses `BufferIndex` to track parameter layout within the buffer

### ParameterGroup

Groups parameters sharing the same (device, dtype, requires_grad):

- `model_weight_buffer` — stores compute weights; replicated for no-shard/ZeRO-1/2 and sharded for ZeRO-3
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

All strategies except `no_shard` use `Shard(0)` DTensor placements. The
strategy controls which buffers and communication collectives are used.

| Strategy | Shard Weights | Shard Gradients | Status | Notes |
|----------|---------------|-----------------|--------|-------|
| `optim_grads_params` | Yes | Yes | **Supported** | Like ZeRO-3: all-gather weights pre-forward, reduce-scatter grads during backward, sharded optimizer states |
| `optim_grads` | No | Yes | **Supported** | Like ZeRO-2: replicated weights, reduce-scatter grads during backward, sharded optimizer states. No param-gather overlap. |
| `optim` | No | No | **Supported** | Like ZeRO-1: replicated weights, grads accumulated in replicated buffer, single reduce-scatter at ``finish_grad_sync``, sharded optimizer states. No param-gather overlap. |
| `no_shard` | No | No | **Supported** | Like DDP: replicated weights, full-gradient all-reduce, replicated optimizer states. No param-gather overlap. |

## Known Limitations

### Parallelism

- **Tensor Parallelism (TP):** The FSDP DeviceMesh contains DP/EDP dimensions,
  not a TP placement. TP layers can be used with MCore training, but their TP
  partitioning is represented by the layer implementation rather than by FSDP
  DTensor placements.
- **Hybrid Sharding (HSDP):** A 2D `(dp_outer, dp/edp)` mesh supports outer
  replication and outer optimizer-state sharding. Outer `optim` currently
  requires inner `optim_grads_params`; NVFP4 outer optimizer sharding is not
  supported. See [design/hsdp_design.md](design/hsdp_design.md).

### Sharding Strategies

`no_shard` (DDP-like), `optim` (ZeRO-1), `optim_grads` (ZeRO-2), and
`optim_grads_params` (ZeRO-3 equivalent) are implemented. `no_shard`,
`optim`, and `optim_grads` use replicated compute weights, so parameter gather
overlap (prefetch/unshard pipelining) is not applicable.

### CUDA Graph

- **Experimental.** Enable via ``enable_cuda_graph=True`` on leaf FSDP modules.
  In Megatron-LM training, use ``--mfsdp-cuda-graph`` with one or more module
  selectors (`transformer`, `mamba`, `attn`, `mlp`, `moe`, `moe_router`).
  Built on vendored [te-graph-runtime](https://github.com/buptzyb/te-graph-runtime)
  with local modifications. See
  [`design/mfsdp_v2_builtin_cuda_graph_design.md`](design/mfsdp_v2_builtin_cuda_graph_design.md).
- **Requires `TracePoolAllocator`.** CUDA graph capture depends on the
  deterministic buffer addresses provided by the trace pool. It is selected
  automatically when ``enable_cuda_graph=True``.
- **Nesting not supported.** Only leaf FSDP modules (those without FSDP children)
  are eligible for capture.

### `fully_shard()` API Parameters

The following PyTorch FSDP2-compatible parameters are present in the function
signature but are **not supported yet**. Passing any of them raises
`NotImplementedError`:

- `reshard_after_forward`
- `shard_placement_fn`; all parameters currently use `Shard(0)` on the DP dimension
- `offload_policy`; CPU offloading is not supported

### Hardware & Platform

- **GPU only.** CUDA devices only. CPU, XPU, and ROCm are not tested or supported.
- **NVFP4** (`mixed_precision.py`): The non-distributed quantization path is
  not implemented. Distributed NVFP4 quantization is implemented for sharded
  model-weight buffers.

### Checkpointing

- **Async checkpoint:** Not supported for the v2 path.
- **`dp_reshardable` checkpoints:** Loading from `dp_reshardable` format is
  not supported. Re-save checkpoints with
  `--dist-ckpt-optim-fully-reshardable` first.

### MCore Integration

- **`--ddp-average-in-collective`:** Not supported with Megatron FSDP v2. The
  adapter raises a `NotImplementedError` if both options are enabled.

## Integration with Megatron

There are two ways to use Megatron FSDP v2:

### Option A: Through Megatron Core (MCore)

Set `--use-megatron-fsdp-v2` in your training arguments. Argument validation
sets `--use-megatron-fsdp` automatically and the adapter (`mcore_fsdp_adapter.py`)
routes model sharding to the v2 `fully_shard` path.

```bash
python pretrain_gpt.py \
    --use-megatron-fsdp-v2 \
    ...
```

This is the recommended path for Megatron-LM training workflows.

### Option B: Standalone API from the source tree

Import `fully_shard` directly from the `megatron_fsdp.v2` package when the
source tree is on `PYTHONPATH`:

```python
from megatron_fsdp.v2 import FSDPModule, fully_shard

model = MyModel().cuda()
fully_shard(model)
```

### Installation

Megatron FSDP v2 is not published as a standalone PyPI package from this source
tree. Use a Megatron-LM checkout and add the FSDP source directory to
`PYTHONPATH`:

```bash
export PYTHONPATH=/path/to/Megatron-LM/megatron/core/distributed/fsdp/src:$PYTHONPATH
```

## Toy Example

See
`examples/megatron_fsdp_v2_prototype/fsdp_toy/fsdp_toy.py` for a standalone
example showing:

- Basic model wrapping with `fully_shard()`
- CUDA graph capture (`--cuda-graph`, disabled by default)
- Training loop with gradient accumulation
- Activation checkpointing (`--activation-checkpoint`)
- Distributed checkpointing with `torch.distributed.checkpoint`

```bash
torchrun --nproc_per_node=2 \
    examples/megatron_fsdp_v2_prototype/fsdp_toy/fsdp_toy.py \
    --model-dim 512 --n-layers 2 --batch-size 4

# With activation checkpointing and Megatron-FSDP
torchrun --nproc_per_node=2 \
    examples/megatron_fsdp_v2_prototype/fsdp_toy/fsdp_toy.py \
    --model-dim 512 --n-layers 2 --batch-size 4 \
    --use-megatron-fsdp --activation-checkpoint

# Enable CUDA graph capture and trace-pool allocation
torchrun --nproc_per_node=2 \
    examples/megatron_fsdp_v2_prototype/fsdp_toy/fsdp_toy.py \
    --model-dim 512 --n-layers 2 --batch-size 4 \
    --use-megatron-fsdp --cuda-graph --use-trace-pool
```

## Gotchas / Pitfalls

- **Zero-numel gradient shards and fused optimizers.** When a parameter's local shard is empty on some DP ranks (e.g., small biases on high DP counts), creating a `DTensor` gradient with `numel() == 0` and passing it to fused multi-tensor optimizers (TE `FusedAdam`) can silently corrupt updates for neighboring non-empty parameters. This manifests only as convergence divergence with no error — see [design.md § Pitfall](design/design.md) for details and the fix in `param_group.py`.
- **Temporary communication bucket lifecycle.** Temporary all-gather /
  reduce-scatter buckets are allocated on the caller CUDA stream. Parameter
  all-gathers run on `ag_stream`; gradient collectives run on `rs_stream`,
  where full-iteration graphs may also stage add/copy/zero work immediately
  before reduction. CUDA events order preparation, communication, consumption,
  and free. All all-gather outputs additionally record their producer stream so
  the allocator cannot recycle a temporary buffer while communication is using it.

## Unit Tests

```bash
# Run v2 unit tests. Most distributed tests require at least 2 GPUs; some
# HSDP/EP cases require more and will skip when the topology is unavailable.
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

## Acknowledgements

Megatron-FSDP v2 prototype development and validation benefited from close
collaboration across the China Devtech Team. Special
thanks go to [Tong Liu](https://github.com/Autumn1998), [Hongbin Liu](https://github.com/lhb8125), [Ruby Chen](https://github.com/RubiaCx), [Robin Zhang](https://github.com/buptzyb), and [Jianbin Chang](https://github.com/shjwudp)
for their contributions to the design, implementation, benchmarking, debugging,
and validation of this work.
