# Fine-Grained Activation Offloading

Fine-grained activation offloading reduces GPU memory by asynchronously transferring activations to CPU at the granularity of individual submodules within a transformer layer. Unlike layer-level offloading, it allows precise control over which activations to offload, enabling a tradeoff between memory savings and PCIe bandwidth overhead.

## User Guide

### Basic Usage

```bash
# Enable fine-grained activation offloading
--fine-grained-activation-offloading

# Specify which modules to offload (can combine multiple)
# Choices: attn_norm, qkv_linear, core_attn, attn_proj, mlp_norm, expert_fc1, moe_act
--offload-modules core_attn attn_proj expert_fc1
```

### Offloadable Modules

Each module offloads its **input** activation to CPU during forward and reloads it before backward:

| Module | Description | Notes |
|---|---|---|
| `attn_norm` | Input layernorm of attention | Skipped if using `IdentityOp` |
| `qkv_linear` | QKV linear projection | |
| `core_attn` | Core attention (softmax + matmul) | |
| `attn_proj` | Output projection of attention | Must be used together with `core_attn` |
| `mlp_norm` | Pre-MLP layernorm | Skipped if using `IdentityOp` |
| `expert_fc1` | First FC layer in MoE experts | MoE models only |
| `moe_act` | Activation function in MoE experts | MoE models only |

### Tuning Parameters

```bash
# Minimum tensor size (in elements) to offload. Smaller tensors are skipped.
# Default: 1048576 (1M elements)
--min-offloaded-tensor-size 1048576

# Fraction of activations to offload, range [0, 1]. Default: 1.0
# Useful for partial offloading when PCIe bandwidth is a bottleneck.
--activation-offload-fraction 0.8

# Reduce offload amount on higher PP ranks (in bytes). Default: 0
# Higher PP ranks have fewer microbatches in flight, so offloading less
# reduces overhead without increasing peak memory.
--delta-offload-bytes-across-pp-ranks 1073741824
```

### CUDA Graph Integration

Fine-grained offloading is compatible with CUDA graphs. When CUDA graph is enabled, the following constraints apply:

- `attn_norm` and `mlp_norm` **cannot** be offloaded (they cross CUDA graph boundaries).
- `cuda_graph_scope` must include `attn` and `moe_router`.
- `cuda_graph_impl` must be `transformer_engine`.
- Requires `torch >= 2.9.0` and `transformer_engine >= 2.13.0`.

```bash
# Delay offloading until CUDA graph launch to hide CPU overhead
--delay-offload-until-cuda-graph
```

### Combining with Fine-Grained Recomputation

Offloading and recomputation are complementary:
- Use **recomputation** for lightweight modules (e.g., layernorm, activation functions) with negligible compute overhead.
- Use **offloading** for heavy modules (e.g., core_attn, expert_fc1) where recomputation would be too costly.

```bash
--recompute-granularity selective
--recompute-modules layernorm moe_act
--fine-grained-activation-offloading
--offload-modules core_attn attn_proj expert_fc1
```

![Fine-grained Activation Offloading and Fine-grained Recomputation](../../images/fine_grained_activation_offloading/offloading_and_recomputing.png)


### Compatibility

| Feature | Supported |
|---|---|
| PP / Interleaved PP / PP=1 | Yes |
| Fine-grained recomputation | Yes |
| FP8 training | Yes |
| MTP (Multi-Token Prediction) | Yes |
| Mixed dense & MoE layers | Yes |
| A2A overlap (EP) | Yes |
| CUDA Graph (TE impl) | Yes |

---

## How It Works

### Architecture Overview

The implementation consists of three layers:

1. **`PipelineOffloadManager`** (singleton): Global coordinator that manages CUDA streams, CPU tensor pools, and chunk lifecycle across pipeline stages.
2. **`ChunkOffloadHandler`**: Per-microbatch handler that tracks tensor groups, executes D2H/H2D transfers, and decides which groups to actually offload.
3. **`FineGrainedActivationOffloadingInterface`**: Lightweight interface used by transformer modules (attention, MoE, etc.) to mark offload boundaries.

### Offload/Reload Flow

```
Forward pass (Layer N):                    Backward pass (Layer N):
┌─────────────────────┐                    ┌───────────────────────┐
│ group_start(input)  │─── register ──►    │                       │
│                     │    tensor group    │ group_commit_backward │
│ module.forward()    │                    │   wait H2D complete   │
│                     │                    │   pop tensors from    │
│ group_offload(out)  │─── D2H async ──►   │   CPU → GPU           │
│   on d2h_stream     │    to pinned CPU   │   on h2d_stream       │
└─────────────────────┘                    └───────────────────────┘
```

1. **`group_start`**: Registers a new tensor group and hooks into `saved_tensors_hooks` to intercept `save_for_backward`.
2. **Forward execution**: All tensors saved by autograd within the group are captured.
3. **`group_offload`**: Triggers asynchronous D2H copy on a dedicated CUDA stream (`d2h_stream`), optionally releases GPU storage of input tensors.
4. **Backward**: Before the group's backward, tensors are reloaded from CPU to GPU on `h2d_stream`, and the compute stream waits for the transfer to complete.

### Warmup and Adaptive Offloading

The first training iteration serves as a **warmup phase** where the manager records tensor groups, their sizes, and the execution order. After warmup, a `post_warmup_callback` runs to:

1. **Reserve margin**: The last N groups (by deduplication count) are kept on GPU to avoid reload blocking the compute stream.
2. **Apply PP rank delta**: Higher PP ranks offload fewer bytes (controlled by `delta_offload_bytes_across_pp_ranks`).
3. **Apply fraction**: Only a fraction of eligible groups are actually offloaded (controlled by `activation_offload_fraction`).
4. **Print summary table**: An ASCII table of per-rank offload bytes is printed for debugging.

### CPU Tensor Pool

A `GPUTensorPool` (on CPU with pinned memory) caches allocated tensors by `(shape, dtype)`. This avoids repeated `cudaMallocHost` / `cudaFreeHost` calls and reduces D2H latency after the first iteration.

### CUDA Graph Support

When offloading modules captured inside a CUDA graph:

- A dedicated `cuda_graph_stream` runs the captured computation, while `d2h_stream` overlaps D2H transfers.
- During CUDA graph **warmup**, offloading is disabled (`pre_warmup_hook` / `post_warmup_hook`).
- The `delay_offload_until_cuda_graph` option defers D2H launches until graph replay, utilizing the CPU idle time during `cudaGraphLaunch` to issue offload commands with near-zero CPU overhead.
