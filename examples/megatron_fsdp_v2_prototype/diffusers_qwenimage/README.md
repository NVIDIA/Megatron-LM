# Diffusers QwenImage: FSDP1 vs Megatron-FSDP

Minimal repro comparing PyTorch FSDP1 against Megatron-FSDP on
`QwenImageTransformer2DModel`.  All code is self-contained in `test_qwenimage.py`;
only stock packages are required below.

Run the commands below from the example directory:

```bash
cd examples/megatron_fsdp_v2_prototype/diffusers_qwenimage
```

## Environment Setup (run once)

```bash
pip install "diffusers>=0.37.0"           # QwenImage model
pip install megatron-fsdp                 # if not installed in repo
pip install huggingface_hub               # model download

# Flash attention (pick one tier):
#   Tier 1: FA3 — best perf, install from PyPI
pip install flash_attn_interface
#   Tier 2: FA2 — if FA3 unavailable
pip install flash-attn --no-build-isolation
#   Tier 3: no flash-attn at all → use --attention native below
```

## Download Model (run once)

```bash
hf download Qwen/Qwen-Image \
  --include "transformer/*" \
  --local-dir /tmp/qwen-image
```

## Run

### Single node, 4 GPU (full shard)

```bash
# Tier 1: FA3 (_flash_3)
torchrun --nproc_per_node=4 test_qwenimage.py \
  --backend mfsdp --sharding full \
  --pretrained_model_name_or_path /tmp/qwen-image \
  --num_gpus_per_node 4 --batch_size 4 --height 512 --width 512 \
  --attention _flash_3 --compile --bench_steps 20 --warmup_steps 3

# Tier 2: FA2 (flash) — if FA3 unavailable
nsys profile \
torchrun --nproc_per_node=4 test_qwenimage.py \
  --backend mfsdp --sharding full \
  --pretrained_model_name_or_path /tmp/qwen-image \
  --num_gpus_per_node 4 --batch_size 4 --height 512 --width 512 \
  --attention flash --compile --bench_steps 3 --warmup_steps 1

# Tier 3: native attention — no flash-attn needed
torchrun --nproc_per_node=4 test_qwenimage.py \
  --backend mfsdp --sharding full \
  --pretrained_model_name_or_path /tmp/qwen-image \
  --num_gpus_per_node 4 --batch_size 4 --height 512 --width 512 \
  --attention native --compile --bench_steps 20 --warmup_steps 3

# PyTorch FSDP1 for comparison (swap --backend)
nsys profile \
torchrun --nproc_per_node=4 test_qwenimage.py \
  --backend fsdp1 --sharding full \
  --pretrained_model_name_or_path /tmp/qwen-image \
  --num_gpus_per_node 4 --batch_size 4 --height 512 --width 512 \
  --attention flash --compile --bench_steps 20 --warmup_steps 3
```

### Megatron-FSDP v2 with CUDA graphs

CUDA graphs are available only with the `mfsdpv2` backend. Enabling CUDA
graphs automatically selects `TracePoolAllocator`; `--trace-pool` is therefore
not required in the CUDA-graph command.

```bash
torchrun --nproc_per_node=4 test_qwenimage.py \
  --backend mfsdpv2 --sharding full \
  --pretrained_model_name_or_path /tmp/qwen-image \
  --num_gpus_per_node 4 --batch_size 4 --height 512 --width 512 \
  --attention flash --compile --cuda-graph \
  --bench_steps 20 --warmup_steps 3 --real-data
```

### Single node, 8 GPU (hybrid shard)

```bash
torchrun --nproc_per_node=8 test_qwenimage.py \
  --backend mfsdp --sharding hybrid \
  --pretrained_model_name_or_path /tmp/qwen-image \
  --batch_size 4 --height 512 --width 512 \
  --attention _flash_3 --compile --bench_steps 20 --warmup_steps 3
```

### With numerical verification (adds sync overhead)

```bash
torchrun --nproc_per_node=4 test_qwenimage.py \
  --backend mfsdp --sharding full \
  --pretrained_model_name_or_path /tmp/qwen-image \
  --num_gpus_per_node 4 --batch_size 4 --height 512 --width 512 \
  --attention flash --compile --verify
```

### Multi-node (2+ nodes)

```bash
torchrun --nnodes=$NNODES --node_rank=$NODE_RANK \
  --nproc_per_node=8 --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT \
  test_qwenimage.py \
  --backend mfsdp --sharding hybrid \
  --pretrained_model_name_or_path /tmp/qwen-image \
  --batch_size 4 --height 512 --width 512 \
  --attention _flash_3 --compile --bench_steps 20 --warmup_steps 3
```

## Benchmark results

The following comparison uses one reproducible configuration for both
backends. It was measured at commit `c2e9b8e00700` after the lazy distributed
gradient-storage fix.

| Setting | Value |
| --- | --- |
| Hardware | 4×GB200 |
| Model | Pretrained 60-block `QwenImageTransformer2DModel` |
| Data | Real-data training path |
| Sharding | Full shard |
| Per-rank batch size | 4 |
| Image size | 512×512 |
| Precision and attention | BF16, FlashAttention 2 |
| Compilation | Per-block `torch.compile` |
| Measurement | 3 warmup steps + 20 measured steps |

### Eager performance and convergence

| Backend | Average step | Median step | Peak memory | Final / initial loss |
| --- | ---: | ---: | ---: | ---: |
| PyTorch FSDP1 | **521.70 ms** | **521.46 ms** | 75.39 GB | 0.641 |
| Megatron-FSDP v2 | 592.67 ms | 552.33 ms | **74.67 GB** | 0.636 |

Both runs pass the 20-step convergence threshold of 0.7. V2 uses 0.72 GB less
peak memory and is 5.9% slower by median step time. Two approximately 0.9-second
v2 outliers widen the average-time gap to 13.6%; both values are retained so
the result does not hide the tail behavior. The memory result confirms that v2
does not retain an additional main-weight buffer and that unused lazy gradient
storage is released before the next unshard.

### CUDA graph validation

CUDA graph capture was also tested at the same commit. Both invocations reach
the first warmup step and record all 60 transformer blocks, but fail while
constructing the static gradient buffers, before any measured iteration.

| Invocation | Result |
| --- | --- |
| `--cuda-graph` | Failed during graph capture |
| `--cuda-graph --trace-pool` | Failed during graph capture |

Both variants report `TracePoolAllocator slot collision` while
`get_main_grad()` allocates a `main_grad` buffer. This is the same path because
CUDA graphs enable the trace-pool allocator internally. No performance number
is reported for CUDA graphs until the traced and replayed gradient-buffer
lifetimes agree.

### Nsight Systems analysis

Three additional steps were captured under Nsight Systems with the same model
and runtime configuration. Profiler-instrumented timings are diagnostic and
should not be compared directly with the 20-step performance table.

| Per-rank metric | PyTorch FSDP1 | Megatron-FSDP v2 |
| --- | ---: | ---: |
| Forward | 252.84 ms | 256.56 ms |
| Backward | 415.52 ms | 520.79 ms |
| Optimizer | 5.79 ms | 17.02 ms |
| Maximum reduce-scatter start skew | 5.42 ms | 357.24 ms |

The slow v2 step is caused by cross-rank launch skew rather than lower
collective bandwidth. One rank remains in a single `MFSDP reduce_grad` CPU
range for 353.08 ms before launching reduce-scatter collective 115. The other
three ranks enter about 357 ms earlier and wait inside NCCL; their kernels last
approximately 358 ms while the late rank's kernel lasts 0.88 ms. Python-side
work before `param_group.reduce_grad()` is therefore the primary optimization
target.

## torch FSDP1 (reference API)

```python
mesh = init_device_mesh("cuda", (num_nodes, gpus_per_node),
                        mesh_dim_names=("replicate", "shard"))
mp = MixedPrecision(param_dtype=bf16, reduce_dtype=bf16, buffer_dtype=bf16)
FSDP(model, device_mesh=mesh, sharding_strategy=ShardingStrategy.HYBRID_SHARD,
     mixed_precision=mp,
     auto_wrap_policy=ModuleWrapPolicy({QwenImageTransformerBlock}),
     backward_prefetch=BackwardPrefetch.BACKWARD_PRE, forward_prefetch=True,
     use_orig_params=True, limit_all_gathers=True)
```

## Megatron-FSDP (reference API)

```python
mesh = init_device_mesh("cuda", (num_nodes, gpus_per_node),
                        mesh_dim_names=("dp_outer", "dp_shard"))
mesh[("dp_outer", "dp_shard")]._flatten("hsdp")
mp = MixedPrecisionPolicy(main_params_dtype=bf16, main_grads_dtype=bf16,
                          grad_comm_dtype=bf16)
fully_shard_model(
    module=model, fsdp_unit_modules=[QwenImageTransformerBlock],
    device_mesh=mesh, dp_shard_dim="dp_shard",
    dp_outer_dim="dp_outer", hybrid_fsdp_group=mesh["hsdp"].get_group(),
    zero_dp_strategy=3,                # ZeRO-3 within node
    outer_dp_sharding_strategy=0,      # REPLICATE across nodes (match FSDP1 HYBRID_SHARD)
    sync_model_each_microbatch=True,
    overlap_grad_reduce=True, overlap_param_gather=True,
    mixed_precision_policy=mp,
)

# Optimizer
fully_shard_optimizer(optim)
optim.step(sync_grad_before_optimizer_step=True, install_optimized_model_weights=True)
optim.zero_grad(set_to_none=True, zero_grad_buffer=True)
```

## Notes

- Per-block `torch.compile` (identical for both backends):
  ```python
  for blk in model.module.transformer_blocks:
      blk.compile()
  ```
- `--verify` gradient probe — mfsdp does not expose gradients via `param.grad`:
  ```python
  def _local_grad(p):
      g = p.grad
      if g is None and hasattr(p, "get_main_grad"): g = p.get_main_grad()
      if g is None: g = getattr(p, "main_grad", None)
      if g is None: g = getattr(p, "decoupled_grad", None)
      if hasattr(g, "to_local"): g = g.to_local()
      return g
  ```
- FA3 autograd shim is applied inline by the script — no manual patching required.
- `hybrid` sharding on a single node falls back to `full` shard automatically.
