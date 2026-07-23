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

The following comparison uses one reproducible configuration for all cases. It
was measured at merged `mfsdp_refactor` commit `31334f8807d6`, including the
lazy gradient-storage, trace-pool gradient-lifetime, and gradient-DTensor
wrapper-reuse fixes.

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
| PyTorch FSDP1 | 516.03 ms | **491.23 ms** | 75.39 GB | 0.642 |
| Megatron-FSDP v2 | 505.89 ms | 506.46 ms | **74.67 GB** | 0.634 |
| Megatron-FSDP v2 + CUDA graph | **385.89 ms** | **385.53 ms** | 86.72 GB | 0.641 |

All three runs pass the 20-step convergence threshold of 0.7. Eager v2 is 1.96%
faster than FSDP1 by average step because FSDP1 contains one 824.48 ms tail,
while FSDP1 is 3.10% faster by median step. Eager v2 uses 0.72 GB less peak
memory. Compared with eager v2, CUDA graph improves average step time by 23.72%
and median step time by 23.88%, while increasing peak memory by 12.05 GB.

The eager v2 result no longer contains the approximately 0.9-second rank-local
stalls seen before gradient-DTensor wrapper reuse. The memory result confirms
that wrapper caching does not retain the released flat gradient storage.

### CUDA graph validation

CUDA graph capture succeeds for all 60 transformer blocks. `TracePoolAllocator`
plans six slots containing 3,318.4 MB of elements, and the run completes all 20
measured iterations without a slot collision. Capture increases peak allocated
memory from 74.67 GB in eager v2 to 86.72 GB.

The `--cuda-graph` option selects `TracePoolAllocator` internally. The explicit
`--trace-pool` used for this measurement documents that allocator choice but
does not select a different execution path.

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
