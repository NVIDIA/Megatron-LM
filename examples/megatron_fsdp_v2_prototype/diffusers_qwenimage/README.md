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

## Benchmarks

`QwenImageTransformer2DModel`, `bs=4`, `512×512`, `bf16`, `torch.compile`, FA2.
`[mfsdpv2+cg]` uses `--cuda-graph --trace-pool`.

| Backend | 8×H100 | 4×GB200 |
|---------|--------|---------|
| **fsdp1** | 729 ms / 60.2 GB | 679 ms / 75.4 GB |
| **mfsdpv2** | 769 ms / 59.3 GB | 647 ms / 74.7 GB |
| **mfsdpv2+cg** | **674 ms** / 68.3 GB | **364 ms** / 88.7 GB |

CG delivers **11% faster** on H100 and **44% faster** on GB200 at the cost
of higher peak memory (pool-backed graph buffers).

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
