<!--
   Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
   NVIDIA CORPORATION and its licensors retain all intellectual property
   and proprietary rights in and to this software, related documentation
   and any modifications thereto. Any use, reproduction, disclosure or
   distribution of this software and related documentation without an express
   license agreement from NVIDIA CORPORATION is strictly prohibited.
-->

# Distributed Data Parallelism Guide

This guide helps you choose between different data parallelism strategies available in Megatron Core: **DDP**, **Megatron-FSDP**, and **Torch-FSDP2**.

## Quick Comparison

| Feature | DDP | Megatron-FSDP | Torch-FSDP2 |
|---------|-----|----------------|------------|
| **Memory Efficiency** | Medium | High | High |
| **Gradient Bucketing** | ✅ Yes | ✅ Yes | ⚠️ No |
| **Communication Overlap** | ✅ Yes | ✅ Yes | ✅ Yes |
| **Learning Curve** | Easy | Medium | Medium |
| **Compatibility** | All configs | All configs | Newer PyTorch |
| **Production Use** | ✅ Stable | ✅ Stable | ⚠️ Experimental |

## DDP (Distributed Data Parallel)

### When to Use
- **Multi-node training** where network bandwidth is limited
- **Small to medium models** (< 100B parameters)
- **Fast convergence needed** - DDP has minimal overhead
- **Stability is critical** - mature, well-tested implementation
- **Custom training loops** with fine-grained control over communication

### Key Features
- Replicate full model on each rank
- Aggregate gradients via all-reduce at end of backward pass
- Optional gradient accumulation in FP32 (fp32_grad_accumulation)
- Bucket-based gradient reduction for communication overlap
- Seamless integration with tensor/pipeline parallelism

### Basic Setup

```python
from megatron.core.distributed import DistributedDataParallel
from megatron.core.distributed import DistributedDataParallelConfig

config = TransformerConfig(
    tensor_model_parallel_size=1,
    pipeline_model_parallel_size=1,
)

ddp_config = DistributedDataParallelConfig(
    use_distributed_optimizer=False,  # Standard DDP (keep full optimizer state per rank)
    overlap_grad_reduce=True,         # Overlap gradient reduction with backprop
    bucket_size=40_000_000,           # Reduce in 40M param buckets
    grad_reduce_in_fp32=True,         # Accumulate gradients in FP32 even if model is BF16
)

model = YourTransformerModel(config=config)
model = DistributedDataParallel(config, ddp_config, model)
```

### Common Configurations

**Configuration A: Training Stability (Recommended for Most Cases)**
```python
DistributedDataParallelConfig(
    use_distributed_optimizer=False,
    overlap_grad_reduce=True,
    bucket_size=40_000_000,
    grad_reduce_in_fp32=True,
)
```

**Configuration B: Maximum Memory Efficiency + Speed**
```python
DistributedDataParallelConfig(
    use_distributed_optimizer=True,   # Keep only partial optimizer state per rank
    overlap_grad_reduce=True,
    bucket_size=100_000_000,          # Larger buckets for less communication overhead
    grad_reduce_in_fp32=True,
)
```

## Megatron-FSDP

### When to Use
- **Very large models** (> 100B parameters) where memory is the bottleneck
- **Multi-node with good connectivity** (low latency network)
- **Memory-limited scenarios** - parameters are sharded across ranks
- **Want to keep using distributed_optimizer** - natural fit with Megatron paradigms
- **Mixed precision training** (BF16/FP16) where memory savings matter most

### Key Features
- Shard parameter and optimizer states across data parallel group
- Reduced memory per rank (parameter memory: 1/dp_size)
- Automatic gradient checkpointing compatible
- Support for activation offloading
- dtensor checkpointing format for faster saves
- Full compatibility with tensor/pipeline parallelism

### Basic Setup

```python
from megatron.core.distributed import DistributedDataParallel
from megatron.core.distributed import DistributedDataParallelConfig

config = TransformerConfig(
    tensor_model_parallel_size=4,      # Use TP to avoid OOM
    pipeline_model_parallel_size=2,    # Use PP if needed
)

ddp_config = DistributedDataParallelConfig(
    use_distributed_optimizer=True,
    overlap_grad_reduce=True,
    bucket_size=40_000_000,
)

model = YourTransformerModel(config=config)

# Enable Megatron-FSDP
model = DistributedDataParallel(config, ddp_config, model)
```

Then in your training script:

```bash
torchrun --nproc_per_node=8 train.py \
    --use-megatron-fsdp \
    --ckpt-format fsdp_dtensor \
    --init-model-with-meta-device
```

### Memory Savings Calculation

For a 175B parameter model with Adam optimizer:
- **DDP**: 175B params × 4 bytes + optimizer state × 8 bytes = ~5.6 TB per rank
- **Megatron-FSDP** (8-way DP): 175B/8 × 4 + optimizer/8 × 8 = ~700 GB per rank

## Torch-FSDP2

### When to Use
- **PyTorch 2.4+** with latest features and bug fixes
- **Native PyTorch integration** preferred over custom implementations
- **Research experiments** with cutting-edge PyTorch APIs
- **Reduced custom code** maintenance burden
- **Automatic composability** with other PyTorch features

### Key Features
- Built-in handling of parameter sharding at module level
- Composable APIs that work with other torch.distributed features
- Native PyTorch maintenance and updates
- Automatic handling of forward/backward pass
- Support for `torch.distributed.checkpoint`

### Basic Setup

```python
from torch.distributed.experimental import enable_2d_fsdp
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

# Enable 2D FSDP if using tensor parallelism
enable_2d_fsdp()

model = YourTransformerModel(config=config)

# Wrap model gradually or recursively
model = FSDP(
    model,
    sharding_strategy=ShardingStrategy.FULL_SHARD,
)

# Training
for batch in dataloader:
    optimizer.zero_grad()
    loss = model(batch)
    loss.backward()
    optimizer.step()
```

Then in your training script:

```bash
torchrun --nproc_per_node=8 train.py \
    --use-torch-fsdp2 \
    --no-gradient-accumulation-fusion \
    --ckpt-format torch_dist
```

## Decision Tree

```
Start
  └─ Model size?
      ├─ < 100B params
      │   └─ Use DDP ✅ (simplest, lowest latency)
      │
      ├─ 100B - 500B params
      │   └─ Network fast & low latency?
      │       ├─ Yes → Megatron-FSDP or DDP
      │       └─ No → Megatron-FSDP (better scaling)
      │
      └─ > 500B params
          └─ Use Megatron-FSDP + TP + PP ✅

Considerations:
  - PyTorch version? → Use Torch-FSDP2 if >= 2.4
  - Stability critical? → Use Megatron-FSDP or DDP
  - Want latest features? → Use Torch-FSDP2
  - Custom training loop? → Use DDP
```

## Performance Tips

### Reduce Latency
1. **Increase bucket size** - fewer communication operations
2. **Enable overlap** - `overlap_grad_reduce=True`
3. **Use gradient accumulation** - batch multiple updates before communication
4. **Tune NCCL parameters** - `NCCL_BUFFSIZE`, `NCCL_MAX_NCHANNELS`

### Reduce Memory
1. **Use Megatron-FSDP** - directly reduces per-rank memory
2. **Enable gradient checkpointing** - trade compute for memory
3. **Reduce batch size** - per GPU, increase gradient accumulation steps instead
4. **Use lower precision** - BF16 instead of FP32

### Improve Throughput
1. **Enable gradient accumulation fusion** - batch multiple gradient reductions
2. **Use distributed optimizer** - reduce optimizer state memory
3. **Increase communication-compute overlap window** - larger batches

## Troubleshooting

### "RuntimeError: NCCL operation failed"
- **Cause**: Network issue or mismatch in process group
- **Solution**: Check network connectivity, verify process group setup, use `--max-jobs=4` if memory limited

### "OOM: out of memory"
- **DDP**: Use Megatron-FSDP instead for parameter sharding
- **Both**: Reduce batch size, enable gradient checkpointing, use TP

### "All-reduce taking too long"
- **Fix**: Increase `bucket_size`, check network bandwidth, reduce data parallel size

## References
- [Parallelism Guide](./parallelism-guide.md)
- [Megatron-LM GitHub](https://github.com/NVIDIA/Megatron-LM)
- [Reducing Activation Recomputation in Large Transformer Models](https://arxiv.org/abs/2205.05198)
- [Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism](https://arxiv.org/abs/1909.08053)
