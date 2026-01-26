# Valid Refit Configurations for 8-GPU Nodes

This guide shows various refit configurations that work on a single 8-GPU node, useful for testing before scaling to multi-node Kimi K2 benchmarks.

## ⚠️ MoE Requirement

All MoE configurations **must** include:
```bash
--expert-tensor-parallel-size 1
--rl-inference-expert-tensor-model-parallel-size 1
--disable-bias-linear
```

See `MOE_ETP_REQUIREMENTS.md` for details. All commands below include these flags.

## Understanding GPU Requirements

For **non-collocated mode** (separate GPU sets):
```
Total GPUs needed = (Source TP × Source EP) + (Target TP × Target EP)
```

For **collocated mode** (shared GPU set):
```
Total GPUs needed = max(Source TP × Source EP, Target TP × Target EP)
```

## Recommended Configurations for 8-GPU Node

### Non-Collocated Configurations (NVSHMEM)

| Config | Source | Target | Total GPUs | Idle GPUs | Use Case |
|--------|--------|--------|------------|-----------|----------|
| **1** | TP=2, EP=2 (4) | TP=2, EP=1 (2) | 6 | 2 | Test EP refit ✓ |
| **2** | TP=2, EP=2 (4) | TP=2, EP=2 (4) | 8 | 0 | Full utilization ✓ |
| **3** | TP=4, EP=1 (4) | TP=2, EP=1 (2) | 6 | 2 | Test TP refit ✓ |
| **4** | TP=2, EP=1 (2) | TP=1, EP=1 (1) | 3 | 5 | Minimal test ✓ |
| **5** | TP=4, EP=1 (4) | TP=4, EP=1 (4) | 8 | 0 | TP-only, full use ✓ |
| **6** | TP=1, EP=4 (4) | TP=1, EP=2 (2) | 6 | 2 | EP-only refit ✓ |

### Collocated Configurations (NCCL)

| Config | Source | Target | Total GPUs | Use Case |
|--------|--------|--------|------------|----------|
| **1** | TP=4, EP=2 (8) | TP=2, EP=4 (8) | 8 | TP↔EP swap ✓ |
| **2** | TP=8, EP=1 (8) | TP=4, EP=2 (8) | 8 | High TP → balanced ✓ |
| **3** | TP=4, EP=2 (8) | TP=4, EP=1 (4) | 8 | Scale down ✓ |
| **4** | TP=2, EP=2 (4) | TP=2, EP=1 (2) | 4 | Small models ✓ |

## Example Commands

### Config 1: Test EP Refit (6 GPUs, 2 idle)

```bash
./benchmark_refit_sbatch.sh \
    --tensor-model-parallel-size 2 \
    --expert-model-parallel-size 2 \
    --expert-tensor-parallel-size 1 \
    --rl-inference-tensor-model-parallel-size 2 \
    --rl-inference-expert-model-parallel-size 1 \
    --rl-inference-expert-tensor-model-parallel-size 1 \
    --refit-mode non-collocated \
    --refit-method nvshmem \
    --num-layers 4 \
    --hidden-size 1024 \
    --num-attention-heads 8 \
    --seq-length 512 \
    --max-position-embeddings 512 \
    --micro-batch-size 1 \
    --num-experts 16 \
    --moe-router-topk 2 \
    --moe-shared-expert-intermediate-size 512 \
    --ffn-hidden-size 2688 \
    --disable-bias-linear \
    --num-benchmark-iterations 3
```

**What it tests:** Expert Parallelism resharding from EP=2 to EP=1 while keeping TP=2 constant

**Note:** ETP=1 required for MoE, bias disabled for compatibility

### Config 2: Full 8-GPU Utilization (0 idle)

```bash
./benchmark_refit_sbatch.sh \
    --tensor-model-parallel-size 2 \
    --expert-model-parallel-size 2 \
    --rl-inference-tensor-model-parallel-size 2 \
    --rl-inference-expert-model-parallel-size 2 \
    --refit-mode non-collocated \
    --refit-method nvshmem \
    --num-layers 4 \
    --hidden-size 1024 \
    --num-attention-heads 8 \
    --seq-length 512 \
    --max-position-embeddings 512 \
    --micro-batch-size 1 \
    --num-experts 16 \
    --moe-router-topk 2 \
    --moe-shared-expert-intermediate-size 512 \
    --ffn-hidden-size 2688 \
    --num-benchmark-iterations 3
```

**What it tests:** Both models use 4 GPUs each (TP=2, EP=2), tests model weight transfer between separate GPU sets

### Config 3: Test TP Refit (6 GPUs, 2 idle)

```bash
./benchmark_refit_sbatch.sh \
    --tensor-model-parallel-size 4 \
    --expert-model-parallel-size 1 \
    --rl-inference-tensor-model-parallel-size 2 \
    --rl-inference-expert-model-parallel-size 1 \
    --refit-mode non-collocated \
    --refit-method nvshmem \
    --num-layers 4 \
    --hidden-size 1024 \
    --num-attention-heads 8 \
    --seq-length 512 \
    --max-position-embeddings 512 \
    --micro-batch-size 1 \
    --num-experts 16 \
    --moe-router-topk 2 \
    --moe-shared-expert-intermediate-size 512 \
    --ffn-hidden-size 2688 \
    --num-benchmark-iterations 3
```

**What it tests:** Tensor Parallelism resharding from TP=4 to TP=2 while keeping EP=1 constant

### Config 4: Minimal Test (3 GPUs, 5 idle)

```bash
./benchmark_refit_sbatch.sh \
    --tensor-model-parallel-size 2 \
    --expert-model-parallel-size 1 \
    --rl-inference-tensor-model-parallel-size 1 \
    --rl-inference-expert-model-parallel-size 1 \
    --refit-mode non-collocated \
    --refit-method nvshmem \
    --num-layers 2 \
    --hidden-size 512 \
    --num-attention-heads 4 \
    --seq-length 256 \
    --max-position-embeddings 256 \
    --micro-batch-size 1 \
    --num-experts 8 \
    --moe-router-topk 2 \
    --moe-shared-expert-intermediate-size 256 \
    --ffn-hidden-size 1344 \
    --num-benchmark-iterations 2
```

**What it tests:** Fastest validation that NVSHMEM works (takes ~30 seconds)

### Collocated Config: TP↔EP Swap (8 GPUs)

```bash
./benchmark_refit_sbatch.sh \
    --tensor-model-parallel-size 4 \
    --expert-model-parallel-size 2 \
    --rl-inference-tensor-model-parallel-size 2 \
    --rl-inference-expert-model-parallel-size 4 \
    --refit-mode collocated \
    --refit-method nccl \
    --num-layers 4 \
    --hidden-size 1024 \
    --num-attention-heads 8 \
    --seq-length 512 \
    --max-position-embeddings 512 \
    --micro-batch-size 1 \
    --num-experts 16 \
    --moe-router-topk 2 \
    --moe-shared-expert-intermediate-size 512 \
    --ffn-hidden-size 2688 \
    --num-benchmark-iterations 3
```

**What it tests:** Swapping from TP-heavy (TP=4, EP=2) to EP-heavy (TP=2, EP=4) on same 8 GPUs

## Understanding Output

### With Idle Ranks

```
World size: 8 (using 6 GPUs)
Source ranks: 0-3 (TP=2, PP=1, EP=2)
Destination ranks: 4-5 (TP=2, PP=1, EP=1)
Idle ranks: 6-7 (not participating)
```

Ranks 6-7 will wait but not build models or participate in refit.

### Full Utilization

```
World size: 8 (using 8 GPUs)
Source ranks: 0-3 (TP=2, PP=1, EP=2)
Destination ranks: 4-7 (TP=2, PP=1, EP=2)
```

All 8 ranks participate in the benchmark.

## Configuration Guidelines

### For Testing Specific Behaviors

| Want to Test | Recommended Config |
|--------------|-------------------|
| **EP refit only** | Keep TP same, change EP (Config 1 or 6) |
| **TP refit only** | Keep EP same, change TP (Config 3 or 5) |
| **Both TP & EP** | Change both dimensions |
| **NVSHMEM works** | Config 4 (minimal, fastest) |
| **Full GPU usage** | Config 2 or 5 (non-collocated) |
| **NCCL comparison** | Any collocated config |

### Scaling to Multi-Node

After validating on 8 GPUs, scale to multi-node for Kimi K2:

```bash
# Single node test (8 GPUs)
salloc --nodes=1 --ntasks-per-node=8 --gpus-per-node=8
./test_nvshmem_refit.sh

# Multi-node Kimi K2 (128 GPUs)
sbatch benchmark_refit_kimi_k2.sh
```

## Common Pitfalls

### ❌ Too Many GPUs Requested

```bash
# This fails: needs 12 GPUs but only 8 available
--tensor-model-parallel-size 8 --expert-model-parallel-size 1 \
--rl-inference-tensor-model-parallel-size 4 --rl-inference-expert-model-parallel-size 1 \
--refit-mode non-collocated
# 8 + 4 = 12 GPUs needed > 8 available
```

### ✓ Fixed Version

```bash
# This works: needs 6 GPUs, 2 will be idle
--tensor-model-parallel-size 4 --expert-model-parallel-size 1 \
--rl-inference-tensor-model-parallel-size 2 --rl-inference-expert-model-parallel-size 1 \
--refit-mode non-collocated
# 4 + 2 = 6 GPUs needed ≤ 8 available ✓
```

## Quick Selection Table

| Your Goal | Config to Use | Command |
|-----------|---------------|---------|
| Quick NVSHMEM test | Config 4 (minimal) | `./test_nvshmem_refit.sh` then edit |
| Test EP refit | Config 1 | See "Config 1" command above |
| Test TP refit | Config 3 | See "Config 3" command above |
| Use all 8 GPUs | Config 2 or 5 | See "Config 2" command above |
| Compare NCCL | Collocated configs | See collocated examples |
| Before Kimi K2 run | Config 1 or 2 | Validates infrastructure |

## Next Steps

1. **Start with minimal test** (Config 4) to validate NVSHMEM works
2. **Try EP refit** (Config 1) to test expert resharding
3. **Try TP refit** (Config 3) to test tensor resharding
4. **Scale to multi-node** with `sbatch benchmark_refit_kimi_k2.sh`

## Files

- `test_nvshmem_refit.sh` - Uses Config 1 by default
- `NVSHMEM_QUICK_TEST.md` - More examples and troubleshooting
- `benchmark_refit_kimi_k2.sh` - Full multi-node benchmark
- `8GPU_CONFIGS.md` - This file
