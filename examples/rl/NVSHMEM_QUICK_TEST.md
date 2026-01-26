# NVSHMEM Quick Test Guide

Quick commands to test NVSHMEM-based refit with a small model before running the full Kimi K2 benchmark.

## ⚠️ Important: MoE Configuration Requirements

When using MoE (Mixture of Experts) models, you **must** set:
- `--expert-tensor-parallel-size 1` for source model
- `--rl-inference-expert-tensor-model-parallel-size 1` for target model
- `--disable-bias-linear` to disable bias (required when ETP > 1 with TP > 1)

These are already included in all commands below.

## Option 1: Automated Test Script (Recommended)

```bash
cd /lustre/fsw/portfolios/adlr/projects/adlr_psx_fp8/users/wdykas/code/ep-refit/mrl_internal/megatron-rl/examples/rl

# Get interactive allocation (1 node, 30 minutes)
salloc --nodes=1 --ntasks-per-node=8 --gpus-per-node=8 --time=30:00 --partition=interactive

# Run automated test
./test_nvshmem_refit.sh
```

**What it tests:**
- Small model: 4 layers, 1024 hidden, 16 experts
- Non-collocated mode with NVSHMEM
- Source: TP=2, EP=2 (4 GPUs) → Target: TP=2, EP=1 (2 GPUs)
- Takes ~1-2 minutes

## Option 2: Manual One-Liner Command

If you prefer manual control or want to customize parameters:

```bash
# 1. Get interactive allocation
salloc --nodes=1 --ntasks-per-node=8 --gpus-per-node=8 --time=30:00 --partition=interactive

# 2. Navigate to directory
cd /lustre/fsw/portfolios/adlr/projects/adlr_psx_fp8/users/wdykas/code/ep-refit/mrl_internal/megatron-rl/examples/rl

# 3. Run benchmark with small model
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
    --num-benchmark-warmup 2 \
    --num-benchmark-iterations 3
```

## Option 3: Use All 8 GPUs (No Idle Ranks)

If you want to use all 8 GPUs without idle ranks:

```bash
salloc --nodes=1 --ntasks-per-node=8 --gpus-per-node=8 --time=30:00 --partition=interactive

cd /lustre/fsw/portfolios/adlr/projects/adlr_psx_fp8/users/wdykas/code/ep-refit/mrl_internal/megatron-rl/examples/rl

./benchmark_refit_sbatch.sh \
    --tensor-model-parallel-size 2 \
    --expert-model-parallel-size 2 \
    --expert-tensor-parallel-size 1 \
    --rl-inference-tensor-model-parallel-size 2 \
    --rl-inference-expert-model-parallel-size 2 \
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
    --num-benchmark-warmup 2 \
    --num-benchmark-iterations 3
```

**Configuration:**
- Source: TP=2, EP=2 (4 GPUs) → Target: TP=2, EP=2 (4 GPUs)
- Uses exactly 8 GPUs, no idle ranks
- Tests EP resharding only (both use same TP/EP but separate GPU sets)
- **Note:** ETP=1 required for MoE, bias disabled

## Option 4: Minimal Model (Fastest Test)

Absolute minimal configuration for quick NVSHMEM validation:

```bash
salloc --nodes=1 --ntasks-per-node=8 --gpus-per-node=8 --time=15:00 --partition=interactive

cd /lustre/fsw/portfolios/adlr/projects/adlr_psx_fp8/users/wdykas/code/ep-refit/mrl_internal/megatron-rl/examples/rl

./benchmark_refit_sbatch.sh \
    --tensor-model-parallel-size 2 \
    --expert-model-parallel-size 1 \
    --expert-tensor-parallel-size 1 \
    --rl-inference-tensor-model-parallel-size 1 \
    --rl-inference-expert-model-parallel-size 1 \
    --rl-inference-expert-tensor-model-parallel-size 1 \
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
    --disable-bias-linear \
    --num-benchmark-warmup 1 \
    --num-benchmark-iterations 2
```

**Configuration:**
- 2 layers, 512 hidden, 8 experts
- Source: TP=2, EP=1 (2 GPUs) → Target: TP=1, EP=1 (1 GPU)
- Takes ~30 seconds

## Test for Collocated Mode (NCCL comparison)

To compare NVSHMEM vs NCCL on the same hardware:

```bash
salloc --nodes=1 --ntasks-per-node=8 --gpus-per-node=8 --time=30:00 --partition=interactive

cd /lustre/fsw/portfolios/adlr/projects/adlr_psx_fp8/users/wdykas/code/ep-refit/mrl_internal/megatron-rl/examples/rl

# Test with NCCL (collocated)
./benchmark_refit_sbatch.sh \
    --tensor-model-parallel-size 2 \
    --expert-model-parallel-size 2 \
    --expert-tensor-parallel-size 1 \
    --rl-inference-tensor-model-parallel-size 2 \
    --rl-inference-expert-model-parallel-size 1 \
    --rl-inference-expert-tensor-model-parallel-size 1 \
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
    --disable-bias-linear \
    --num-benchmark-warmup 2 \
    --num-benchmark-iterations 3
```

## Expected Output

Successful run shows:

```
================================================================================
COLLOCATED MODE REFIT BENCHMARK  (or NON-COLLOCATED MODE)
================================================================================
World size: 8
Source (training): TP=2, PP=1, EP=2, DP=2
Destination (inference): TP=2, PP=1, EP=1, DP=4
Model: 4 layers, 1024 hidden, 8 heads
Refit backend: nvshmem
================================================================================

Building models...
building GPT model ...
Source model size on rank 0: XX.XX MB
Destination model size on rank 0: XX.XX MB

Warmup: 2 iterations...
  (First iteration builds refit plan, subsequent iterations reuse cached plan)
  Warmup iteration 1/2 complete
  Warmup iteration 2/2 complete
  Plan building complete, now benchmarking execution only...

Benchmarking: 3 iterations...
  Iteration 1/3: X.XX ms
  Iteration 2/3: X.XX ms
  Iteration 3/3: X.XX ms

================================================================================
RESULTS
================================================================================
Mean refit time: X.XX ms
Min refit time:  X.XX ms
Max refit time:  X.XX ms
================================================================================

Benchmark completed successfully!
```

## Troubleshooting

### NVSHMEM Not Found

If you get an error about NVSHMEM not being available:

```bash
# Check if NVSHMEM Python bindings are installed
python -c "import nvshmem" 2>/dev/null && echo "✓ NVSHMEM installed" || echo "✗ NVSHMEM missing"

# If missing, you may need to:
# 1. Use a different container image with NVSHMEM support
# 2. Or fall back to NCCL with collocated mode
```

### Container Issues

If running outside container, make sure NVSHMEM libraries are available:

```bash
# Check NVSHMEM library
ldconfig -p | grep nvshmem

# May need to load modules
module load nvshmem
```

### GPU Allocation Issues

If you get "not enough GPUs" errors:

```bash
# Check available GPUs in your allocation
echo "GPUs: ${SLURM_GPUS_ON_NODE}"
nvidia-smi

# Make sure: src_gpus + dst_gpus ≤ total_gpus
# For non-collocated: TP_src × EP_src + TP_dst × EP_dst ≤ 8
```

### Switch to NCCL/Collocated for Testing

If NVSHMEM issues persist, test with NCCL first:

```bash
# Change these two parameters:
--refit-mode collocated \      # instead of non-collocated
--refit-method nccl \           # instead of nvshmem
```

## Next Steps

After successful test:

1. **✓ NVSHMEM works** → Run full Kimi K2 benchmark:
   ```bash
   sbatch benchmark_refit_kimi_k2.sh
   ```

2. **✗ NVSHMEM fails** → Debug or use NCCL:
   - Check container/module configuration
   - Test with NCCL in collocated mode
   - Contact system administrator about NVSHMEM support

3. **Compare performance** → Run both NVSHMEM and NCCL tests to compare

## Parameter Reference

Quick reference for customizing test configurations:

| Parameter | Small Test | Full 8-GPU | Minimal Test | Large (Kimi K2) |
|-----------|------------|------------|--------------|-----------------|
| `--num-layers` | 4 | 4 | 2 | 61 |
| `--hidden-size` | 1024 | 1024 | 512 | 7168 |
| `--num-attention-heads` | 8 | 8 | 4 | 64 |
| `--num-experts` | 16 | 16 | 8 | 384 |
| `--seq-length` | 512 | 512 | 256 | 4096-8192 |
| Source TP×EP | 2×2 (4 GPUs) | 2×2 (4 GPUs) | 2×1 (2 GPUs) | 8-32 × 4-8 |
| Target TP×EP | 2×1 (2 GPUs) | 2×2 (4 GPUs) | 1×1 (1 GPU) | 8-32 × 2-8 |
| Total GPUs used | 6 (2 idle) | 8 (0 idle) | 3 (5 idle) | 96-128 |
| Expected time | 1-2 min | 1-2 min | 30 sec | 5-10 min/config |

## Files

- `test_nvshmem_refit.sh` - Automated test script
- `benchmark_refit_sbatch.sh` - Interactive benchmark wrapper
- `benchmark_refit.py` - Core benchmark implementation
- `NVSHMEM_QUICK_TEST.md` - This guide
