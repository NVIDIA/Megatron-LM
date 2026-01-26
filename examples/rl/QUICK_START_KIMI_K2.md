# Quick Start: Kimi K2 Refit Benchmark

## TL;DR

```bash
# Navigate to the benchmark directory
cd /lustre/fsw/portfolios/adlr/projects/adlr_psx_fp8/users/wdykas/code/ep-refit/mrl_internal/megatron-rl/examples/rl

# Submit the benchmark job (requires 16 nodes, 128 GPUs)
sbatch benchmark_refit_kimi_k2.sh

# Check job status
squeue -u $USER

# View results when complete
cat benchmark_results/kimi_k2/logs/*_summary.txt
```

## What This Benchmarks

The script tests **refit (weight resharding) performance** on the **Kimi K2** model, which is a:
- **1.04 trillion parameter** Mixture-of-Experts model
- **384 experts**, 8 active per token
- **61 layers**, hidden size 7168
- Developed by Moonshot AI

## Configurations Tested

By default, the script tests these refit scenarios:

| Source TP | Target TP | Mode | Method | Description |
|-----------|-----------|------|--------|-------------|
| 8 | 4 | collocated | nccl | Training model with TP=8 refits to inference model with TP=4 |
| 8 | 2 | collocated | nccl | Training model with TP=8 refits to inference model with TP=2 |
| 8 | 1 | collocated | nccl | Training model with TP=8 refits to inference model with TP=1 |
| 4 | 2 | collocated | nccl | Training model with TP=4 refits to inference model with TP=2 |
| 4 | 1 | collocated | nccl | Training model with TP=4 refits to inference model with TP=1 |

**Collocated mode** means both source and target models share the same GPUs.

## Resource Requirements

- **Default:** 16 nodes × 8 GPUs = **128 GPUs**
- **Time:** 2 hours
- **Partition:** batch
- **Account:** llmservice_fm_text

## Customizing the Benchmark

### 1. Use Fewer Nodes

Edit `benchmark_refit_kimi_k2.sh` and change:

```bash
#SBATCH --nodes=8  # Use 8 nodes instead of 16
```

Then adjust the TP configurations to fit available GPUs:

```bash
CONFIGS=(
    "4:2:collocated:nccl"
    "4:1:collocated:nccl"
    "2:1:collocated:nccl"
)
```

### 2. Test Specific Configuration

Run interactively on allocated nodes:

```bash
# Get interactive allocation (example: 2 nodes)
salloc --nodes=2 --ntasks-per-node=8 --gpus-per-node=8 --time=1:00:00 --partition=interactive

# Run single benchmark
./benchmark_refit_sbatch.sh \
    --tensor-model-parallel-size 4 \
    --rl-inference-tensor-model-parallel-size 2 \
    --refit-mode collocated \
    --refit-method nccl \
    --num-layers 61 \
    --hidden-size 7168 \
    --num-attention-heads 64 \
    --seq-length 4096 \
    --max-position-embeddings 131072 \
    --micro-batch-size 1 \
    --num-experts 384 \
    --moe-router-topk 8 \
    --moe-shared-expert-intermediate-size 2048 \
    --ffn-hidden-size 18432 \
    --num-benchmark-iterations 5
```

### 3. Reduce Sequence Length for Faster Testing

Edit the script:

```bash
SEQ_LENGTH=2048  # Smaller sequence length for faster tests
```

## Understanding the Output

### Summary File

After completion, check: `benchmark_results/kimi_k2/logs/<RUN_ID>_summary.txt`

Example:

```
Kimi K2 Refit Benchmark Summary
================================
Job ID: 9876543
Nodes: 16
Total GPUs: 128

Results:
--------

Config: TP 8 -> TP 4 (collocated, nccl)
Mean refit time: 45.23 ms  ← Average time to reshard weights
Min refit time:  44.87 ms  ← Fastest iteration
Max refit time:  45.61 ms  ← Slowest iteration

Config: TP 8 -> TP 2 (collocated, nccl)
Mean refit time: 52.14 ms
...
```

### What's Being Measured

- **Only execution time** (not plan building)
- Plan is built once during warmup and cached
- Measurements exclude synchronization overhead
- Times show how long it takes to transfer/reshard weights between models

### Individual Logs

Detailed logs per configuration: `benchmark_results/kimi_k2/logs/<RUN_ID>_tp<SRC>_to_tp<DST>_*.log`

## Troubleshooting

### Out of Memory

- Reduce `SEQ_LENGTH` in the script
- Use higher TP values to distribute model across more GPUs
- Reduce `NUM_LAYERS` for testing (won't match real Kimi K2 but can test refit logic)

### Job Fails to Start

- Check partition and account settings match your allocation
- Verify node availability: `sinfo -p batch`

### Slow Performance

- Ensure using NCCL for collocated mode (fastest)
- Check GPU interconnect (NVLink/NVSwitch)
- Verify no other jobs competing for GPU resources

## Next Steps

After benchmarking, you can:

1. **Compare TP configurations** to find optimal refit scenario
2. **Test non-collocated mode** (uncomment in script) for separate GPU sets
3. **Adjust model size** to match your actual deployment needs
4. **Integrate refit** into your training/inference pipeline

## Files Created

- `benchmark_refit_kimi_k2.sh` - Main sbatch script
- `BENCHMARK_KIMI_K2.md` - Detailed documentation
- `QUICK_START_KIMI_K2.md` - This quick start guide

## Questions?

See full documentation: `BENCHMARK_KIMI_K2.md`
