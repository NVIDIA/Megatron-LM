# Kimi K2 Refit Benchmark - Realistic High-Parallelism Testing

This benchmark tests refit (weight resharding) performance on the **Kimi K2** model (1.04T parameters) with realistic high-parallelism configurations that would actually be used in production.

## Why These Configurations Matter

For a 1T parameter MoE model on 80GB GPUs:
- **Simple TP-only** (e.g., TP=8→4) is not realistic at this scale
- **Need Expert Parallelism (EP)** to distribute 384 experts across GPUs
- **Need high TP** to split dense layers and attention across GPUs
- **Typical pattern**: High training parallelism (64-128 GPUs) → Lower inference parallelism (32-64 GPUs)

## Memory Math

```
Model Size: 1.04T params × 2 bytes (BF16) = 2.08 TB
Per GPU with TP=32: 1.04T / 32 = 32.5B params = 65GB ✓
Per GPU with EP=64: 384 experts / 64 = 6 experts per GPU

Realistic configs for 128 GPUs:
- TP=16, EP=8  → 16×8  = 128 GPUs
- TP=32, EP=4  → 32×4  = 128 GPUs
- TP=8,  EP=16 → 8×16  = 128 GPUs
```

## Benchmark Scenarios

### Non-Collocated Mode (NVSHMEM)

**Scenario 1: EP Refit** (64→32 GPUs)
```
Source:  TP=8,  EP=8  (64 GPUs for training)
Target:  TP=8,  EP=4  (32 GPUs for inference)
```
- Tests Expert Parallelism resharding (8→4 EP)
- TP stays constant, isolates EP refit behavior
- Training: 384/8 = 48 experts per group
- Inference: 384/4 = 96 experts per group

**Scenario 2: TP Refit** (64→32 GPUs)
```
Source:  TP=16, EP=4  (64 GPUs for training)
Target:  TP=8,  EP=4  (32 GPUs for inference)
```
- Tests Tensor Parallelism resharding (16→8 TP)
- EP stays constant, isolates TP refit behavior
- Training: Higher TP for better memory distribution
- Inference: Lower TP for better throughput

**Scenario 3: EP Refit with High TP** (64→32 GPUs)
```
Source:  TP=16, EP=4  (64 GPUs for training)
Target:  TP=16, EP=2  (32 GPUs for inference)
```
- Tests EP refit with same high TP
- Maintains fine-grained tensor splits
- Training: 384/4 = 96 experts per group
- Inference: 384/2 = 192 experts per group

**Scenario 4: TP Refit with EP Change** (64→64 GPUs)
```
Source:  TP=16, EP=4  (64 GPUs for training)
Target:  TP=32, EP=2  (64 GPUs for inference)
```
- Tests both TP and EP refit simultaneously
- Same GPU count, different parallelism strategy
- Doubles TP while halving EP

**Scenario 5: Both TP & EP Refit** (64→64 GPUs)
```
Source:  TP=8,  EP=8  (64 GPUs for training)
Target:  TP=16, EP=4  (64 GPUs for inference)
```
- Tests complex refit with both dimensions changing
- From balanced TP/EP to TP-heavy configuration

### Collocated Mode (NCCL) - For Comparison

**Scenario 6: Full GPU Set TP/EP Swap** (128 GPUs)
```
Source:  TP=16, EP=8  (128 GPUs, TP-heavy)
Target:  TP=8,  EP=16 (128 GPUs, EP-heavy)
```
- Both models share all 128 GPUs
- Tests swapping between TP-heavy and EP-heavy
- Uses NCCL for fast on-node communication

**Scenario 7: TP Scaling** (128 GPUs)
```
Source:  TP=32, EP=4  (128 GPUs, very high TP)
Target:  TP=16, EP=8  (128 GPUs, balanced)
```
- Tests moving from extreme TP to balanced config
- Useful for understanding TP refit overhead

## Why NVSHMEM for Non-Collocated?

- **NVSHMEM**: Optimized for GPU-to-GPU communication across nodes
- **Non-collocated**: Models on separate GPU sets, requires inter-node transfers
- **NCCL**: Best for on-node (collocated) communication
- **NVSHMEM**: Better for cross-node communication patterns in non-collocated refit

## Usage

### Submit Full Benchmark Suite

```bash
cd /lustre/fsw/portfolios/adlr/projects/adlr_psx_fp8/users/wdykas/code/ep-refit/mrl_internal/megatron-rl/examples/rl

sbatch benchmark_refit_kimi_k2.sh
```

This will:
- Request 16 nodes (128 GPUs)
- Run 7 benchmark scenarios
- Generate summary with all results
- Take approximately 1-2 hours

### Run Single Configuration

For testing, run a single config on interactive allocation:

```bash
# Allocate nodes (e.g., 12 nodes = 96 GPUs for 64→32 scenario)
salloc --nodes=12 --ntasks-per-node=8 --gpus-per-node=8 --time=1:00:00 --partition=interactive

# Run non-collocated EP refit benchmark
./benchmark_refit_sbatch.sh \
    --tensor-model-parallel-size 8 \
    --expert-model-parallel-size 8 \
    --rl-inference-tensor-model-parallel-size 8 \
    --rl-inference-expert-model-parallel-size 4 \
    --refit-mode non-collocated \
    --refit-method nvshmem \
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

### Customize Configurations

Edit `benchmark_refit_kimi_k2.sh` and modify the `CONFIGS` array:

```bash
CONFIGS=(
    # Format: "SRC_TP:SRC_EP:DST_TP:DST_EP:REFIT_MODE:REFIT_METHOD:DESCRIPTION"
    "8:8:8:4:non-collocated:nvshmem:my_custom_test"
    # Add more configs...
)
```

## Expected Results

### Typical Refit Times (estimates)

Based on model size and communication patterns:

| Scenario | Mode | GPUs | Expected Time |
|----------|------|------|---------------|
| TP=8, EP=8→4 | non-collocated | 64→32 | 30-60ms |
| TP=16, EP=4→2 | non-collocated | 64→32 | 40-80ms |
| TP=16→8, EP=4 | non-collocated | 64→32 | 50-100ms |
| TP=16→32, EP=4→2 | non-collocated | 64→64 | 60-120ms |
| TP=16↔8, EP=8↔16 | collocated | 128 | 20-40ms |

*Note: Actual times depend on GPU interconnect, network topology, and system load*

### What Affects Performance

1. **TP Refit**: Bandwidth-bound, depends on inter-node network
2. **EP Refit**: Router/expert redistribution, can be compute-bound
3. **Non-collocated**: Requires data movement between GPU sets
4. **Collocated**: Can leverage shared memory, faster
5. **NVSHMEM**: Lower latency for cross-node GPU-direct RDMA

## Output

### Summary File

`benchmark_results/kimi_k2/logs/<RUN_ID>_summary.txt`:

```
Kimi K2 Refit Benchmark Summary
================================
Date: 2026-01-26 15:30:00
Job ID: 9876543
Nodes: 16
Total GPUs: 128

Results:
--------

Config: TP8_EP8_to_TP8_EP4_64to32
  Source: TP=8, EP=8 (64 GPUs)
  Target: TP=8, EP=4 (32 GPUs)
  Mode: non-collocated, Method: nvshmem
Mean refit time: 45.23 ms
Min refit time:  44.87 ms
Max refit time:  45.61 ms

Config: TP16_EP4_to_TP8_EP4_64to32
  Source: TP=16, EP=4 (64 GPUs)
  Target: TP=8, EP=4 (32 GPUs)
  Mode: non-collocated, Method: nvshmem
Mean refit time: 62.14 ms
Min refit time:  61.89 ms
Max refit time:  62.47 ms
...
```

### Individual Logs

`benchmark_results/kimi_k2/logs/<RUN_ID>_<DESCRIPTION>.log`:
- Full verbose output per configuration
- Includes warmup iterations
- Model building logs
- Detailed timing information

## Interpreting Results

### Key Metrics

1. **Mean refit time**: Average execution time across iterations
2. **Min/Max spread**: Indicates performance consistency
3. **GPU utilization**: Check if bandwidth-saturated
4. **Comparison across configs**: Identify optimal parallelism strategy

### Analysis Questions

- **EP vs TP**: Which refit type is faster for your workload?
- **Non-collocated overhead**: How much does separate GPU sets cost?
- **Scaling behavior**: Does refit time scale linearly with data size?
- **NVSHMEM vs NCCL**: Performance difference for your topology?

## Troubleshooting

### OOM (Out of Memory)

```bash
# Reduce sequence length
SEQ_LENGTH=2048

# Or increase parallelism
# E.g., use TP=32 instead of TP=16
```

### NVSHMEM Errors

```bash
# Check NVSHMEM is available in container
srun --nodes=1 --ntasks=1 bash -c "python -c 'import nvshmem'"

# May need to load NVSHMEM modules
module load nvshmem
```

### Non-Collocated Hangs

- Verify GPU count: src_gpus + dst_gpus ≤ total_gpus
- Check rank allocation matches expected layout
- Ensure NVSHMEM is properly initialized

### Slow Performance

- Check network topology: Are nodes on same switch?
- Verify GPU interconnect: NVLink/NVSwitch working?
- Monitor network: `nvidia-smi topo -m`
- Check for competing jobs: `squeue`

## Files

- `benchmark_refit_kimi_k2.sh` - Multi-node sbatch script
- `benchmark_refit.py` - Core benchmark implementation
- `benchmark_refit_sbatch.sh` - Interactive wrapper

## References

- [Kimi K2 Official](https://moonshotai.github.io/Kimi-K2/)
- [Kimi K2 HuggingFace](https://huggingface.co/moonshotai/Kimi-K2-Instruct)
- [Megatron-Core Refit Documentation](../../megatron/core/resharding/)
