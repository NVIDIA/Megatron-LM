# Refit Benchmark Suite

Complete standalone benchmarking suite for measuring refit performance in Megatron-RL.

## What's Included

- **`benchmark_refit.py`** - Core benchmark script (uses Megatron infrastructure)
- **`benchmark_refit_sbatch.sh`** - SLURM launcher (works in both batch and interactive mode)
- **`launch_benchmark.sh`** - Convenience wrapper with pre-configured scenarios
- **`run_refit_benchmarks.sh`** - Alternative launcher with numbered scenarios
- **`BENCHMARK_REFIT.md`** - Detailed documentation
- **`BENCHMARK_EXAMPLES.md`** - Comprehensive examples and use cases

## Quick Start

### Option 1: Interactive Mode (Recommended for Testing)

```bash
# 1. Request interactive allocation
salloc --nodes=1 --gpus-per-node=2 --partition=interactive --time=00:30:00

# 2. Run benchmark
cd /path/to/megatron-rl/examples/rl
./benchmark_refit_sbatch.sh \
    --tensor-model-parallel-size 2 \
    --rl-inference-tensor-model-parallel-size 1 \
    --refit-mode collocated \
    --refit-method nccl \
    --num-layers 4 --hidden-size 1024 --num-attention-heads 8 \
    --seq-length 512 --max-position-embeddings 512 --micro-batch-size 1
```

### Option 2: Batch Mode (Production Benchmarking)

```bash
cd /path/to/megatron-rl/examples/rl
sbatch --nodes=1 --gpus-per-node=2 benchmark_refit_sbatch.sh \
    --tensor-model-parallel-size 2 \
    --rl-inference-tensor-model-parallel-size 1 \
    --refit-mode collocated \
    --refit-method nccl \
    --num-layers 4 --hidden-size 1024 --num-attention-heads 8 \
    --seq-length 512 --max-position-embeddings 512 --micro-batch-size 1
```

### Option 3: Pre-Configured Scenarios

```bash
# Show available scenarios
./launch_benchmark.sh

# Run a specific scenario (interactive)
./launch_benchmark.sh tp2-tp1 interactive

# Submit batch job with NVSHMEM backend
./launch_benchmark.sh large batch nvshmem
```

## Common Scenarios

| Scenario | GPUs | Command |
|----------|------|---------|
| TP2→TP1 (collocated) | 2 | `./launch_benchmark.sh tp2-tp1 interactive` |
| TP2→TP4 (collocated) | 4 | `./launch_benchmark.sh tp2-tp4 batch` |
| TP2,PP2→TP4,PP1 | 4 | `./launch_benchmark.sh tp2pp2-tp4pp1 batch` |
| EP2→EP4 MoE | 4 | `./launch_benchmark.sh ep2-ep4 batch` |
| Large model | 4 | `./launch_benchmark.sh large batch nvshmem` |

## Key Features

### 🔄 Modes
- **Collocated**: Training and inference models share GPUs
- **Non-collocated**: Models use separate GPU sets

### ⚡ Backends
- **NCCL**: Recommended for most cases
- **NVSHMEM**: Optimized GPU communication (may be faster)
- **Gloo**: CPU-based (debugging only)

### 📊 Measurements
- Mean/min/max refit time across iterations
- Model size reporting
- Per-iteration timing
- Comprehensive logging

## File Guide

- **`BENCHMARK_REFIT.md`** - Start here for detailed documentation
- **`BENCHMARK_EXAMPLES.md`** - Browse real-world examples and use cases
- **`benchmark_refit.py`** - Main Python script (can be used directly)
- **`benchmark_refit_sbatch.sh`** - SLURM wrapper (auto-detects interactive/batch)
- **`launch_benchmark.sh`** - Quick launcher for common scenarios
- **`run_refit_benchmarks.sh`** - Alternative launcher (numbered menu)

## Workflow

1. **Test interactively** with small config
2. **Verify results** look reasonable
3. **Scale up** to production config
4. **Submit batch job** for final benchmarking
5. **Analyze results** from logs

## Example Workflow

```bash
# Step 1: Test small config interactively
salloc --nodes=1 --gpus-per-node=2 --partition=interactive --time=00:30:00
./benchmark_refit_sbatch.sh \
    --tensor-model-parallel-size 2 --rl-inference-tensor-model-parallel-size 1 \
    --refit-mode collocated --refit-method nccl \
    --num-layers 2 --hidden-size 512 --num-attention-heads 4 \
    --seq-length 256 --max-position-embeddings 256 --micro-batch-size 1 \
    --num-benchmark-iterations 3

# Step 2: Verify results look good
# Check that mean refit time is reported and reasonable

# Step 3: Scale up and submit batch job
sbatch --nodes=1 --gpus-per-node=2 benchmark_refit_sbatch.sh \
    --tensor-model-parallel-size 2 --rl-inference-tensor-model-parallel-size 1 \
    --refit-mode collocated --refit-method nccl \
    --num-layers 4 --hidden-size 1024 --num-attention-heads 8 \
    --seq-length 512 --max-position-embeddings 512 --micro-batch-size 1 \
    --num-benchmark-iterations 20

# Step 4: Check results
cd benchmark_results/logs
tail -50 <latest_log_file>
grep "Mean refit time" *.log
```

## Output Location

Results are saved in `benchmark_results/logs/` by default:
```bash
cd examples/rl/benchmark_results/logs
ls -lt  # List logs by time
tail -50 <log_file>  # View results
```

## Multi-Node Usage

```bash
# 2 nodes, 8 GPUs each (16 total)
sbatch --nodes=2 --ntasks-per-node=8 --gpus-per-node=8 benchmark_refit_sbatch.sh \
    --tensor-model-parallel-size 4 --pipeline-model-parallel-size 2 \
    --rl-inference-tensor-model-parallel-size 8 --rl-inference-pipeline-model-parallel-size 1 \
    --refit-mode collocated \
    --num-layers 16 --hidden-size 4096 --num-attention-heads 32 \
    --seq-length 2048 --max-position-embeddings 2048 --micro-batch-size 1
```

## Troubleshooting

### Common Issues

| Issue | Solution |
|-------|----------|
| Missing arguments | Check `BENCHMARK_REFIT.md` for required args |
| Out of memory | Reduce `--num-layers` and `--hidden-size` |
| Wrong GPU count | Ensure GPUs match parallelism requirements |
| Container not found | Set `CONTAINER_IMAGE` environment variable |

### Get Help

```bash
# Show all available arguments
python benchmark_refit.py --help

# Show launcher scenarios
./launch_benchmark.sh --help

# Read detailed docs
cat BENCHMARK_REFIT.md
cat BENCHMARK_EXAMPLES.md
```

## Advanced Options

### Custom Container
```bash
export CONTAINER_IMAGE="/path/to/container.sqsh"
sbatch --nodes=1 --gpus-per-node=2 benchmark_refit_sbatch.sh [args...]
```

### Custom Output Directory
```bash
export OUTPUT_DIR="/path/to/results"
sbatch --nodes=1 --gpus-per-node=2 benchmark_refit_sbatch.sh [args...]
```

### Extended Benchmarking
```bash
sbatch --nodes=1 --gpus-per-node=4 benchmark_refit_sbatch.sh \
    --num-benchmark-warmup 10 \
    --num-benchmark-iterations 100 \
    [other args...]
```

## Integration with RL Training

These benchmarks measure standalone refit performance. In actual RL training, refit happens between training and inference phases. Use these benchmarks to:

- Choose optimal parallelism configurations
- Select the best refit backend
- Estimate refit overhead in training loop
- Compare collocated vs non-collocated modes

## Next Steps

1. Read `BENCHMARK_REFIT.md` for detailed documentation
2. Browse `BENCHMARK_EXAMPLES.md` for more examples
3. Run a test benchmark interactively
4. Submit production benchmarks with batch mode
5. Analyze results and tune configurations
