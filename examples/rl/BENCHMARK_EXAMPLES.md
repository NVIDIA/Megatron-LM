# Refit Benchmark Examples

This guide provides practical examples for running the refit benchmark in various scenarios.

## Quick Start

### Interactive Single-Node

For quick debugging on a single node with 2 GPUs:

```bash
# Request an interactive allocation
salloc --nodes=1 --gpus-per-node=2 --partition=interactive --time=00:30:00

# Run benchmark directly
cd /lustre/fsw/portfolios/adlr/users/wdykas/code/ep-refit/mrl_internal/megatron-rl/examples/rl

./benchmark_refit_sbatch.sh \
    --tensor-model-parallel-size 2 \
    --rl-inference-tensor-model-parallel-size 1 \
    --refit-mode collocated \
    --refit-method nccl \
    --num-layers 4 \
    --hidden-size 1024 \
    --num-attention-heads 8 \
    --seq-length 512 \
    --max-position-embeddings 512 \
    --micro-batch-size 1 \
    --num-benchmark-iterations 10
```

### Batch Single-Node

For production benchmarking:

```bash
sbatch --nodes=1 --gpus-per-node=2 benchmark_refit_sbatch.sh \
    --tensor-model-parallel-size 2 \
    --rl-inference-tensor-model-parallel-size 1 \
    --refit-mode collocated \
    --refit-method nccl \
    --num-layers 4 \
    --hidden-size 1024 \
    --num-attention-heads 8 \
    --seq-length 512 \
    --max-position-embeddings 512 \
    --micro-batch-size 1 \
    --num-benchmark-iterations 20
```

## Common Scenarios

### 1. Basic TP Changes (Single Node)

#### TP2 → TP1 (Collocated, 2 GPUs)
```bash
# Interactive
salloc --nodes=1 --gpus-per-node=2 --partition=interactive --time=00:30:00
./benchmark_refit_sbatch.sh \
    --tensor-model-parallel-size 2 \
    --rl-inference-tensor-model-parallel-size 1 \
    --refit-mode collocated \
    --num-layers 4 --hidden-size 1024 --num-attention-heads 8 \
    --seq-length 512 --max-position-embeddings 512 --micro-batch-size 1

# Batch
sbatch --nodes=1 --gpus-per-node=2 benchmark_refit_sbatch.sh \
    --tensor-model-parallel-size 2 \
    --rl-inference-tensor-model-parallel-size 1 \
    --refit-mode collocated \
    --num-layers 4 --hidden-size 1024 --num-attention-heads 8 \
    --seq-length 512 --max-position-embeddings 512 --micro-batch-size 1
```

#### TP2 → TP4 (Collocated, 4 GPUs)
```bash
sbatch --nodes=1 --gpus-per-node=4 benchmark_refit_sbatch.sh \
    --tensor-model-parallel-size 2 \
    --rl-inference-tensor-model-parallel-size 4 \
    --refit-mode collocated \
    --num-layers 4 --hidden-size 1024 --num-attention-heads 8 \
    --seq-length 512 --max-position-embeddings 512 --micro-batch-size 1
```

#### TP2 → TP1 (Non-Collocated, 3 GPUs)
```bash
sbatch --nodes=1 --gpus-per-node=3 benchmark_refit_sbatch.sh \
    --tensor-model-parallel-size 2 \
    --rl-inference-tensor-model-parallel-size 1 \
    --refit-mode non-collocated \
    --num-layers 4 --hidden-size 1024 --num-attention-heads 8 \
    --seq-length 512 --max-position-embeddings 512 --micro-batch-size 1
```

### 2. Pipeline Parallelism Changes

#### TP2,PP2 → TP4,PP1 (Collocated, 4 GPUs)
```bash
sbatch --nodes=1 --gpus-per-node=4 benchmark_refit_sbatch.sh \
    --tensor-model-parallel-size 2 \
    --pipeline-model-parallel-size 2 \
    --rl-inference-tensor-model-parallel-size 4 \
    --rl-inference-pipeline-model-parallel-size 1 \
    --refit-mode collocated \
    --num-layers 8 --hidden-size 2048 --num-attention-heads 16 \
    --seq-length 1024 --max-position-embeddings 1024 --micro-batch-size 1
```

#### TP1,PP2 → TP2,PP1 (Collocated, 2 GPUs)
```bash
sbatch --nodes=1 --gpus-per-node=2 benchmark_refit_sbatch.sh \
    --tensor-model-parallel-size 1 \
    --pipeline-model-parallel-size 2 \
    --rl-inference-tensor-model-parallel-size 2 \
    --rl-inference-pipeline-model-parallel-size 1 \
    --refit-mode collocated \
    --num-layers 8 --hidden-size 1024 --num-attention-heads 8 \
    --seq-length 512 --max-position-embeddings 512 --micro-batch-size 1
```

### 3. Expert Parallelism (MoE Models)

#### EP2 → EP1 (Collocated, 2 GPUs)
```bash
sbatch --nodes=1 --gpus-per-node=2 benchmark_refit_sbatch.sh \
    --tensor-model-parallel-size 1 \
    --expert-model-parallel-size 2 \
    --rl-inference-tensor-model-parallel-size 1 \
    --rl-inference-expert-model-parallel-size 1 \
    --num-experts 8 \
    --refit-mode collocated \
    --num-layers 4 --hidden-size 1024 --num-attention-heads 8 \
    --ffn-hidden-size 4096 \
    --seq-length 512 --max-position-embeddings 512 --micro-batch-size 1
```

#### EP2 → EP4 (Collocated, 4 GPUs)
```bash
sbatch --nodes=1 --gpus-per-node=4 benchmark_refit_sbatch.sh \
    --tensor-model-parallel-size 1 \
    --expert-model-parallel-size 2 \
    --rl-inference-tensor-model-parallel-size 1 \
    --rl-inference-expert-model-parallel-size 4 \
    --num-experts 8 \
    --refit-mode collocated \
    --num-layers 4 --hidden-size 1024 --num-attention-heads 8 \
    --ffn-hidden-size 4096 \
    --seq-length 512 --max-position-embeddings 512 --micro-batch-size 1
```

### 4. Multi-Node Benchmarks

#### TP4,PP2 → TP8,PP1 (2 nodes, 8 GPUs each = 16 total)
```bash
sbatch --nodes=2 --ntasks-per-node=8 --gpus-per-node=8 benchmark_refit_sbatch.sh \
    --tensor-model-parallel-size 4 \
    --pipeline-model-parallel-size 2 \
    --rl-inference-tensor-model-parallel-size 8 \
    --rl-inference-pipeline-model-parallel-size 1 \
    --refit-mode collocated \
    --num-layers 16 --hidden-size 4096 --num-attention-heads 32 \
    --seq-length 2048 --max-position-embeddings 2048 --micro-batch-size 1 \
    --num-benchmark-iterations 50
```

#### TP8,PP4 → TP16,PP2 (4 nodes, 8 GPUs each = 32 total)
```bash
sbatch --nodes=4 --ntasks-per-node=8 --gpus-per-node=8 benchmark_refit_sbatch.sh \
    --tensor-model-parallel-size 8 \
    --pipeline-model-parallel-size 4 \
    --rl-inference-tensor-model-parallel-size 16 \
    --rl-inference-pipeline-model-parallel-size 2 \
    --refit-mode collocated \
    --num-layers 32 --hidden-size 8192 --num-attention-heads 64 \
    --seq-length 4096 --max-position-embeddings 4096 --micro-batch-size 1 \
    --num-benchmark-iterations 100
```

### 5. Backend Comparison

Compare different refit backends on the same configuration:

```bash
# NCCL
sbatch --nodes=1 --gpus-per-node=2 --job-name=refit-nccl benchmark_refit_sbatch.sh \
    --tensor-model-parallel-size 2 --rl-inference-tensor-model-parallel-size 1 \
    --refit-mode collocated --refit-method nccl \
    --num-layers 4 --hidden-size 1024 --num-attention-heads 8 \
    --seq-length 512 --max-position-embeddings 512 --micro-batch-size 1

# NVSHMEM
sbatch --nodes=1 --gpus-per-node=2 --job-name=refit-nvshmem benchmark_refit_sbatch.sh \
    --tensor-model-parallel-size 2 --rl-inference-tensor-model-parallel-size 1 \
    --refit-mode collocated --refit-method nvshmem \
    --num-layers 4 --hidden-size 1024 --num-attention-heads 8 \
    --seq-length 512 --max-position-embeddings 512 --micro-batch-size 1

# Gloo (debugging only)
sbatch --nodes=1 --gpus-per-node=2 --job-name=refit-gloo benchmark_refit_sbatch.sh \
    --tensor-model-parallel-size 2 --rl-inference-tensor-model-parallel-size 1 \
    --refit-mode collocated --refit-method gloo \
    --num-layers 4 --hidden-size 1024 --num-attention-heads 8 \
    --seq-length 512 --max-position-embeddings 512 --micro-batch-size 1
```

## Advanced Usage

### Custom Container Image

```bash
export CONTAINER_IMAGE="/path/to/your/container.sqsh"

sbatch --nodes=1 --gpus-per-node=2 benchmark_refit_sbatch.sh \
    --tensor-model-parallel-size 2 --rl-inference-tensor-model-parallel-size 1 \
    --refit-mode collocated \
    --num-layers 4 --hidden-size 1024 --num-attention-heads 8 \
    --seq-length 512 --max-position-embeddings 512 --micro-batch-size 1
```

### Custom Output Directory

```bash
export OUTPUT_DIR="/lustre/fsw/path/to/your/results"

sbatch --nodes=1 --gpus-per-node=2 benchmark_refit_sbatch.sh \
    --tensor-model-parallel-size 2 --rl-inference-tensor-model-parallel-size 1 \
    --refit-mode collocated \
    --num-layers 4 --hidden-size 1024 --num-attention-heads 8 \
    --seq-length 512 --max-position-embeddings 512 --micro-batch-size 1
```

### Extended Benchmarking

For more thorough benchmarking with warmup:

```bash
sbatch --nodes=1 --gpus-per-node=4 benchmark_refit_sbatch.sh \
    --tensor-model-parallel-size 2 --pipeline-model-parallel-size 2 \
    --rl-inference-tensor-model-parallel-size 4 --rl-inference-pipeline-model-parallel-size 1 \
    --refit-mode collocated \
    --num-layers 16 --hidden-size 4096 --num-attention-heads 32 \
    --seq-length 2048 --max-position-embeddings 2048 --micro-batch-size 1 \
    --num-benchmark-warmup 10 \
    --num-benchmark-iterations 100
```

### Mixed Precision

Use bf16 or fp16 for faster benchmarking:

```bash
sbatch --nodes=1 --gpus-per-node=2 benchmark_refit_sbatch.sh \
    --tensor-model-parallel-size 2 --rl-inference-tensor-model-parallel-size 1 \
    --refit-mode collocated \
    --bf16 \
    --num-layers 4 --hidden-size 1024 --num-attention-heads 8 \
    --seq-length 512 --max-position-embeddings 512 --micro-batch-size 1
```

## Interactive Debugging Tips

### Single GPU Debug

For single-GPU debugging (useful for testing code changes):

```bash
salloc --nodes=1 --gpus-per-node=1 --partition=interactive --time=00:30:00

# Force single GPU mode by setting CUDA_VISIBLE_DEVICES
CUDA_VISIBLE_DEVICES=0 python benchmark_refit.py \
    --tensor-model-parallel-size 1 \
    --rl-inference-tensor-model-parallel-size 1 \
    --refit-mode collocated \
    --num-layers 2 --hidden-size 512 --num-attention-heads 4 \
    --seq-length 256 --max-position-embeddings 256 --micro-batch-size 1 \
    --num-benchmark-iterations 2
```

### Verbose Logging

Enable detailed logging for debugging:

```bash
salloc --nodes=1 --gpus-per-node=2 --partition=interactive

# Set log level
export NCCL_DEBUG=INFO

./benchmark_refit_sbatch.sh \
    --tensor-model-parallel-size 2 --rl-inference-tensor-model-parallel-size 1 \
    --refit-mode collocated \
    --num-layers 2 --hidden-size 512 --num-attention-heads 4 \
    --seq-length 256 --max-position-embeddings 256 --micro-batch-size 1 \
    --num-benchmark-iterations 2
```

## Monitoring Results

### Check Job Status

```bash
# List running jobs
squeue -u $USER

# Monitor specific job
watch -n 1 squeue -j <job_id>

# View job details
scontrol show job <job_id>
```

### View Logs

```bash
# Logs are saved in benchmark_results/logs/ by default
cd examples/rl/benchmark_results/logs

# View latest log
tail -f $(ls -t | head -1)

# Search for results
grep "RESULTS" *.log

# Extract timing info
grep "Mean refit time" *.log
```

### Parse Results

Extract mean refit times from all logs:

```bash
cd examples/rl/benchmark_results/logs
for log in *.log; do
    echo -n "$log: "
    grep "Mean refit time" "$log" | awk '{print $4, $5}'
done
```

## Troubleshooting

### Job Fails Immediately

Check the log file for errors:
```bash
cd examples/rl/benchmark_results/logs
tail -100 <latest_log_file>
```

Common issues:
- Missing required arguments (see BENCHMARK_REFIT.md for required args)
- Insufficient GPUs for requested parallelism
- Container image not found

### Out of Memory

Reduce model size:
```bash
--num-layers 2 \
--hidden-size 512 \
--seq-length 256
```

Or use CPU initialization:
```bash
--use-cpu-initialization
```

### Slow Performance

Ensure you're using:
- NCCL or NVSHMEM backend (not Gloo)
- Sufficient warmup iterations (--num-benchmark-warmup 5)
- Production partition (not interactive) for final benchmarks

## Best Practices

1. **Start Small**: Begin with small models and few iterations in interactive mode
2. **Use Interactive for Debug**: Test configurations interactively before batch submission
3. **Multiple Runs**: Run benchmarks multiple times to ensure stable results
4. **Log Everything**: Use custom OUTPUT_DIR to organize different benchmark runs
5. **Compare Backends**: Test both NCCL and NVSHMEM to find the best for your hardware
6. **Scale Up Gradually**: Test single-node before multi-node configurations

## Environment Variables

- `CONTAINER_IMAGE`: Path to container image (default: megatron container)
- `OUTPUT_DIR`: Directory for logs and results (default: ./benchmark_results)
- `RUN_ID`: Custom run identifier (default: auto-generated)
- `NCCL_DEBUG`: NCCL logging level (INFO, WARN, ERROR)
- `CUDA_VISIBLE_DEVICES`: Limit visible GPUs (for debugging)
