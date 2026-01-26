# Refit Benchmark

Standalone benchmark script for measuring the performance of refit operations in Megatron-RL.

## What is Refit?

Refit is the process of resharding model weights between different parallelism configurations. In RL training, this allows you to use different parallelism strategies for training and inference phases:
- Training might use TP=2, PP=2 for optimal training throughput
- Inference might use TP=8, PP=1 for optimal inference latency

The refit operation transfers weights between these configurations, supporting changes in:
- **Tensor Parallelism (TP)**: Splitting model layers across GPUs
- **Pipeline Parallelism (PP)**: Splitting model layers into pipeline stages
- **Expert Parallelism (EP)**: Splitting MoE experts across GPUs

## Modes

### Collocated Mode
Training and inference models share the same set of GPUs. This is memory-efficient but requires careful scheduling to avoid OOM.

Example: 4 GPUs running both training (TP=2, PP=2) and inference (TP=4, PP=1) models.

### Non-Collocated Mode
Training and inference models use separate GPU sets. This provides isolation but requires more GPUs.

Example: 8 GPUs total, with GPUs 0-3 for training (TP=2, PP=2) and GPUs 4-7 for inference (TP=4, PP=1).

## Usage

The benchmark script uses the standard Megatron argument system. You specify the training parallelism with the standard Megatron arguments and the inference parallelism with the RL-specific arguments.

### Basic Syntax

```bash
python benchmark_refit.py \
    --tensor-model-parallel-size <training_tp> \
    --pipeline-model-parallel-size <training_pp> \
    --expert-model-parallel-size <training_ep> \
    --rl-inference-tensor-model-parallel-size <inference_tp> \
    --rl-inference-pipeline-model-parallel-size <inference_pp> \
    --rl-inference-expert-model-parallel-size <inference_ep> \
    --refit-mode <collocated|non-collocated> \
    --refit-method <nccl|gloo|nvshmem> \
    --num-layers <num_layers> \
    --hidden-size <hidden_size> \
    [additional Megatron options]
```

### Example Commands

#### 1. Collocated: TP2 → TP1 (2 GPUs, NCCL backend)

```bash
python benchmark_refit.py \
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

#### 2. Collocated: TP2,PP2 → TP4,PP1 (4 GPUs, NVSHMEM backend)

```bash
python benchmark_refit.py \
    --tensor-model-parallel-size 2 \
    --pipeline-model-parallel-size 2 \
    --rl-inference-tensor-model-parallel-size 4 \
    --rl-inference-pipeline-model-parallel-size 1 \
    --refit-mode collocated \
    --refit-method nvshmem \
    --num-layers 8 \
    --hidden-size 2048 \
    --num-attention-heads 16 \
    --seq-length 1024 \
    --max-position-embeddings 1024 \
    --micro-batch-size 1
```

#### 3. Non-Collocated: TP2 → TP4 (6 GPUs: 2 for training, 4 for inference)

```bash
python benchmark_refit.py \
    --tensor-model-parallel-size 2 \
    --rl-inference-tensor-model-parallel-size 4 \
    --refit-mode non-collocated \
    --refit-method nccl \
    --num-layers 4 \
    --hidden-size 1024 \
    --num-attention-heads 8 \
    --seq-length 512 \
    --max-position-embeddings 512 \
    --micro-batch-size 1
```

#### 4. MoE Model: EP2 → EP4 (4 GPUs, collocated)

```bash
python benchmark_refit.py \
    --tensor-model-parallel-size 1 \
    --expert-model-parallel-size 2 \
    --rl-inference-tensor-model-parallel-size 1 \
    --rl-inference-expert-model-parallel-size 4 \
    --num-experts 8 \
    --refit-mode collocated \
    --refit-method nccl \
    --num-layers 4 \
    --hidden-size 1024 \
    --num-attention-heads 8 \
    --ffn-hidden-size 4096 \
    --seq-length 512 \
    --max-position-embeddings 512 \
    --micro-batch-size 1
```

#### 5. Large Model Configuration

```bash
python benchmark_refit.py \
    --tensor-model-parallel-size 2 \
    --pipeline-model-parallel-size 4 \
    --rl-inference-tensor-model-parallel-size 8 \
    --rl-inference-pipeline-model-parallel-size 1 \
    --refit-mode collocated \
    --refit-method nvshmem \
    --num-layers 32 \
    --hidden-size 4096 \
    --num-attention-heads 32 \
    --seq-length 2048 \
    --max-position-embeddings 2048 \
    --micro-batch-size 1 \
    --num-benchmark-warmup 5 \
    --num-benchmark-iterations 50
```

## Command Line Arguments

### Benchmark-Specific Arguments
- `--refit-mode`: Operation mode (`collocated` or `non-collocated`) **[REQUIRED]**
- `--num-benchmark-warmup`: Number of warmup iterations (default: 2)
- `--num-benchmark-iterations`: Number of benchmark iterations (default: 10)

### Training Parallelism (Standard Megatron Args)
- `--tensor-model-parallel-size`: Training tensor parallelism (default: 1)
- `--pipeline-model-parallel-size`: Training pipeline parallelism (default: 1)
- `--expert-model-parallel-size`: Training expert parallelism (default: 1)

### Inference Parallelism (RL Args - already exist in Megatron)
- `--rl-inference-tensor-model-parallel-size`: Inference tensor parallelism (default: 1)
- `--rl-inference-pipeline-model-parallel-size`: Inference pipeline parallelism (default: 1)
- `--rl-inference-expert-model-parallel-size`: Inference expert parallelism (default: 1)

### Model Configuration (Standard Megatron Args)
- `--num-layers`: Number of transformer layers (default: 2) **[REQUIRED]**
- `--hidden-size`: Hidden size (default: 512) **[REQUIRED]**
- `--num-attention-heads`: Number of attention heads (default: 8) **[REQUIRED]**
- `--seq-length`: Sequence length **[REQUIRED]**
- `--max-position-embeddings`: Maximum position embeddings **[REQUIRED]**
- `--num-experts`: Number of MoE experts, enables MoE (default: None)
- `--ffn-hidden-size`: FFN hidden size (default: 4*hidden_size)
- `--micro-batch-size`: Micro batch size (default: 1)

### Backend Selection (Already in Megatron)
- `--refit-method`: Refit backend (`nccl`, `gloo`, or `nvshmem`, default: `nvshmem`)
  - `nccl`: Uses NCCL for GPU-to-GPU communication (recommended for most cases)
  - `gloo`: Uses Gloo over CPU (useful for debugging)
  - `nvshmem`: Uses NVSHMEM for optimized GPU communication (default in Megatron)

### Other Useful Megatron Args
- `--use-cpu-initialization`: Initialize model on CPU to save GPU memory
- `--bf16` / `--fp16`: Use mixed precision
- `--use-flash-attn`: Use Flash Attention

## GPU Requirements

### Collocated Mode
Number of GPUs = LCM(training_world_size, inference_world_size)

Where:
- training_world_size = training_tp × training_pp × training_ep
- inference_world_size = inference_tp × inference_pp × inference_ep
- LCM = Least Common Multiple

Examples:
- TP2→TP1: needs 2 GPUs (LCM(2,1) = 2)
- TP2,PP2→TP4,PP1: needs 4 GPUs (LCM(4,4) = 4)
- TP2→TP3: needs 6 GPUs (LCM(2,3) = 6)

### Non-Collocated Mode
Number of GPUs = training_world_size + inference_world_size

Examples:
- TP2→TP4: needs 6 GPUs (2+4)
- TP2,PP2→TP4,PP1: needs 8 GPUs (4+4)

## Output

The benchmark reports:
- Configuration details (parallelism settings, model size)
- Per-iteration timing
- Statistical summary:
  - Mean refit time
  - Min refit time
  - Max refit time

Example output:
```
================================================================================
COLLOCATED MODE REFIT BENCHMARK
================================================================================
World size: 4
Source (training): TP=2, PP=2, EP=1, DP=1
Destination (inference): TP=4, PP=1, EP=1, DP=1
Model: 4 layers, 1024 hidden, 8 heads
Refit backend: nccl
================================================================================

Building models...
Source model size on rank 0: 45.23 MB
Destination model size on rank 0: 45.23 MB

Warmup: 2 iterations...

Benchmarking: 10 iterations...
  Iteration 1/10: 123.45 ms
  Iteration 2/10: 121.34 ms
  ...

================================================================================
RESULTS
================================================================================
Mean refit time: 122.15 ms
Min refit time:  119.87 ms
Max refit time:  125.43 ms
================================================================================
```

## Helper Script

A convenience shell script `run_refit_benchmarks.sh` provides pre-configured benchmark scenarios:

```bash
# Show available benchmarks
./run_refit_benchmarks.sh

# Run a specific benchmark
./run_refit_benchmarks.sh 1  # TP2→TP1 Collocated
```

## Tips for Accurate Benchmarking

1. **Warmup**: Use sufficient warmup iterations (at least 2-5) to ensure CUDA kernels are compiled and caches are warm.

2. **Iterations**: Use enough benchmark iterations (10-50) to get stable statistics.

3. **System Load**: Ensure the system is not under heavy load from other processes.

4. **Backend Selection**:
   - Use NCCL for production-like measurements
   - Use NVSHMEM if available for potentially better performance (default)
   - Use Gloo only for debugging

5. **Model Size**: Test with model configurations representative of your actual workload.

6. **Multiple Runs**: Run the benchmark multiple times and average results for the most accurate measurements.

## Required Arguments

The benchmark requires several Megatron arguments that don't have defaults. Here's a minimal working command:

```bash
python benchmark_refit.py \
    --tensor-model-parallel-size 2 \
    --rl-inference-tensor-model-parallel-size 1 \
    --refit-mode collocated \
    --num-layers 4 \
    --hidden-size 1024 \
    --num-attention-heads 8 \
    --seq-length 512 \
    --max-position-embeddings 512 \
    --micro-batch-size 1
```

## Troubleshooting

### Out of Memory (OOM)
- Reduce model size (fewer layers, smaller hidden size)
- Use non-collocated mode if possible
- Add `--use-cpu-initialization`
- Use `--bf16` or `--fp16` for mixed precision

### NVSHMEM Not Available
- Use `--refit-method nccl` instead
- NVSHMEM requires special installation and may not be available on all systems

### World Size Mismatch
- Ensure the number of GPUs matches the requirements for your chosen mode
- Check the GPU Requirements section above

### Missing Required Arguments
- The script will tell you which arguments are missing
- See "Required Arguments" section above for a minimal working command

## Integration with RL Training

This benchmark measures the standalone refit performance. In actual RL training, refit happens between:
1. Training phase: Update policy using collected rollouts
2. Inference phase: Generate new rollouts with the updated policy

The refit overhead should be minimized relative to the training and inference phases. Use this benchmark to:
- Choose optimal parallelism configurations
- Select the best refit backend for your hardware
- Estimate the overhead of refit in your training loop
- Compare collocated vs non-collocated modes
