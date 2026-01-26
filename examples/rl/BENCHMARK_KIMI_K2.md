# Kimi K2 Refit Benchmark

This directory contains scripts for benchmarking refit (weight resharding) performance on the Kimi K2 model.

## Kimi K2 Model Specifications

**Kimi K2** is a 1.04 trillion parameter Mixture-of-Experts (MoE) model developed by Moonshot AI.

### Architecture Details

| Parameter | Value |
|-----------|-------|
| **Total Parameters** | 1.04T (1 trillion) |
| **Active Parameters** | 32B per token |
| **Number of Layers** | 61 (60 MoE + 1 dense) |
| **Hidden Dimension** | 7,168 |
| **Number of Attention Heads** | 64 |
| **Vocabulary Size** | 160K tokens |
| **Context Length** | 128K tokens |
| **Number of Experts** | 384 |
| **Active Experts per Token** | 8 |
| **Shared Experts** | 1 |
| **MoE Intermediate Size** | 2,048 |
| **Attention Mechanism** | Multi-head Latent Attention (MLA) |
| **Activation Function** | SwiGLU |
| **Position Embeddings** | RoPE |

## Benchmarking Scripts

### Multi-Node Benchmark: `benchmark_refit_kimi_k2.sh`

This script runs comprehensive refit benchmarks on the Kimi K2 model using multiple nodes via SLURM.

**Default Configuration:**
- **Nodes:** 16
- **GPUs per node:** 8
- **Total GPUs:** 128
- **Time limit:** 2 hours
- **Partition:** batch
- **Account:** llmservice_fm_text

**Tested Configurations:**

The script tests various tensor parallelism (TP) refit scenarios:

1. **Collocated Mode** (both models share GPUs):
   - TP 8 → TP 4 (NCCL)
   - TP 8 → TP 2 (NCCL)
   - TP 8 → TP 1 (NCCL)
   - TP 4 → TP 2 (NCCL)
   - TP 4 → TP 1 (NCCL)

2. **Non-Collocated Mode** (models on separate GPU sets):
   - Commented out by default, uncomment to test
   - TP 8 → TP 4 (NVSHMEM)
   - TP 4 → TP 2 (NVSHMEM)

## Usage

### Submit Multi-Node Benchmark

```bash
cd /lustre/fsw/portfolios/adlr/projects/adlr_psx_fp8/users/wdykas/code/ep-refit/mrl_internal/megatron-rl/examples/rl

# Submit the job
sbatch benchmark_refit_kimi_k2.sh
```

### Customize the Benchmark

You can modify the script to test different configurations:

1. **Change number of nodes:**
   ```bash
   #SBATCH --nodes=8  # Use 8 nodes instead of 16
   ```

2. **Modify TP configurations:**
   Edit the `CONFIGS` array in the script:
   ```bash
   CONFIGS=(
       "8:4:collocated:nccl"
       "16:8:collocated:nccl"  # Add more configs
   )
   ```

3. **Change sequence length:**
   ```bash
   SEQ_LENGTH=4096  # Use smaller sequence length
   ```

4. **Adjust benchmark iterations:**
   ```bash
   NUM_BENCHMARK_WARMUP=3
   NUM_BENCHMARK_ITERATIONS=10
   ```

### Run a Single Configuration

To test a specific configuration without sbatch (on an interactive node):

```bash
cd /lustre/fsw/portfolios/adlr/projects/adlr_psx_fp8/users/wdykas/code/ep-refit/mrl_internal/megatron-rl/examples/rl

# Example: TP 8 -> TP 4, collocated mode
./benchmark_refit_sbatch.sh \
    --tensor-model-parallel-size 8 \
    --rl-inference-tensor-model-parallel-size 4 \
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

## Output

The benchmark generates several output files:

1. **Summary file:** `benchmark_results/kimi_k2/logs/<RUN_ID>_summary.txt`
   - Contains results for all configurations in one file

2. **Individual logs:** `benchmark_results/kimi_k2/logs/<RUN_ID>_tp<SRC>_to_tp<DST>_<MODE>_<METHOD>.log`
   - Detailed logs for each configuration

3. **Job info:** `benchmark_results/kimi_k2/logs/job_info_<RUN_ID>.log`
   - SLURM job information

### Example Output

```
Kimi K2 Refit Benchmark Summary
================================
Date: 2026-01-26 12:30:00
Job ID: 9876543
Nodes: 16
Total GPUs: 128

Model Configuration:
  Layers: 61
  Hidden size: 7168
  Attention heads: 64
  MoE experts: 384
  Active experts: 8
  Vocabulary: 160000

Results:
--------

Config: TP 8 -> TP 4 (collocated, nccl)
Mean refit time: 45.23 ms
Min refit time:  44.87 ms
Max refit time:  45.61 ms

Config: TP 8 -> TP 2 (collocated, nccl)
Mean refit time: 52.14 ms
Min refit time:  51.89 ms
Max refit time:  52.47 ms
...
```

## Notes

- The benchmark measures **only execution time**, not plan building time
- The refit plan is built during warmup iterations and cached
- Proper CUDA and distributed synchronization ensures accurate timing
- For the full 128K context, you may need to adjust `SEQ_LENGTH` and allocate more memory

## References

- [Kimi K2 Official Page](https://moonshotai.github.io/Kimi-K2/)
- [Kimi K2 Hugging Face](https://huggingface.co/moonshotai/Kimi-K2-Instruct)
- [Kimi K2 Technical Report](https://arxiv.org/abs/2507.20534)
- [Fireworks AI Blog: Kimi K2 Deep Dive](https://fireworks.ai/blog/kimi-k2-deepdive)
