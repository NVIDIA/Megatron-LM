# Kimi K2 Benchmark Configuration Summary

## Visual Overview of Test Scenarios

```
Total Resources: 16 nodes × 8 GPUs = 128 GPUs
Model: Kimi K2 (1.04T params, 384 experts, 8 active per token)
```

### Scenario Matrix

| # | Source Config | Target Config | Total GPUs | Mode | Method | Purpose |
|---|---------------|---------------|------------|------|--------|---------|
| 1 | TP=8, EP=8 (64) | TP=8, EP=4 (32) | 96 | non-collocated | NVSHMEM | EP refit isolation |
| 2 | TP=16, EP=4 (64) | TP=8, EP=4 (32) | 96 | non-collocated | NVSHMEM | TP refit isolation |
| 3 | TP=16, EP=4 (64) | TP=16, EP=2 (32) | 96 | non-collocated | NVSHMEM | EP refit (high TP) |
| 4 | TP=16, EP=4 (64) | TP=32, EP=2 (64) | 128 | non-collocated | NVSHMEM | TP+EP refit (equal GPUs) |
| 5 | TP=8, EP=8 (64) | TP=16, EP=4 (64) | 128 | non-collocated | NVSHMEM | TP+EP refit (balanced→TP-heavy) |
| 6 | TP=16, EP=8 (128) | TP=8, EP=16 (128) | 128 | collocated | NCCL | TP↔EP swap (full resources) |
| 7 | TP=32, EP=4 (128) | TP=16, EP=8 (128) | 128 | collocated | NCCL | TP scaling (very high→balanced) |

## Configuration Details

### Non-Collocated Scenarios (1-5): NVSHMEM

These test **realistic training→inference** patterns where models use separate GPU sets.

#### Scenario 1: Pure EP Refit
```
┌─────────────────────────────────────┐     ┌──────────────────────┐
│   Training Model (64 GPUs)          │ ──→ │ Inference (32 GPUs)  │
│   TP=8  : 8-way tensor split        │     │ TP=8  : same        │
│   EP=8  : 384/8=48 experts/group    │     │ EP=4  : 96 experts/group │
│   Layout: 8 TP groups × 8 EP groups │     │ Layout: 8×4          │
└─────────────────────────────────────┘     └──────────────────────┘

Tests: Expert redistribution, router updates
Expected overhead: Expert gathering/scattering
```

#### Scenario 2: Pure TP Refit
```
┌─────────────────────────────────────┐     ┌──────────────────────┐
│   Training Model (64 GPUs)          │ ──→ │ Inference (32 GPUs)  │
│   TP=16 : 16-way tensor split       │     │ TP=8  : 8-way split  │
│   EP=4  : 96 experts/group          │     │ EP=4  : same         │
│   Layout: 16 TP groups × 4 EP groups│     │ Layout: 8×4          │
└─────────────────────────────────────┘     └──────────────────────┘

Tests: Attention/MLP weight gathering
Expected overhead: All-gather for dense layers
```

#### Scenario 3: EP Refit (High TP)
```
┌─────────────────────────────────────┐     ┌──────────────────────┐
│   Training Model (64 GPUs)          │ ──→ │ Inference (32 GPUs)  │
│   TP=16 : 16-way tensor split       │     │ TP=16 : same         │
│   EP=4  : 96 experts/group          │     │ EP=2  : 192 experts/group │
│   Layout: 16 TP groups × 4 EP groups│     │ Layout: 16×2         │
└─────────────────────────────────────┘     └──────────────────────┘

Tests: EP refit with fine-grained TP
Expected: Higher communication for TP maintenance
```

#### Scenario 4: TP+EP Refit (Same GPU Count)
```
┌─────────────────────────────────────┐     ┌──────────────────────────┐
│   Training Model (64 GPUs)          │ ──→ │ Inference (64 GPUs)      │
│   TP=16 : 16-way tensor split       │     │ TP=32 : 32-way split     │
│   EP=4  : 96 experts/group          │     │ EP=2  : 192 experts/group│
│   Layout: 16×4                       │     │ Layout: 32×2             │
└─────────────────────────────────────┘     └──────────────────────────┘

Tests: Simultaneous TP and EP refit
Expected: Combined overhead from both dimensions
```

#### Scenario 5: Balanced→TP-heavy
```
┌─────────────────────────────────────┐     ┌──────────────────────────┐
│   Training Model (64 GPUs)          │ ──→ │ Inference (64 GPUs)      │
│   TP=8  : Balanced parallelism      │     │ TP=16 : TP-heavy         │
│   EP=8  : 48 experts/group          │     │ EP=4  : 96 experts/group │
│   Layout: 8×8 (balanced)             │     │ Layout: 16×4 (TP-heavy)  │
└─────────────────────────────────────┘     └──────────────────────────┘

Tests: Strategy shift from balanced to TP-dominant
Expected: Major data redistribution pattern
```

### Collocated Scenarios (6-7): NCCL

These test **on-demand refit** where both models share all 128 GPUs.

#### Scenario 6: TP↔EP Swap
```
┌──────────────────────────────────────────────┐
│         All 128 GPUs (shared)                │
│                                              │
│  Model A (TP-heavy):  TP=16, EP=8           │
│     16 tensor groups × 8 expert groups      │
│     Better for: Dense layer performance     │
│                      ↕ REFIT                │
│  Model B (EP-heavy):  TP=8, EP=16           │
│     8 tensor groups × 16 expert groups      │
│     Better for: Expert diversity, routing   │
└──────────────────────────────────────────────┘

Tests: Swapping parallelism strategy on-the-fly
Expected: Fast refit using shared GPU memory
```

#### Scenario 7: TP Scaling
```
┌──────────────────────────────────────────────┐
│         All 128 GPUs (shared)                │
│                                              │
│  Model A (Very high TP): TP=32, EP=4        │
│     32 tensor groups × 4 expert groups      │
│     Max memory distribution                 │
│                      ↕ REFIT                │
│  Model B (Balanced):     TP=16, EP=8        │
│     16 tensor groups × 8 expert groups      │
│     Balanced performance                    │
└──────────────────────────────────────────────┘

Tests: Moving between TP scaling strategies
Expected: TP refit overhead with full GPU set
```

## Key Differences

### Non-Collocated vs Collocated

| Aspect | Non-Collocated | Collocated |
|--------|----------------|------------|
| **GPU Sets** | Separate (disjoint) | Shared (same GPUs) |
| **Memory** | Independent allocations | Can share/reuse |
| **Communication** | Cross-node (inter-GPU-set) | On-node possible |
| **Use Case** | Training→Inference pipeline | On-demand switching |
| **Method** | NVSHMEM (optimized for cross-node) | NCCL (optimized for on-node) |
| **Overhead** | Higher (data transfer) | Lower (shared memory) |

### TP vs EP Refit

| Dimension | What Changes | Communication Pattern | Expected Overhead |
|-----------|--------------|----------------------|-------------------|
| **TP (Tensor Parallel)** | How dense layers split | All-gather/reduce-scatter | Bandwidth-bound |
| **EP (Expert Parallel)** | How experts distribute | Expert routing redistribution | Compute+communication |
| **Both** | Complete resharding | Complex multi-stage | Highest overhead |

## What Each Test Tells Us

1. **Scenario 1**: Baseline for EP refit performance
2. **Scenario 2**: Baseline for TP refit performance
3. **Scenario 3**: How TP granularity affects EP refit
4. **Scenario 4**: Cost of changing both dimensions
5. **Scenario 5**: Cost of strategy shifts
6. **Scenario 6**: Best-case refit (collocated, shared memory)
7. **Scenario 7**: TP scaling behavior at full scale

## Expected Performance Ranking

From fastest to slowest (estimated):

1. **Scenario 6** (collocated TP↔EP): ~20-40ms ⭐ Fastest
2. **Scenario 7** (collocated TP scaling): ~25-50ms
3. **Scenario 1** (EP refit only, 64→32): ~30-60ms
4. **Scenario 2** (TP refit only, 64→32): ~50-100ms
5. **Scenario 3** (EP refit, high TP): ~60-120ms
6. **Scenario 4** (TP+EP, 64→64): ~80-150ms
7. **Scenario 5** (Both, strategy shift): ~100-200ms

*Actual times depend on hardware, network topology, and system load*

## Quick Reference Commands

### Submit all benchmarks
```bash
sbatch benchmark_refit_kimi_k2.sh
```

### Run specific scenario (interactive)
```bash
# Scenario 1: EP refit (needs 12 nodes = 96 GPUs)
salloc --nodes=12 --ntasks-per-node=8 --gpus-per-node=8 --time=1:00:00
./benchmark_refit_sbatch.sh \
    --tensor-model-parallel-size 8 \
    --expert-model-parallel-size 8 \
    --rl-inference-tensor-model-parallel-size 8 \
    --rl-inference-expert-model-parallel-size 4 \
    --refit-mode non-collocated \
    --refit-method nvshmem \
    [... other args ...]

# Scenario 6: Collocated TP↔EP (needs 16 nodes = 128 GPUs)
salloc --nodes=16 --ntasks-per-node=8 --gpus-per-node=8 --time=1:00:00
./benchmark_refit_sbatch.sh \
    --tensor-model-parallel-size 16 \
    --expert-model-parallel-size 8 \
    --rl-inference-tensor-model-parallel-size 8 \
    --rl-inference-expert-model-parallel-size 16 \
    --refit-mode collocated \
    --refit-method nccl \
    [... other args ...]
```

## Files
- `benchmark_refit_kimi_k2.sh` - Main benchmark script with all 7 scenarios
- `BENCHMARK_KIMI_K2_UPDATED.md` - Detailed documentation
- `KIMI_K2_CONFIG_SUMMARY.md` - This file (visual overview)
