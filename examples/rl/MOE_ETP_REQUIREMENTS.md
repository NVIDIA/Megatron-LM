# MoE Expert Tensor Parallelism (ETP) Requirements

## TL;DR

When benchmarking MoE models, **always** add these flags:
```bash
--expert-tensor-parallel-size 1 \
--rl-inference-expert-tensor-model-parallel-size 1 \
--disable-bias-linear
```

## What is ETP?

There are two types of parallelism for MoE experts:

| Parallelism Type | Abbreviation | What it does |
|------------------|--------------|--------------|
| **Expert Parallelism** | EP | Distributes **different experts** across GPUs |
| **Expert Tensor Parallelism** | ETP | Splits **each expert's weights** across GPUs (like TP for experts) |

### Example:
- Model has 384 experts
- EP=4: Each GPU group has 384/4 = 96 experts
- ETP=2: Each of those 96 experts is split across 2 GPUs

## The Problem

Megatron-Core has a restriction:
```python
assert self.expert_tensor_parallel_size == 1 or not self.add_bias_linear
# Translation: If ETP > 1, then bias must be disabled
```

By default, ETP inherits from TP:
```python
if args.expert_tensor_parallel_size is None:
    args.expert_tensor_parallel_size = args.tensor_model_parallel_size
```

So if you set `--tensor-model-parallel-size 2`, then ETP becomes 2 automatically!

## The Solution

### Option 1: Set ETP=1 (Recommended for Benchmarking)

```bash
--expert-tensor-parallel-size 1 \
--rl-inference-expert-tensor-model-parallel-size 1
```

This keeps experts whole (not split), which is typical for MoE models.

### Option 2: Disable Bias

```bash
--disable-bias-linear
```

This allows ETP > 1 but removes bias from linear layers.

### Best Practice: Use Both

For benchmarking, use both to be safe:
```bash
--expert-tensor-parallel-size 1 \
--rl-inference-expert-tensor-model-parallel-size 1 \
--disable-bias-linear
```

## When Do You Need This?

✅ **YES** - Need these flags when:
- Using `--num-experts` (MoE model)
- With `--tensor-model-parallel-size > 1`

❌ **NO** - Don't need these flags when:
- Not using MoE (`--num-experts` not specified)
- Using TP=1 (ETP defaults to 1)

## Real-World Impact

For refit benchmarking:
- ✓ ETP=1 is standard for MoE models (experts not split)
- ✓ Bias disabled has minimal impact on refit performance measurement
- ✓ These settings match production MoE configurations

## Examples

### ❌ This Fails:
```bash
./benchmark_refit_sbatch.sh \
    --tensor-model-parallel-size 2 \    # TP=2
    --expert-model-parallel-size 2 \    # EP=2
    --num-experts 16 \                  # MoE enabled
    # ❌ Missing: ETP defaults to TP=2, bias enabled → FAILS
```

**Error:** `AssertionError: Bias in Moe is only supported when ETP==1`

### ✅ This Works:
```bash
./benchmark_refit_sbatch.sh \
    --tensor-model-parallel-size 2 \              # TP=2
    --expert-model-parallel-size 2 \              # EP=2
    --expert-tensor-parallel-size 1 \             # ✓ ETP=1
    --num-experts 16 \                            # MoE enabled
    --disable-bias-linear \                       # ✓ Bias disabled
    # ... other args ...
```

### ✅ Alternative (No MoE):
```bash
./benchmark_refit_sbatch.sh \
    --tensor-model-parallel-size 2 \
    # No --num-experts → Not MoE → No ETP restriction
```

## Kimi K2 Configuration

Kimi K2 uses MoE (384 experts), so all configs **must** include:
```bash
--expert-tensor-parallel-size 1
--rl-inference-expert-tensor-model-parallel-size 1
--disable-bias-linear
```

These are already added to:
- ✅ `test_nvshmem_refit.sh`
- ✅ `benchmark_refit_kimi_k2.sh`
- ✅ All examples in `NVSHMEM_QUICK_TEST.md`

## Quick Reference

| Scenario | Flags Needed |
|----------|--------------|
| MoE + TP > 1 | `--expert-tensor-parallel-size 1`, `--disable-bias-linear` |
| MoE + TP = 1 | None (ETP defaults to 1) |
| No MoE | None (ETP not relevant) |
| Kimi K2 | Always include ETP=1 + disable-bias |

## Verification

To check if your command will work:
```python
# These must both be true:
ETP == 1 OR bias_disabled == True

# Where ETP defaults to:
ETP = expert_tensor_parallel_size if specified else tensor_model_parallel_size
```

## Files Updated

All these files now include proper ETP settings:
- `test_nvshmem_refit.sh`
- `benchmark_refit_kimi_k2.sh`
- `NVSHMEM_QUICK_TEST.md`
- `8GPU_CONFIGS.md` (examples)
- `MOE_ETP_REQUIREMENTS.md` (this file)
