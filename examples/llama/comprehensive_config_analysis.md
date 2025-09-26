# Megatron-LM FP8 Configuration Analysis - Comprehensive Report

## Table of Contents
1. [Overview and Common Notes](#overview-and-common-notes)
2. [Model Architecture Specifications](#model-architecture-specifications)
3. [Detailed Experiment Results](#detailed-experiment-results)
4. [Performance Analysis](#performance-analysis)
5. [Issues and Troubleshooting](#issues-and-troubleshooting)
6. [File Structure Reference](#file-structure-reference)
7. [Future Work](#future-work)

---

## Overview and Common Notes

### Key notes
- All runs are terminated by time limit unless otherwise mentioned
- 1B model runs show FP8 performs poorly compared to BF16, due to overhead of fp8 conversions. As matrix size increase the fp8 performance starts improving. Larger the model, larger the MATMUL operation larger pref gain compared to bf16.
- Evaluation was disabled to avoid memory surge crashes
- All experiments use SlimPajama dataset

---

## Model Architecture Specifications

### Llama-3-8B Architecture
- **Layers**: 32
- **Hidden Size**: 4096
- **FFN Hidden Size**: 14336
- **Attention Heads**: 32
- **Query Groups**: 8
- **Sequence Length**: 8192

### Llama-3-Small (1GPU) Architecture
- **Layers**: 12
- **Hidden Size**: 2048
- **FFN Hidden Size**: 5504
- **Attention Heads**: 16
- **Query Groups**: 4
- **Sequence Length**: 2048

---

## Detailed Experiment Results

### 1. train_llama3_8b_h200_bf16.sh
**Configuration File**: `train_llama3_8b_h200_bf16.sh`

**Key Parameters**:
- **Machine**: NVIDIA H200
- **Precision**: BF16
- **Wandb Name**: llama3x-8b-bf16-tp1

**Results**:
- **Runtime**: 1hr 56m
- **Min Loss**: ~6.4
- **Log File**: `./logs/train_llama3_8b_h200_bf16.log`

**Wandb Metrics**:
**Training Loss Plot:**
<img src="./assets/train_llama3_8b_h200_bf16_loss.png" width="600" alt="BF16 Training Loss">

**Iteration Time Analysis:**
<img src="./assets/train_llama3_8b_h200_bf16_itrtime.png" width="600" alt="BF16 Iteration Time">

---

### 2. train_llama3_8b_h200_fp8.sh
**Configuration File**: `train_llama3_8b_h200_fp8.sh`

**Key Parameters**:
- **Machine**: NVIDIA H200
- **Precision**: FP8 (delayed recipe - default)
- **Wandb Name**: llama3x-8b-fp8-tp1

**FP8 Configuration**:
```bash
--fp8-format hybrid
--fp8-amax-history-len 1024
--fp8-amax-compute-algo max
--fp8-param-gather
# Default recipe: delayed (not explicitly specified)
```

**Results**:
- **Runtime**: 1hr 56m
- **Min Loss**: ~5.9
- **Log File**: `./logs/train_llama3_8b_h200_fp8.log`

**Issues Encountered**:
- Failed consistently at evaluation intervals Required disabling evaluation to proceed

**Wandb Metrics**:
**Training Loss Plot:**
<img src="./assets/train_llama3_8b_h200_fp8_loss.png" width="600" alt="FP8 Delayed Training Loss">

**Iteration Time Analysis:**
<img src="./assets/train_llama3_8b_h200_fp8_itrtime.png" width="600" alt="FP8 Delayed Iteration Time">

---

### 3. train_llama3_8b_h200_fp8_cs.sh
**Configuration File**: `train_llama3_8b_h200_fp8_cs.sh`

**Key Parameters**:
- **Machine**: NVIDIA H200
- **Precision**: FP8 (tensorwise recipe)
- **Wandb Name**: llama3x-8b-fp8-tp1_cs

**FP8 Configuration**:
```bash
--fp8-format hybrid
--fp8-amax-history-len 1024
--fp8-amax-compute-algo max
--fp8-param-gather
--fp8-recipe tensorwise  # Explicit tensorwise recipe
```

**Results**:
- **Runtime**: 1hr 56m
- **Min Loss**: ~5.9
- **Log File**: `./logs/train_llama3_8b_h200_fp8_cs.logs`

**Wandb Metrics**:
**Training Loss Plot:**
<img src="./assets/train_llama3_8b_h200_fp8_cs_loss.png" width="600" alt="FP8 Tensorwise Training Loss">

**Iteration Time Analysis:**
<img src="./assets/train_llama3_8b_h200_fp8_cs_itrtime.png" width="600" alt="FP8 Tensorwise Iteration Time">

---

### 4. train_llama3_8b_h200_fp8_cs_1F1L.sh
**Configuration File**: `train_llama3_8b_h200_fp8_cs_1F1L.sh`

**Key Parameters**:
- **Machine**: NVIDIA H200
- **Precision**: FP8 (tensorwise recipe) with BF16 first/last layers
- **Wandb Name**: llama3x-8b-fp8-tp1_cs_1F1L

**FP8 Configuration**:
```bash
--fp8-format hybrid
--fp8-amax-history-len 1024
--fp8-amax-compute-algo max
--fp8-param-gather
--fp8-recipe tensorwise
--first-last-layers-bf16              # Keep first/last layers in BF16
--num-layers-at-start-in-bf16 1
--num-layers-at-end-in-bf16 1
```

**Results**:
- **Runtime**: 1hr 56m
- **Min Loss**: ~6.0
- **Log File**: `./logs/train_llama3_8b_h200_fp8_cs_1F1L`

**Wandb Metrics**:
**Training Loss Plot:**
<img src="./assets/train_llama3_8b_h200_fp8_cs_1F1L_loss.png" width="600" alt="FP8 Tensorwise First and last layer in BF16 Training Loss">

**Iteration Time Analysis:**
<img src="./assets/train_llama3_8b_h200_fp8_cs_1F1L_itrtime.png" width="600" alt="FP8 Tensorwise First and last layer in BF16 Iteration Time">

---

### 5. train_llama3_8b_h200_fp8_blockwise.sh
**Configuration File**: `train_llama3_8b_h200_fp8_blockwise.sh`

**Key Parameters**:
- **Machine**: NVIDIA H200 NVL
- **Precision**: FP8 (blockwise recipe)
- **Wandb Name**: llama3x-8b-fp8-tp1_blockwise_machineNVL

**FP8 Configuration**:
```bash
--fp8-format hybrid
--fp8-amax-history-len 1024
--fp8-amax-compute-algo max
--fp8-param-gather
--fp8-recipe blockwise  # Blockwise recipe (unsupported)
```

Note attention has to be`fused`
```bash
    --attention-backend fused
```

**Results**:
- **Runtime**: 1hr 38m
- **Min Loss**: ~6.69
- **Log File**: NA




**Wandb Metrics**:
**Training Loss Plot:**
<img src="./assets/train_llama3_8b_h200_fp8_blockwise_itrtime.png" width="600" alt="FP8 Blockwise Training Loss">

**Iteration Time Analysis:**
<img src="./assets/train_llama3_8b_h200_fp8_blockwise_loss.png" width="600" alt="FP8 Blockwise Iteration Time">
---

## Performance Analysis

### Loss Comparison Summary
| Configuration | Min Loss | Runtime | itr time | Notes |
|---------------|----------|---------|-------|
| BF16 (8B) | ~6.4 | 1hr 56m | 12.9 Baseline |
| FP8 Delayed (8B) | ~5.9 | 1hr 56m | 9.5 | Best loss |
| FP8 Tensorwise (8B) | ~5.9 | 1hr 56m | 9.5 | Same as delayed |
| FP8 Tensorwise + 1F1L (8B) | ~6.0 | 9.5 | 1hr 56m | Slightly worse |
| FP8 Blockwise (8B) | 6.69 | 1hr 38m | 12.4 | slower than delayed/cs |


---

## Issues and Troubleshooting

### 1. Evaluation Memory Surge
**Problem**: All FP8 runs failed at evaluation intervals
**Solution**: short term solution, not recommended, Disabled evaluation by commenting out:
```bash
# --eval-iters 32
# --eval-interval 100
```

### 2. Blockwise Recipe Compatibility
**Problem**: Blockwise FP8 recipe not supported
**Error**: `ValueError: Float8CurrentScaling, MXFP8BlockScaling, Float8BlockwiseScaling and DelayedScaling are the only supported FP8 recipes`
**Solution**: Requires Transformer Engine update

### 4. Small Model FP8 Overhead
**Observation**: FP8 overhead exceeds benefits for 1B models
**Reason**: Matmul size too small to benefit from FP8 acceleration

### 5. TRITON_CACHE_DIR Export Issue
**Problem**: "Command not found" error related to TRITON_CACHE_DIR
**Solution**: Export the `TRITON_CACHE_DIR` to the desired location with enough space.
```bash
TRITON_CACHE_DIR=/path/to/workspace/.triton
```

### 6. Tmux Environment Setup
**Problem**: if default directory of .triton do not have enough space, export the TRITON_CACHE_DIR to new location.

### 7. ValueError: Float8CurrentScaling ....
**OLD Results**: Resolved by buiding the new image, refer `blockwise_scaling_setup.md`
- **Status**: ❌ Failed
- **Error**: `ValueError: Float8CurrentScaling, MXFP8BlockScaling, Float8BlockwiseScaling and DelayedScaling are the only supported FP8 recipes. Please also make sure you are using a compatible TE version.`
- **Log File**: `./logs/train_llama3_8b_h200_fp8_blockwise_fail.log`

```bash
# Create triton cache directory
mkdir -p /path/to/workspace/.triton

# Set environment variables for csh/tcsh (if using)
setenv TRITON_CACHE_DIR /path/to/workspace/.triton

# For bash sessions
export TRITON_CACHE_DIR=/path/to/workspace/.triton
```

---

## File Structure Reference

### Configuration Files
- [`train_llama3_8b_h200_bf16.sh`](./train_llama3_8b_h200_bf16.sh) - BF16 baseline
- [`train_llama3_8b_h200_fp8.sh`](./train_llama3_8b_h200_fp8.sh) - FP8 delayed recipe
- [`train_llama3_8b_h200_fp8_cs.sh`](./train_llama3_8b_h200_fp8_cs.sh) - FP8 tensorwise recipe
- [`train_llama3_8b_h200_fp8_cs_1F1L.sh`](./train_llama3_8b_h200_fp8_cs_1F1L.sh) - FP8 tensorwise with BF16 first/last layers
- [`train_llama3_8b_h200_fp8_blockwise.sh`](./train_llama3_8b_h200_fp8_blockwise.sh) - FP8 blockwise recipe (failed)
- [`train_llama_1gpu_bf16_slimpajama.sh`](./train_llama_1gpu_bf16_slimpajama.sh) - Small model baseline

### Log Files
- [`./logs/train_llama3_8b_h200_bf16.log`](./logs/train_llama3_8b_h200_bf16.log)
- [`./logs/train_llama3_8b_h200_fp8.log`](./logs/train_llama3_8b_h200_fp8.log)
- [`./logs/train_llama3_8b_h200_fp8_cs.logs`](./logs/train_llama3_8b_h200_fp8_cs.logs)
- [`./logs/train_llama3_8b_h200_fp8_cs_1F1L`](./logs/train_llama3_8b_h200_fp8_cs_1F1L)
- [`./logs/train_llama3_8b_h200_fp8_blockwise_fail.log`](./logs/train_llama3_8b_h200_fp8_blockwise_fail.log)
- [`./logs/train_llama3_8b_h200_fp8_eval_fail.log`](./logs/train_llama3_8b_h200_fp8_eval_fail.log)

---

## Future Work

1. Update Transformer Engine to support blockwise FP8
2. Investigate MXFP8
4. Explore more sophisticated layer-wise precision strategies
5. Extend analysis to larger model sizes and multi-node setups

---