# Blockwise Scaling Setup Guide

This document provides a comprehensive guide for setting up blockwise scaling with TransformerEngine on NVIDIA H200 clusters, including troubleshooting common installation issues.

## Overview

Blockwise scaling is an advanced technique for training large language models that involves:
- **Blockwise quantization**: Dividing tensors into blocks for more efficient computation
- **Mixed precision training**: Using FP8 for compute-intensive operations while maintaining numerical stability
- **Memory optimization**: Reducing memory footprint while preserving model accuracy

## Prerequisites

### Hardware Requirements
- NVIDIA H200 GPU(s)
- CUDA 12.9+ support
- Sufficient system memory (32GB+ recommended)

### Software Requirements
- NVIDIA PyTorch container (nvcr.io/nvidia/pytorch:25.04-py3)
- CUDA 12.9 toolkit
- Python 3.12+
- Git

## Container Setup

### 1. Launch Container with H200 Support

```bash
crun -q 'gpu.product_name="NVIDIA H200"' --gpus=1 --cpu-arch-agnostic -i \
  -img nvcr.io/nvidia/pytorch:25.04-py3 \
  -a="--shm-size=8g -v /path/to/workspace:/path/to/workspace"
```

### 2. Container Environment Verification

Once inside the container, verify the environment:

```bash
# Check Python and PyTorch
python3 --version
python3 -c "import torch; print(f'PyTorch version: {torch.__version__}')"

# Check CUDA availability
python3 -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
python3 -c "import torch; print(f'CUDA device count: {torch.cuda.device_count()}')"
```

## TransformerEngine Installation

### Step 1: Clone Repository

```bash
cd /path/to/workspace
git clone https://github.com/NVIDIA/TransformerEngine.git
cd TransformerEngine
```

### Step 2: Checkout Specific Commit

```bash
git checkout f966d5f7f0136b611c2a00ac71c98000e21b679c
```

### Step 3: Update Submodules

```bash
git submodule update --init --recursive
```

### Step 4: Set Up CUDA Environment

**Critical**: The container may not have CUDA tools in PATH by default.

```bash
# Find CUDA installation
ls -la /usr/local/cuda*

# Set up CUDA environment variables
export PATH=/usr/local/cuda-12.9/bin:$PATH
export CUDACXX=/usr/local/cuda-12.9/bin/nvcc

# Verify nvcc is available
nvcc --version
```

**⚠️ Important**: These environment variables must be set in **every new container session**. If you restart the container or open a new terminal, you'll need to run these export commands again.

### Step 5: Install TransformerEngine

**Important**: Use the container's Python, not a virtual environment, to access pre-installed PyTorch.

```bash
# Clean any previous build artifacts
rm -rf build

# Set CMake variables to avoid RPATH issues
export CMAKE_BUILD_WITH_INSTALL_RPATH=TRUE
export CMAKE_GENERATOR="Unix Makefiles"

# Install with proper environment variables
unset PIP_CONSTRAINT
NVTE_FRAMEWORK=pytorch \
NVTE_CUDA_ARCHS="80;90" \
NVTE_BUILD_THREADS_PER_JOB=8 \
python3 -m pip install -e . -v
```

## Common Issues and Solutions

### Issue 1: CUDA Compiler Not Found

**Error**: `No CMAKE_CUDA_COMPILER could be found`

**Solution**:
```bash
export PATH=/usr/local/cuda-12.9/bin:$PATH
export CUDACXX=/usr/local/cuda-12.9/bin/nvcc
```

### Issue 2: Externally Managed Environment

**Error**: `externally-managed-environment`

**Solution**: Use the container's Python directly instead of creating a virtual environment:
```bash
# Don't use: python3 -m venv env
# Instead, use container's Python directly
python3 -m pip install -e .
```

### Issue 3: CMake RPATH Error with Ninja

**Error**: `The install of the transformer_engine target requires changing an RPATH`

**Solution**: Use Unix Makefiles generator instead of Ninja:
```bash
export CMAKE_BUILD_WITH_INSTALL_RPATH=TRUE
export CMAKE_GENERATOR="Unix Makefiles"
rm -rf build  # Clean previous build
```

### Issue 4: Permission Denied

**Error**: `Permission denied` when cloning to `/opt`

**Solution**: Clone to user-writable directory:
```bash
cd /path/to/workspace
git clone https://github.com/NVIDIA/TransformerEngine.git
```

## Verification

### Test Installation

```bash
python3 -c "
import transformer_engine as te
import torch
print('TransformerEngine imported successfully')
print(f'TE version: {te.__version__}')
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
"
```

### Test FP8 Operations

```bash
python3 -c "
import torch
import transformer_engine as te
from transformer_engine.common import recipe

# Create test tensors
x = torch.randn(32, 512, 1024, device='cuda')
y = torch.randn(32, 512, 1024, device='cuda')

# Test FP8 linear layer
fp8_recipe = recipe.DelayedScaling()
linear = te.Linear(1024, 1024, bias=False)
linear = linear.to(dtype=torch.float16)

with te.fp8_autocast(enabled=True, fp8_recipe=fp8_recipe):
    output = linear(x)
    print('FP8 linear layer test passed')
"
```

## Blockwise Scaling Configuration

### Environment Variables

```bash
# Set these for optimal blockwise scaling performance
export NVTE_FRAMEWORK=pytorch
export NVTE_CUDA_ARCHS="80;90"  # H200 and H200 architectures
export NVTE_BUILD_THREADS_PER_JOB=8
export CMAKE_BUILD_WITH_INSTALL_RPATH=TRUE
export CMAKE_GENERATOR="Unix Makefiles"
```

### Training Script Configuration

When using blockwise scaling in your training scripts:

```python
import transformer_engine as te
from transformer_engine.common import recipe

# Configure FP8 recipe for blockwise scaling
fp8_recipe = recipe.DelayedScaling(
    margin=0,
    interval=1,
    fp8_format=recipe.Format.E4M3
)

# Use in training loop
with te.fp8_autocast(enabled=True, fp8_recipe=fp8_recipe):
    # Your training code here
    pass
```