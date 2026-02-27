<!---
   Copyright (c) 2022-2026, NVIDIA CORPORATION. All rights reserved.
   NVIDIA CORPORATION and its licensors retain all intellectual property
   and proprietary rights in and to this software, related documentation
   and any modifications thereto. Any use, reproduction, disclosure or
   distribution of this software and related documentation without an express
   license agreement from NVIDIA CORPORATION is strictly prohibited.
-->

# Installation

We recommend installing Megatron Core inside the [PyTorch NGC Container](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/pytorch), which ships with all dependencies pre-installed (PyTorch, CUDA, cuDNN, NCCL, Transformer Engine) in versions that are tested and optimized for NVIDIA GPUs.

## System Requirements

### Hardware

- **Recommended**: NVIDIA Turing architecture or later
- **FP8 Support**: Requires NVIDIA Hopper, Ada, or Blackwell GPUs

### Software

- **CUDA/cuDNN/NCCL**: Latest stable versions
- **PyTorch**: Latest stable version
- **Transformer Engine**: Latest stable version
- **Python**: 3.12 recommended


## Step 1: Launch the NGC Container

We recommend using the **previous month's** NGC container rather than the latest one to ensure compatibility with the current Megatron Core release and testing matrix.

```bash
docker run --gpus all -it --rm \
  -v /path/to/dataset:/workspace/dataset \
  -v /path/to/checkpoints:/workspace/checkpoints \
  -e PIP_CONSTRAINT= \
  nvcr.io/nvidia/pytorch:25.09-py3
```

```{note}
The NGC PyTorch container constrains the Python environment globally via `PIP_CONSTRAINT`. The `-e PIP_CONSTRAINT=` flag above unsets this so that Megatron Core and its dependencies install correctly.
```


## Step 2: Install Megatron Core

### Option A: Pip Install (Recommended)

Install the latest stable release from PyPI:

```bash
pip install "setuptools<80.0.0,>=77.0.0" "packaging>=24.2"
pip install --no-build-isolation megatron-core[mlm,dev]
```

To clone the repository for examples:

```bash
git clone https://github.com/NVIDIA/Megatron-LM.git
```

### Option B: Install from Source

For development or to run the latest unreleased code:

```bash
git clone https://github.com/NVIDIA/Megatron-LM.git
cd Megatron-LM
pip install --no-build-isolation -e .[mlm,dev]
```

You are now ready to run training. See [Your First Training Run](quickstart.md) for next steps.
