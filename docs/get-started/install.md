<!---
   Copyright (c) 2022-2026, NVIDIA CORPORATION. All rights reserved.
   NVIDIA CORPORATION and its licensors retain all intellectual property
   and proprietary rights in and to this software, related documentation
   and any modifications thereto. Any use, reproduction, disclosure or
   distribution of this software and related documentation without an express
   license agreement from NVIDIA CORPORATION is strictly prohibited.
-->

# Installation

## System Requirements

### Hardware

- **Recommended**: NVIDIA Turing architecture or later
- **FP8 Support**: Requires NVIDIA Hopper, Ada, or Blackwell GPUs

### Software

- **CUDA/cuDNN/NCCL**: Latest stable versions
- **PyTorch**: Latest stable version
- **Transformer Engine**: Latest stable version
- **Python**: 3.12 recommended


## Docker Installation (Recommended)

The [PyTorch NGC Container](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/pytorch) ships with all dependencies pre-installed (PyTorch, CUDA, cuDNN, NCCL, Transformer Engine) in versions that are tested and optimized for NVIDIA GPUs.

We recommend using the **previous month's** NGC container rather than the latest one to ensure compatibility with the current Megatron Core release and testing matrix.

1. Clone the repository:

    ```bash
    git clone https://github.com/NVIDIA/Megatron-LM.git
    ```

2. Launch the container with your clone mounted:

    ```bash
    docker run --gpus all -it --rm \
      -v /path/to/Megatron-LM:/workspace/megatron-lm \
      -v /path/to/dataset:/workspace/dataset \
      -v /path/to/checkpoints:/workspace/checkpoints \
      -e PIP_CONSTRAINT= \
      nvcr.io/nvidia/pytorch:26.01-py3
    ```

    ```{note}
    The NGC PyTorch container constrains the Python environment globally via `PIP_CONSTRAINT`. The `-e PIP_CONSTRAINT=` flag above unsets this so that Megatron Core and its dependencies install correctly.
    ```

3. Install Megatron Core from source inside the container:

    ```bash
    cd /workspace/megatron-lm
    pip install --no-build-isolation -e .
    ```

You are now ready to run training. See [Your First Training Run](quickstart.md) for next steps.


## Pip Installation

Pip installation requires you to provide all necessary dependencies (CUDA, cuDNN, NCCL, PyTorch, Transformer Engine) yourself. This path is for users who cannot use Docker or need a custom environment.

### Install Megatron Core Only

For a minimal install with only PyTorch as a dependency:

```bash
pip install megatron-core
```

### Install with Latest Dependencies (`dev`)

Tracks the most recent upstream dependency versions:

```bash
pip install "setuptools<80.0.0,>=77.0.0" "packaging>=24.2"
pip install --no-build-isolation megatron-core[dev]
```

To also include Megatron-LM training scripts and examples:

```bash
pip install "setuptools<80.0.0,>=77.0.0" "packaging>=24.2"
pip install --no-build-isolation megatron-core[mlm,dev]
```

### Install with LTS Dependencies

Long-term support pinned to NGC PyTorch 24.01:

```bash
pip install "setuptools<80.0.0,>=77.0.0" "packaging>=24.2"
pip install --no-build-isolation megatron-core[lts]
```

To also include Megatron-LM training scripts and examples:

```bash
pip install "setuptools<80.0.0,>=77.0.0" "packaging>=24.2"
pip install --no-build-isolation megatron-core[mlm,lts]
```

### Install from Source

To install from source (useful for development or running examples directly):

```bash
git clone https://github.com/NVIDIA/Megatron-LM.git
cd Megatron-LM
pip install --no-build-isolation -e .[mlm,dev]
```
