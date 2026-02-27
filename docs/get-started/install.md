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

- **Python**: >= 3.10 (3.12 recommended)
- **PyTorch**: >= 2.6.0
- **CUDA Toolkit**: Latest stable version


## Option A: Pip Install (Recommended)

Install the latest stable release from PyPI:

```bash
pip install megatron-core
```

To include optional training dependencies (Weights & Biases, SentencePiece, HF Transformers):

```bash
pip install megatron-core[mlm]
```

For all extras including [Transformer Engine](https://github.com/NVIDIA/TransformerEngine):

```bash
pip install "setuptools<80.0.0,>=77.0.0" "packaging>=24.2" pybind11 "torch>=2.6.0"
pip install --no-build-isolation megatron-core[mlm,dev]
```

```{note}
`--no-build-isolation` requires build dependencies to be pre-installed in the environment. `torch` is needed because several `[dev]` packages (`mamba-ssm`, `nv-grouped-gemm`, `transformer-engine`) import it at build time to compile CUDA kernels. Expect this step to take **20+ minutes** depending on your hardware. If you prefer pre-built binaries, the [NGC Container](#option-c-ngc-container) ships with these pre-compiled.
```

To clone the repository for examples:

```bash
git clone https://github.com/NVIDIA/Megatron-LM.git
```


## Option B: Install from Source

For development or to run the latest unreleased code:

```bash
git clone https://github.com/NVIDIA/Megatron-LM.git
cd Megatron-LM
pip install "setuptools<80.0.0,>=77.0.0" "packaging>=24.2" pybind11 "torch>=2.6.0"
pip install --no-build-isolation -e .[mlm,dev]
```


## Option C: NGC Container

For a pre-configured environment with all dependencies pre-installed (PyTorch, CUDA, cuDNN, NCCL, Transformer Engine), use the [PyTorch NGC Container](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/pytorch).

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

Then install Megatron Core inside the container (torch is already available in the NGC image):

```bash
pip install --no-build-isolation megatron-core[mlm,dev]
```


You are now ready to run training. See [Your First Training Run](quickstart.md) for next steps.
