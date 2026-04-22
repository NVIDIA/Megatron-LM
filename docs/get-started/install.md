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


## Prerequisites

Install [uv](https://docs.astral.sh/uv/), a fast Python package installer:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```


## Option A: Pip Install (Recommended)

Install the latest stable release from PyPI:

```bash
uv pip install megatron-core
```

To include optional training dependencies (Weights & Biases, SentencePiece, HF Transformers):

```bash
uv pip install "megatron-core[training]"
```

For all extras including [Transformer Engine](https://github.com/NVIDIA/TransformerEngine):

```bash
uv pip install --group build
uv pip install --no-build-isolation "megatron-core[training,dev]"
```

```{note}
`--no-build-isolation` requires build dependencies to be pre-installed in the environment. `torch` is needed because several `[dev]` packages (`mamba-ssm`, `nv-grouped-gemm`, `transformer-engine`) import it at build time to compile CUDA kernels. Expect this step to take **20+ minutes** depending on your hardware. If you prefer pre-built binaries, the [NGC Container](#option-c-ngc-container) ships with these pre-compiled.
```

```{warning}
Building from source can consume a large amount of memory. By default the build runs one compiler job per CPU core, which can cause out-of-memory failures on machines with many cores. To limit parallel compilation jobs, set the `MAX_JOBS` environment variable before installing (for example, `MAX_JOBS=4`).
```

```{tip}
For a lighter set of development dependencies without Transformer Engine and ModelOpt, use `[lts]` instead of `[dev]`: `uv pip install --no-build-isolation "megatron-core[training,lts]"`. The `[lts]` and `[dev]` extras are mutually exclusive.
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
uv pip install -e .
```

To install with all development dependencies (includes Transformer Engine, requires pre-installed build deps):

```bash
uv pip install --group build
uv pip install --no-build-isolation -e ".[training,dev]"
```

```{tip}
If the build runs out of memory, limit parallel compilation jobs with `MAX_JOBS=4 uv pip install --no-build-isolation -e ".[training,dev]"`.
```


## Option C: NGC Container

For a pre-configured environment with all dependencies pre-installed (PyTorch, CUDA, cuDNN, NCCL, Transformer Engine), use the [PyTorch NGC Container](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/pytorch).

We recommend using the **previous month's** NGC container rather than the latest one to ensure compatibility with the current Megatron Core release and testing matrix.

```bash
docker run --gpus all -it --rm \
  -v /path/to/dataset:/workspace/dataset \
  -v /path/to/checkpoints:/workspace/checkpoints \
  -e PIP_CONSTRAINT= \
  nvcr.io/nvidia/pytorch:26.01-py3
```

```{note}
The NGC PyTorch container constrains the Python environment globally using `PIP_CONSTRAINT`. The `-e PIP_CONSTRAINT=` flag above unsets this so that Megatron Core and its dependencies install correctly.
```

Then install Megatron Core inside the container (torch is already available in the NGC image):

```bash
pip install uv
uv pip install --no-build-isolation "megatron-core[training,dev]"
```

**Important:** The `megatron-core` pip package does not install the example and training scripts from this repository. The `[training]` extra, and deprecated `[mlm]` alias, add Python dependencies but do **not** provide the full `Megatron-LM` source tree. To run repository training scripts, see [Running Training Scripts](#running-training-scripts) below.

You are now ready to run training. Refer to [Your First Training Run](quickstart.md) for next steps.

## Running Training Scripts

To run training scripts from this repository, clone the source tree and install it in editable mode with the required extras:

```bash
git clone https://github.com/NVIDIA/Megatron-LM.git
cd Megatron-LM
uv pip install --group build
uv pip install --no-build-isolation -e ".[training,dev]"
python examples/gpt3/train_gpt3.py ...
```

```{tip}
`[mlm]` is a deprecated alias for `[training]`. Prefer `[training]` in new environments and scripts.
```
