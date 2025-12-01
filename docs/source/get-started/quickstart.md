# Quick Start

## Installation

### Docker (Recommended)

We strongly recommend using the previous releases of [PyTorch NGC Container](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/pytorch) rather than the latest one for optimal compatibility with Megatron Core release and testing. Our releases are always based on the previous month's NGC container, so this ensures compatibility and stability.

**Note:** The NGC PyTorch container constraints the python environment globally via `PIP_CONSTRAINT`. In the following examples we will unset the variable.

This container comes with all dependencies pre-installed with compatible versions and optimized configurations for NVIDIA GPUs:

- PyTorch (latest stable version)
- CUDA, cuDNN, NCCL (latest stable versions)
- Support for FP8 on NVIDIA Hopper, Ada, and Blackwell GPUs
- For best performance, use NVIDIA Turing GPU architecture generations and later

```bash
# Run container with mounted directories
docker run --runtime --nvidia --gpus all -it --rm \
  -v /path/to/megatron:/workspace/megatron \
  -v /path/to/dataset:/workspace/dataset \
  -v /path/to/checkpoints:/workspace/checkpoints \
  -e PIP_CONSTRAINT= \
  nvcr.io/nvidia/pytorch:25.04-py3
```

### Pip Installation

Megatron Core offers support for two NGC PyTorch containers:

- `dev`: Moving head that supports the most recent upstream dependencies
- `lts`: Long-term support of NGC PyTorch 24.01

Both containers can be combined with `mlm` which adds package dependencies for Megatron-LM on top of Megatron Core.

```bash
# Install the latest release dependencies
pip install "setuptools<80.0.0,>=77.0.0" "packaging>=24.2"
pip install --no-build-isolation megatron-core[dev]
# For running an M-LM application:
pip install "setuptools<80.0.0,>=77.0.0" "packaging>=24.2"
pip install --no-build-isolation megatron-core[mlm,dev]
```

```bash
# Install packages for LTS support NGC PyTorch 24.01
pip install "setuptools<80.0.0,>=77.0.0" "packaging>=24.2"
pip install --no-build-isolation megatron-core[lts]
# For running an M-LM application:
pip install "setuptools<80.0.0,>=77.0.0" "packaging>=24.2"
pip install --no-build-isolation megatron-core[mlm,lts]
```

For a version of Megatron Core with only torch, run:

```bash
pip install megatron-core
```

## System Requirements

### Hardware Requirements

- **FP8 Support**: NVIDIA Hopper, Ada, Blackwell GPUs
- **Recommended**: NVIDIA Turing architecture or later

### Software Requirements

- **CUDA/cuDNN/NCCL**: Latest stable versions
- **PyTorch**: Latest stable version
- **Transformer Engine**: Latest stable version
- **Python**: 3.12 recommended

## Your First Training Run

### Simple Training Example

```bash
# Distributed training example (2 GPUs, mock data)
torchrun --nproc_per_node=2 examples/run_simple_mcore_train_loop.py
```

### LLaMA-3 Training Example

```bash
# 8 GPUs, FP8 precision, mock data
./examples/llama/train_llama3_8b_fp8.sh
```

## Data Preparation

### JSONL Data Format

```json
{"text": "Your training text here..."}
{"text": "Another training sample..."}
```

### Basic Preprocessing

```bash
python tools/preprocess_data.py \
    --input data.jsonl \
    --output-prefix processed_data \
    --tokenizer-type HuggingFaceTokenizer \
    --tokenizer-model /path/to/tokenizer.model \
    --workers 8 \
    --append-eod
```

### Key Arguments

- `--input`: Path to input JSON/JSONL file
- `--output-prefix`: Prefix for output binary files (.bin and .idx)
- `--tokenizer-type`: Tokenizer type (`HuggingFaceTokenizer`, `GPT2BPETokenizer`, etc.)
- `--tokenizer-model`: Path to tokenizer model file
- `--workers`: Number of parallel workers for processing
- `--append-eod`: Add end-of-document token

## Next Steps

- Explore [Parallelism Strategies](../user-guide/parallelism/index.rst) to scale your training
- Learn about [Performance Optimization](../user-guide/performance/index.rst) techniques
- Check out [Features](../user-guide/features/index.rst) for advanced capabilities
