# GPT-OSS Training Tutorial

## Setup

```bash
# Run this from the Megatron-LM root directory
docker run --runtime=nvidia --gpus all -it --rm \
  -v $(pwd):/workspace/megatron \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  -w /workspace/megatron \
  nvcr.io/nvidia/pytorch:25.12-py3 \
  bash
```

## Step 1: Convert HuggingFace to Megatron (Optional)

Reference: [Megatron-Bridge Dockerfile](https://github.com/NVIDIA-NeMo/Megatron-Bridge/blob/main/docker/Dockerfile.ci)

### Install megatron bridge

Inside the pytorch container,
```
cd /opt
git clone --recursive https://github.com/NVIDIA-NeMo/Megatron-Bridge.git
cd Megatron-Bridge

# Make sure submodules are initialized (for 3rdparty/Megatron-LM)
git submodule update --init --recursive

export PATH="/root/.local/bin:$PATH"
export UV_PROJECT_ENVIRONMENT=/opt/venv
export VIRTUAL_ENV=/opt/venv
export PATH="$UV_PROJECT_ENVIRONMENT/bin:$PATH"
export UV_LINK_MODE=copy
export UV_VERSION="0.7.2"

# Install UV
curl -LsSf https://astral.sh/uv/${UV_VERSION}/install.sh | sh

# Create virtual environment and build the package
uv venv ${UV_PROJECT_ENVIRONMENT} --system-site-packages

uv sync --locked --only-group build
uv sync --locked --link-mode copy --all-extras --all-groups

uv cache prune
```

### Run the conversion script

Set `--nproc-per-node` to be the number of GPUs per node. Set `hf_model_name` to be the Huggingface model. E.g. `openai/gpt-oss-20b`

```bash
torchrun --nproc-per-node=8 examples/gptoss/01_convert_hf.py <hf_model_name>
```

## Step 2: Pretrain from Scratch

```bash
torchrun --nproc-per-node=8 examples/gptoss/02_pretrain.py
```

To load the converted checkpoint, uncomment the `pretrained_checkpoint` line in `02_pretrain.py`.