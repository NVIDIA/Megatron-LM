# =========================
# Build the image
# =========================
# nvidia-docker build --target base -f GB200.Dockerfile --tag megatron-moe-scripts:mcore-moe-pytorch25.09-te2.9.0-cudnn9.14 --network host .

FROM nvcr.io/nvidia/pytorch:25.12-py3 AS base

ENV SHELL=/bin/bash

# =========================
# Install system packages
# =========================
RUN rm -rf /opt/megatron-lm && \
    apt-get update && \
    apt-get install -y sudo gdb bash-builtins git zsh autojump tmux curl gettext libfabric-dev && \
    wget https://github.com/mikefarah/yq/releases/download/v4.27.5/yq_linux_arm64 -O /usr/bin/yq && \
    chmod +x /usr/bin/yq

# =========================
# Install Python packages
# =========================
# NOTE: `unset PIP_CONSTRAINT` to install packages that do not meet the default constraint in the base image.
# Some package requirements and related versions are from 
#   https://github.com/NVIDIA/Megatron-LM/blob/core_v0.12.0/Dockerfile.linting.
#   https://github.com/NVIDIA/Megatron-LM/blob/core_v0.12.0/requirements_mlm.txt.
#   https://github.com/NVIDIA/Megatron-LM/blob/core_v0.12.0/requirements_ci.txt.
RUN unset PIP_CONSTRAINT && pip install --no-cache-dir debugpy dm-tree torch_tb_profiler einops wandb \
    sentencepiece tokenizers transformers==4.57.1 torchvision ftfy modelcards datasets tqdm pydantic \
    nvidia-pytriton py-spy yapf darker \
    tiktoken flask-restful \
    nltk wrapt pytest pytest_asyncio pytest-cov pytest_mock pytest-random-order \
    black==24.4.2 isort==5.13.2 flake8==7.1.0 pylint==3.2.6 coverage mypy \
    one-logger --index-url https://sc-hw-artf.nvidia.com/artifactory/api/pypi/hwinf-mlwfo-pypi/simple \
    setuptools==69.5.1 nvidia-cutlass-dsl==4.2.1

# =========================
# Install cudnn 9.14.0.64 for correct mxfp8 quantization and layernorm fusion
# =========================
RUN apt-get update && \
    wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/sbsa/cuda-keyring_1.1-1_all.deb && \
    dpkg -i cuda-keyring_1.1-1_all.deb && \
    apt-get update && \
    apt-get -y install libcudnn9-cuda-13

# =========================
# Update cublas for better performance
# =========================
# CUDA in the base container is enough.

# =========================
# Install latest TE
# Use a specific commit instead of main to make it more stable.
# This is based on release_v2.9 branch and contains some CPU and quantization optimizations.
# =========================
ARG COMMIT="99df881061ba6949081fdd8f00dccd0f617c6594"
ARG TE="git+https://github.com/NVIDIA/TransformerEngine.git@${COMMIT}"
RUN pip install nvidia-mathdx==25.1.1 && \
    unset PIP_CONSTRAINT && \
    NVTE_CUDA_ARCHS="100a;103a" NVTE_BUILD_THREADS_PER_JOB=8 NVTE_FRAMEWORK=pytorch pip install --no-build-isolation --no-cache-dir $TE

# =========================
# Install HybridEP
# =========================
WORKDIR /home/
RUN git clone --branch hybrid-ep https://github.com/deepseek-ai/DeepEP.git && \
    cd DeepEP && git checkout 3d4a884b8bc4e63786b6d7012f7767fe3c91cfb5 && \
    TORCH_CUDA_ARCH_LIST="10.0" pip install --no-build-isolation .

# =========================
# Clean cache
# =========================
RUN rm -rf /root/.cache /tmp/*

# =========================
# Publish the image
# =========================
# docker push url/[repo]:[tag]
