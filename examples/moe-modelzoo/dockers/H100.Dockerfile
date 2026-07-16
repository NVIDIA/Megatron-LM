# =========================
# Build the image
# =========================
# nvidia-docker build --target base -f dockers/Dockerfile --tag gitlab-master.nvidia.com:5005/arch_moe_exploration/megatron-moe-scripts:mcore-moe-pytorch25.06-te2.6.0.dev0-cudnn9.11.0.92 --network host .

# =========================
# Use the pytorch image which has CUDA 12.9 inside.
# =========================
FROM nvcr.io/nvidia/pytorch:25.06-py3 AS base

ENV SHELL=/bin/bash

# =========================
# Install system packages
# =========================
RUN rm -rf /opt/megatron-lm && \
    apt-get update && \
    apt-get install -y sudo gdb pstack bash-builtins git zsh autojump tmux curl gettext libfabric-dev && \
    wget https://github.com/mikefarah/yq/releases/download/v4.27.5/yq_linux_amd64 -O /usr/bin/yq && \
    chmod +x /usr/bin/yq

# =========================
# Install Python packages
# =========================
# NOTE: `unset PIP_CONSTRAINT` to install packages that do not meet the default constraint in the base image.
# Some package requirements and related versions are from 
#   https://github.com/NVIDIA/Megatron-LM/blob/core_v0.12.0/Dockerfile.linting.
#   https://github.com/NVIDIA/Megatron-LM/blob/core_v0.12.0/requirements_mlm.txt.
#   https://github.com/NVIDIA/Megatron-LM/blob/core_v0.12.0/requirements_ci.txt.
RUN unset PIP_CONSTRAINT && pip install debugpy dm-tree torch_tb_profiler einops wandb \
    sentencepiece tokenizers transformers torchvision ftfy modelcards datasets tqdm pydantic \
    nvidia-pytriton py-spy yapf darker \
    tiktoken flask-restful \
    nltk wrapt pytest pytest_asyncio pytest-cov pytest_mock pytest-random-order \
    black==24.4.2 isort==5.13.2 flake8==7.1.0 pylint==3.2.6 coverage mypy \
    one-logger --index-url https://sc-hw-artf.nvidia.com/artifactory/api/pypi/hwinf-mlwfo-pypi/simple \
    setuptools==69.5.1

# =========================
# Install cudnn because the latest build contains optimizations for MLA attention
# =========================
# fwd: https://gitlab-master.nvidia.com/cudnn/cudnn/-/merge_requests/329
# bwd: https://gitlab-master.nvidia.com/cudnn/cudnn/-/merge_requests/417
#      https://gitlab-master.nvidia.com/cudnn/cudnn/-/merge_requests/540
#      https://gitlab-master.nvidia.com/cudnn/cudnn/-/merge_requests/640 (claim to be included in 9.11.0.24)
# The latest build is in https://urm.nvidia.com/artifactory/hw-cudnn-generic/CUDNN/v9.11_cuda_12.9/9.11.0.92/debug_cudnn-linux-x86_64-9.11.0.92.tar.gz
# NOTE: Download (wget --user [username] --ask-password [url]) and extract (tar -xvf *.tar.gz) the package to ./cudnn/cudnn[version]
COPY ./cudnn/cudnn9.11.0.92/include/ /usr/include/
COPY ./cudnn/cudnn9.11.0.92/include/ /usr/include/x86_64-linux-gnu/
COPY ./cudnn/cudnn9.11.0.92/lib64/ /usr/local/cuda/targets/x86_64-linux/lib/


# =========================
# Update cublas for better performance
# =========================
# CUDA 12.9u1 has been included in the base container.

# =========================
# Install grouped_gemm
# =========================
RUN TORCH_CUDA_ARCH_LIST="8.0 9.0 10.0" pip install git+https://github.com/fanshiqing/grouped_gemm@v1.1.4

# =========================
# Install latest TE
# Use a specific commit instead of main to make it more stable.
# =========================
ARG COMMIT="07afda98585511ede420971ca8223a8ac2c07d1a"
ARG TE="git+https://github.com/NVIDIA/TransformerEngine.git@${COMMIT}"
RUN unset PIP_CONSTRAINT && NVTE_CUDA_ARCHS="80;90;100" NVTE_BUILD_THREADS_PER_JOB=8 NVTE_FRAMEWORK=pytorch pip install --no-cache-dir --no-build-isolation $TE

# =========================
# Install DeepEP
# =========================
# the dependency of IBGDA
RUN ln -s /usr/lib/x86_64-linux-gnu/libmlx5.so.1 /usr/lib/x86_64-linux-gnu/libmlx5.so

# Clone and build deepep and deepep-nvshmem
WORKDIR /home/dpsk_a2a
RUN git clone -b v2.3.1 https://github.com/NVIDIA/gdrcopy.git && \
    git clone https://github.com/deepseek-ai/DeepEP.git && cd DeepEP && git checkout a84a248 && \
    cd /home/dpsk_a2a && \
    wget https://developer.nvidia.com/downloads/assets/secure/nvshmem/nvshmem_src_3.2.5-1.txz && \
    tar -xvf nvshmem_src_3.2.5-1.txz && mv nvshmem_src deepep-nvshmem && \
    cd deepep-nvshmem && git apply /home/dpsk_a2a/DeepEP/third-party/nvshmem.patch && \
    sed -i '16i#include <getopt.h>' /home/dpsk_a2a/deepep-nvshmem/examples/moe_shuffle.cu

ENV CUDA_HOME=/usr/local/cuda
# Set MPI environment variables. Having errors when not set.
ENV CPATH=/usr/local/mpi/include:$CPATH
ENV LD_LIBRARY_PATH=/usr/local/mpi/lib:$LD_LIBRARY_PATH
ENV LD_LIBRARY_PATH=/usr/local/x86_64-linux-gnu:$LD_LIBRARY_PATH
ENV GDRCOPY_HOME=/home/dpsk_a2a/gdrcopy

# Build deepep-nvshmem
WORKDIR /home/dpsk_a2a/deepep-nvshmem
RUN NVSHMEM_SHMEM_SUPPORT=0 \
    NVSHMEM_UCX_SUPPORT=0 \
    NVSHMEM_USE_NCCL=0 \
    NVSHMEM_MPI_SUPPORT=0 \
    NVSHMEM_IBGDA_SUPPORT=1 \
    NVSHMEM_PMIX_SUPPORT=0 \
    NVSHMEM_TIMEOUT_DEVICE_POLLING=0 \
    NVSHMEM_USE_GDRCOPY=1 \
    cmake -S . -B build/ -DCMAKE_INSTALL_PREFIX=/home/dpsk_a2a/deepep-nvshmem/install -DCMAKE_CUDA_ARCHITECTURES=90 && cd build && make install -j

# Build deepep
WORKDIR /home/dpsk_a2a/DeepEP
ENV NVSHMEM_DIR=/home/dpsk_a2a/deepep-nvshmem/install
RUN NVSHMEM_DIR=/home/dpsk_a2a/deepep-nvshmem/install python setup.py develop && \
    NVSHMEM_DIR=/home/dpsk_a2a/deepep-nvshmem/install python setup.py install

# =========================
# Avoid record stream issue in NCCL. Only needed in the 25.04 container.
# =========================
# ENV TORCH_NCCL_AVOID_RECORD_STREAMS=0

# =========================
# Change the workspace
# =========================
WORKDIR /home/

# =========================
# Publish the image
# =========================
# docker push gitlab-master.nvidia.com:5005/[repo]:[tag]
