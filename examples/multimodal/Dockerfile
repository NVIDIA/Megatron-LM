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
ARG COMMIT="1c702b4cff6fcee1b92857c283e7c3eb20534923"
ARG TE="git+https://github.com/NVIDIA/TransformerEngine.git@${COMMIT}"
RUN unset PIP_CONSTRAINT && NVTE_FRAMEWORK=pytorch pip install --no-cache-dir --no-build-isolation $TE
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
# :$CPATH
ENV CPATH=/usr/local/mpi/include
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
# Install mamba and causal-conv1d
# Mamba dependencies.
RUN pip uninstall -y triton && pip install "triton<3.4.0"
RUN unset PIP_CONSTRAINT && cd /tmp && \
    git clone https://github.com/Dao-AILab/causal-conv1d.git && \
    cd causal-conv1d && \
    CAUSAL_CONV1D_FORCE_BUILD=TRUE pip install . && \
    cd .. && \
    rm -rf causal-conv1d

COPY ./mamba /tmp/mamba
RUN unset PIP_CONSTRAINT && cd /tmp/mamba && \
    MAMBA_FORCE_BUILD=TRUE pip install . && \
    cd .. && \
    rm -rf mamba


# Install miscellaneous packages necessary for pre-training and inference.
RUN pip install boto3 einops flash-attn sentencepiece tiktoken wandb flask flask-restful

# Install Onelogger.
RUN pip install --index-url=https://sc-hw-artf.nvidia.com/artifactory/api/pypi/hwinf-mlwfo-pypi/simple one-logger one-logger-utils

# For debugging.
RUN apt-get update && \
   apt-get install -y gdb strace && \
   pip install py-spy && \
   pip install pynvml

RUN apt update && \
    apt -y upgrade && \
    apt install -y --no-install-recommends \
        software-properties-common \
        build-essential \
        python3-pip \
        python3-dev \
        bash \
        git \
        vim \
        python-is-python3 \
        default-jre \
        net-tools \
        wget \
        curl \
        rsync \
        zip \
        unzip \
        htop \
        tmux \
        bmon

RUN pip install einops einops-exts sentencepiece braceexpand webdataset packaging
RUN pip install transformers datasets accelerate timm
RUN pip install pytest-cov pytest_mock nltk wrapt
RUN pip install black isort pylint mypy click
RUN pip install mistral-common tiktoken
RUN pip install git+https://github.com/openai/CLIP.git
RUN pip install fairscale fire blobfile
# Use --no-deps for the following to avoid outdated and unnecessary dependencies.
RUN pip install mmf --no-deps
RUN pip install open_clip_torch open-flamingo[eval] --no-deps
RUN pip install zarr "tensorstore==0.1.45"
RUN pip install git+https://github.com/NVIDIA/Megatron-Energon.git#egg=megatron-energon[av_decode]
RUN pip install -U nemo_toolkit['asr']
RUN pip install librosa albumentations timm
RUN apt-get install -y git-lfs
RUN pip uninstall --yes multi-storage-client && pip install multi-storage-client==0.20.3

WORKDIR /home/