FROM nvcr.io/nvidia/pytorch:25.12-py3 AS base

# Build the image
# docker build --target deepep -f dockers/B200.dockerfile -t container_url --network host .
ENV SHELL=/bin/bash

RUN rm /opt/megatron-lm -rf && \
    apt-get update && \
    apt-get install -y sudo gdb pstack bash-builtins git zsh autojump tmux curl gettext && \
    wget https://github.com/mikefarah/yq/releases/download/v4.27.5/yq_linux_amd64 -O /usr/bin/yq && \
    chmod +x /usr/bin/yq && \
    unset PIP_CONSTRAINT && pip install debugpy dm-tree torch_tb_profiler einops wandb \
    sentencepiece tokenizers transformers==4.57.1 torchvision ftfy modelcards datasets tqdm pydantic \
    nvidia-pytriton py-spy yapf darker \
    tiktoken flask-restful \
    nltk wrapt pytest pytest_asyncio pytest-cov pytest_mock pytest-random-order \
    black==24.4.2 isort==5.13.2 flake8==7.1.0 pylint==3.2.6 coverage mypy \
    setuptools==69.5.1 nvidia-cutlass-dsl=4.3.5 apache-tvm-ffi torch-c-dlpack-ext

# =========================
# Install cudnn because the latest build contains optimizations for MLA attention
# =========================
RUN apt-get update && \
    wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb && \
    dpkg -i cuda-keyring_1.1-1_all.deb && \
    apt-get update && \
    apt-get -y install libcudnn9-cuda-13

# =========================
# Install latest TE
# Use a specific commit instead of main to make it more stable.
# =========================
ARG COMMIT="99df881061ba6949081fdd8f00dccd0f617c6594"
ARG TE="git+https://github.com/nvidia/TransformerEngine.git@${COMMIT}"
RUN unset PIP_CONSTRAINT && \
    NVTE_CUDA_ARCHS="100a;103a" NVTE_BUILD_THREADS_PER_JOB=8 NVTE_FRAMEWORK=pytorch pip install --no-build-isolation --no-cache-dir $TE


# =========================
# Option 2: Install DeepEP
# =========================
FROM base AS deepep
## the dependency of IBGDA
RUN ln -s /usr/lib/x86_64-linux-gnu/libmlx5.so.1 /usr/lib/x86_64-linux-gnu/libmlx5.so

## Clone and build deepep and deepep-nvshmem
ENV CPATH=${CUDA_HOME}/include/cccl:$CPATH
WORKDIR /workspace/dpsk_a2a
RUN git clone https://github.com/deepseek-ai/DeepEP.git ./deepep && \
    cd ./deepep && git checkout v1.2.1 && cd /workspace/dpsk_a2a && \
    wget https://developer.download.nvidia.com/compute/nvshmem/redist/libnvshmem/linux-x86_64/libnvshmem-linux-x86_64-3.4.5_cuda13-archive.tar.xz -O nvshmem_src.tar.xz && \
    tar -xvf nvshmem_src.tar.xz && mv libnvshmem-linux-x86_64-3.4.5_cuda13-archive deepep-nvshmem

ENV NVSHMEM_DIR=/workspace/dpsk_a2a/deepep-nvshmem/
ENV LD_LIBRARY_PATH=${NVSHMEM_DIR}/lib:$LD_LIBRARY_PATH
ENV PATH=${NVSHMEM_DIR}/bin:$PATH

## Build deepep
WORKDIR /workspace/dpsk_a2a/deepep
RUN TORCH_CUDA_ARCH_LIST="10.0" NVSHMEM_DIR=/workspace/dpsk_a2a/deepep-nvshmem MAX_JOBS=8 pip install --no-build-isolation .

# =========================
# Clean up
# =========================
RUN rm -rf /root/.cache /tmp/* /var/lib/apt/lists/*
WORKDIR /workspace/
