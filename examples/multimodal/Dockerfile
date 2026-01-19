# Base image: NVIDIA PyTorch container with CUDA, cuDNN, NCCL, Python, and uv pre-installed
FROM nvcr.io/nvidia/pytorch:25.11-py3

# Install JRE for pycocoevalcap
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        default-jre && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Using --break-system-packages to allow system-wide installation in managed environment
RUN uv pip install --system --no-cache --break-system-packages \
    einops \
    einops-exts \
    sentencepiece \
    braceexpand \
    webdataset \
    packaging \
    transformers \
    datasets \
    accelerate \
    timm \
    pytest-cov \
    pytest_mock \
    nltk \
    wrapt \
    zarr \
    tensorstore \
    black \
    isort \
    click \
    pycocoevalcap \
    megatron-energon \
    mistral-common \
    tiktoken \
    # Additional dependencies for megatron-core[mlm]
    flask-restful \
    wandb

# Install CLIP from GitHub
RUN uv pip install --system --no-cache --break-system-packages \
    git+https://github.com/openai/CLIP.git

# Install packages with --no-deps to avoid outdated and unnecessary dependencies
RUN uv pip install --system --no-cache --break-system-packages --no-deps \
    open_clip_torch \
    "open-flamingo[eval]"

# Copy Megatron-LM source and install megatron-core
# This assumes the build context is the Megatron-LM root directory
# Build with: docker build -t megatron-multimodal -f examples/multimodal/Dockerfile .
WORKDIR /workspace/megatron-lm
COPY . .

# Install megatron-core in editable mode for development
RUN uv pip install --system --no-cache --break-system-packages --no-build-isolation -e ".[mlm]"

# Set working directory to examples for convenience
WORKDIR /workspace/megatron-lm
