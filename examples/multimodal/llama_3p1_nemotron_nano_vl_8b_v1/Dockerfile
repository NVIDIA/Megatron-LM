FROM nvcr.io/nvidia/pytorch:25.04-py3

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

RUN pip install --upgrade pip
RUN git clone https://github.com/Dao-AILab/causal-conv1d.git && cd causal-conv1d && git checkout && CAUSAL_CONV1D_FORCE_BUILD=TRUE pip install . --no-build-isolation
RUN git clone https://github.com/state-spaces/mamba.git && cd mamba && git checkout && MAMBA_FORCE_BUILD=TRUE pip install . --no-build-isolation
RUN pip install numpy
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
