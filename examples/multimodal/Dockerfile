FROM nvcr.io/nvidia/pytorch:24.02-py3

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
        default-jre

RUN pip install --upgrade pip
RUN pip install einops sentencepiece braceexpand webdataset
RUN pip install pytest-cov pytest_mock nltk wrapt
RUN pip install zarr "tensorstore==0.1.45"
RUN pip install git+https://github.com/fanshiqing/grouped_gemm@main
RUN pip install black==19.10b0 isort click==8.0.2
RUN pip install pycocoevalcap megatron-energon