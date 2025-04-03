#!/bin/bash

pip3 install --upgrade pip
pip3 install packaging
pip3 install torch==2.5.0 torchaudio==2.5.0 torchvision --extra-index-url https://download.pytorch.org/whl/cu121

pip3 install --extra-index-url https://pypi.nvidia.com --no-cache-dir --upgrade-strategy only-if-needed -v \
    einops \
    flask-restful \
    nltk \
    pytest \
    pytest-cov \
    pytest_mock \
    pytest_asyncio \
    pytest-random-order \
    sentencepiece \
    tiktoken \
    wrapt \
    zarr \
    wandb \
    tensorstore==0.1.45 \
    pynvml==11.5.3 \
    triton==3.1.0

pip3 install git+https://github.com/ajayvohra2005/nvidia-resiliency-ext-x.git@e4b22cfb45d9e078b77242b68a35d9df4947dc91

