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

pip3 install git+https://github.com/state-spaces/mamba.git@v2.2.2
pip3 install git+https://github.com/Dao-AILab/causal-conv1d.git@v1.4.0
pip3 install git+https://github.com/ajayvohra2005/nvidia-resiliency-ext-x.git@87a2c60e494498525a163af598dacbd57c8528b0
pip3 install multi-storage-client==0.20.3
pip3 install pybind11==2.13.6
pip3 install transformers==4.47.1
pip3 install boto3==1.38.45
pip3 install absl-py==2.1.0

