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
    pytest-random-order \
    sentencepiece \
    tiktoken \
    wrapt \
    zarr \
    wandb \
    tensorstore==0.1.45 \
    nvidia-modelopt[torch]>=0.19.0 \
    pynvml==11.5.3

pip3 install git+https://github.com/fanshiqing/grouped_gemm@v1.1.4
pip3 install git+https://github.com/state-spaces/mamba.git@v2.2.2
pip3 install git+https://github.com/Dao-AILab/causal-conv1d.git@v1.4.0

pip3 install git+https://github.com/Dao-AILab/flash-attention@v2.5.8
pip3 install git+https://github.com/NVIDIA/TransformerEngine.git@v1.9


