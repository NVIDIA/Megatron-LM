ARG BASE_DOCKER=rocm/pytorch:latest
FROM $BASE_DOCKER
ENV WORKSPACE_DIR=/workspace
ENV STAGE_DIR=/workspace/installs
RUN mkdir -p $WORKSPACE_DIR
RUN mkdir -p ${STAGE_DIR}
WORKDIR $WORKSPACE_DIR

  
RUN pip3 install \
numpy==1.26.4 \
scipy \
einops \
flask-restful \
nltk \
pytest \
pytest-cov \
pytest_mock \
pytest-csv \
pytest-random-order \
sentencepiece \
wrapt \
zarr \
wandb \
tensorstore==0.1.45 \
pytest_mock \
pybind11 \
wrapt \ 
setuptools==69.5.1 \
datasets \
tiktoken \
pynvml

RUN pip3 install "huggingface_hub[cli]" 
RUN python3 -m nltk.downloader punkt_tab


# Install Causal-Conv1d and its dependencies
WORKDIR ${STAGE_DIR}
ENV CAUSAL_CONV1D_FORCE_BUILD=TRUE
ENV MAMBA_FORCE_BUILD=TRUE
ENV HIP_ARCHITECTURES="gfx942"
RUN git clone https://github.com/Dao-AILab/causal-conv1d causal-conv1d &&\
    cd causal-conv1d &&\
    git show --oneline -s &&\
    pip install .

# Install mamba
WORKDIR ${STAGE_DIR}
RUN git clone https://github.com/state-spaces/mamba mamba &&\
    cd mamba &&\
    git checkout bc84fb1 &&\
    git show --oneline -s &&\
    pip install --no-build-isolation .

# Clone TE repo and submodules
WORKDIR ${STAGE_DIR}
ENV NVTE_FRAMEWORK=pytorch 
ENV PYTORCH_ROCM_ARCH=gfx942
ENV NVTE_USE_HIPBLASLT=1
RUN git clone --recursive https://github.com/ROCmSoftwarePlatform/TransformerEngine-private.git &&\
    cd TransformerEngine-private &&\
    pip install .

WORKDIR $WORKSPACE_DIR
RUN git clone https://github.com/ROCm/Megatron-LM.git Megatron-LM &&\
    cd Megatron-LM &&\
    git checkout hakiymaz/deepseek_v2 &&\
    pip install -e .

WORKDIR $WORKSPACE_DIR/Megatron-LM

# record configuration for posterity
RUN pip list