FROM nvcr.io/nvidia/pytorch:25.06-py3

ARG PIP_CONSTRAINT=

WORKDIR /workspace/nmm-sandbox

RUN pip install jsonlines omegaconf pulp torchprofile
RUN pip install flask flask_restful fire nltk
RUN pip install tiktoken blobfile

RUN pip install datasets transformers

RUN pip install triton==3.3.1
RUN pip install git+https://github.com/state-spaces/mamba.git
RUN pip install git+https://github.com/Dao-AILab/causal-conv1d.git

RUN pip install -U nvidia-modelopt
