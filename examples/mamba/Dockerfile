FROM nvcr.io/nvidia/pytorch:23.12-py3

RUN pip uninstall -y causal-conv1d triton && \
    pip install causal-conv1d==1.2.2.post1 sentencepiece==0.1.99 triton==2.1.0 flask-restful

WORKDIR /tmp

RUN git clone https://github.com/state-spaces/mamba.git && \
    cd mamba && \
    git checkout v2.0.3 && \
    python setup.py install && \
    cd .. && \
    rm -rf mamba

