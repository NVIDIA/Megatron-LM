FROM nvcr.io/nvidia/pytorch:24.01-py3

RUN pip uninstall -y triton && \
    pip install triton==2.1.0 sentencepiece==0.1.99 flask-restful

# The causal-conv1d and mamba-ssm packages below are built from scratch here
# (which takes significant time) because there are no wheels available on PyPI
# for these relatively newer versions of the packages that are compatible with
# the older NGC-variant PyTorch version (e.g. version 2.2.0.dev231106) that we
# are using (in the NGC base container). Generally, if the package is not
# compatible with the PyTorch version, then it will generate a Python import
# error. The package authors tend to only release wheels for new versions of
# these pacakges which are compatible with the versions of regular PyTorch and
# NGC-variant PyTorch that are newer at the time of release. So, to use newer
# versions of these packages with relatively older versions of the NGC PyTorch
# container, we tend to have to build the packages from scratch.

RUN cd /tmp && \
    git clone https://github.com/Dao-AILab/causal-conv1d.git && \
    cd causal-conv1d && \
    git checkout v1.2.2.post1 && \
    CAUSAL_CONV1D_FORCE_BUILD=TRUE pip install . && \
    cd .. && \
    rm -rf causal-conv1d

RUN cd /tmp && \
    git clone https://github.com/state-spaces/mamba.git && \
    cd mamba && \
    git checkout v2.0.3 && \
    MAMBA_FORCE_BUILD=TRUE pip install . && \
    cd .. && \
    rm -rf mamba
