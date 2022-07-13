FROM mcr.microsoft.com/azureml/aifx/stable-ubuntu2004-cu115-py38-torch1110

USER root:root

RUN pip install pybind11

RUN pip install git+https://github.com/microsoft/DeepSpeed.git

# add a100-topo.xml
RUN mkdir -p /opt/microsoft/
RUN wget -O /opt/microsoft/a100-topo.xml https://hpcbenchmarks.blob.core.windows.net/bookcorpus/data/a100-topo.xml

# to use on A100, enable env var below in your job
ENV NCCL_TOPO_FILE="/opt/microsoft/a100-topo.xml"
