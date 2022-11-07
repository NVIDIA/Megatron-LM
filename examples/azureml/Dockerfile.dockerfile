FROM mcr.microsoft.com/azureml/curated/acpt-pytorch-1.11-py38-cuda11.5-gpu
USER root:root

RUN pip install pybind11
RUN pip install regex