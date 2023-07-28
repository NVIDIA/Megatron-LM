FROM mcr.microsoft.com/azureml/curated/acpt-pytorch-1.11-cuda11.3:12
USER root:root

RUN pip install pybind11
RUN pip install regex