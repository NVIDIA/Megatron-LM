FROM nvcr.io/nvidia/pytorch:23.04-py3

RUN pip install -U faiss-gpu

RUN apt update

RUN apt install -qy htop

RUN pip install -U transformers

RUN pip install --upgrade google-api-python-client

RUN pip install sentencepiece

RUN pip install h5py

RUN pip install nltk

RUN pip install einops
