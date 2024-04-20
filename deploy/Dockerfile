# 分散学習用コンテナのための Dockerfile
# Usage:
#   docker build -t megatron-lm:latest -f deploy/Dockerfile .
#   docker tag megatron-lm:latest asia-southeast1-docker.pkg.dev/abeja-geniac/megatron-lm/megatron-lm:latest
#   docker push asia-southeast1-docker.pkg.dev/abeja-geniac/megatron-lm/megatron-lm:latest

FROM nvcr.io/nvidia/pytorch:23.10-py3

ENV DEBIAN_FRONTEND="noninteractive"
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
    sudo \
    git-lfs \
    libaio-dev \
    && apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Add user
ARG USERNAME=ubuntu
ARG USER_UID=1000
ARG USER_GID=1000
RUN groupadd --gid ${USER_GID} ${USERNAME} \
    && useradd --uid ${USER_UID} --gid ${USER_GID} -m ${USERNAME}
# add user in sudoers
RUN echo "${USERNAME} ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers
RUN visudo -c
USER ${USERNAME}

ENV PATH=/home/${USERNAME}/.local/bin:${PATH}

WORKDIR /app
COPY --chown=${USERNAME}:${USERNAME} . .
RUN pip install --no-cache-dir -r requirements.txt
