FROM nvcr.io/nvidia/pytorch:23.10-py3

ENV DEBIAN_FRONTEND="noninteractive"
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
    zsh \
    curl \
    tmux \
    sudo \
    git-lfs \
    libaio-dev \
    && apt-get clean && \
    rm -rf /var/lib/apt/lists/*

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
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
