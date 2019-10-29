# ===========
# base images
# ===========
FROM nvcr.io/nvidia/pytorch:19.09-py3


# ===============
# system packages
# ===============
RUN apt-get update && apt-get install -y \
    bash-completion \
    emacs \
    git \
    graphviz \
    htop \
    libopenexr-dev \
    rsync \
    wget \
&& rm -rf /var/lib/apt/lists/*


# ============
# pip packages
# ============
RUN pip install --upgrade pip && \
    pip install --upgrade setuptools
COPY requirements.txt /tmp/
RUN pip install --upgrade --ignore-installed -r /tmp/requirements.txt

