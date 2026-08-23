# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

#!/bin/bash

set -ex

NCCL_VER="2.30.7-1+cuda13.3"

for i in "$@"; do
    case $i in
        --NCCL_VER=?*) NCCL_VER="${i#*=}";;
        *) ;;
    esac
    shift
done

ARCH=$(uname -m)
if [ "$ARCH" = "amd64" ];then ARCH="x86_64";fi
if [ "$ARCH" = "aarch64" ];then ARCH="sbsa";fi

CUDA_KEYRING_SHA256_X86_64="d2a6b11c096396d868758b86dab1823b25e14d70333f1dfa74da5ddaf6a06dba"
CUDA_KEYRING_SHA256_SBSA="6ea7d2737648936820e85677177957a0f6521b840d98eb0bbae0a4f003fa7249"
case "$ARCH" in
    x86_64) CUDA_KEYRING_SHA256="$CUDA_KEYRING_SHA256_X86_64";;
    sbsa) CUDA_KEYRING_SHA256="$CUDA_KEYRING_SHA256_SBSA";;
    *) echo "Unsupported architecture: $ARCH" >&2; exit 1;;
esac

curl -fsSLO https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/${ARCH}/cuda-keyring_1.1-1_all.deb
echo "${CUDA_KEYRING_SHA256}  cuda-keyring_1.1-1_all.deb" | sha256sum --check --strict
dpkg -i cuda-keyring_1.1-1_all.deb
rm cuda-keyring_1.1-1_all.deb

apt-get update

if [[ $(apt list --installed | grep libnccl) ]]; then
  apt-get remove --purge -y --allow-change-held-packages libnccl*
fi

apt-get install -y --no-install-recommends \
    libnccl2=${NCCL_VER} \
    libnccl-dev=${NCCL_VER} \

apt-get clean
rm -rf /var/lib/apt/lists/*
