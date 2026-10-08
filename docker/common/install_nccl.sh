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

NCCL_VER="2.30.4-1+cuda13.2"

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

curl -fsSLO https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/${ARCH}/cuda-keyring_1.1-1_all.deb
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
