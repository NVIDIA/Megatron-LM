#!/usr/bin/env bash
# Adapted from https://github.com/NVIDIA-NeMo/Megatron-Bridge/blob/08035723e7a86f94fc54a7f4473bd0519f2ac941/scripts/common/enroot_entrypoint.sh
# Source license: Apache-2.0.
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

# Embedded in the rank-local script by container_runtime.py, not run standalone.
# Never trace this script: inherited environment values may contain credentials.
set -euo pipefail
fail() { echo "Direct Enroot: $*" >&2; exit 1; }
command -v enroot >/dev/null || fail "enroot is not installed on this node"
[[ ${SLURM_JOB_ID:-} =~ ^[0-9]+$ && ${SLURM_PROCID:-} =~ ^[0-9]+$ ]] || fail "Slurm task identity is missing"
for path in "${host_paths[@]}"; do
    [[ -e "$path" ]] || fail "a required cluster path does not exist: $path"
    [[ $(stat -f -c %T -- "$path") == lustre ]] || fail "path is not on Lustre: $path"
done
[[ -d "$runtime_parent" && -O "$runtime_parent" && -w "$runtime_parent" ]] || fail "runtime root must be owned and writable"
[[ -f "$image_path" && -r "$image_path" ]] || fail "image must be an existing readable SquashFS file"
umask 077
runtime_dir=$(mktemp -d "$runtime_parent/job-${SLURM_JOB_ID}-rank-${SLURM_PROCID}-XXXXXXXX")
export ENROOT_DATA_PATH="$runtime_dir/data" ENROOT_RUNTIME_PATH="$runtime_dir/runtime"
export ENROOT_CACHE_PATH="$runtime_dir/cache" ENROOT_TEMP_PATH="$runtime_dir/temp"
export ENROOT_CONFIG_PATH="$runtime_dir/config" TMPDIR="$runtime_dir/tmp"
export ENROOT_MOUNT_HOME=n ENROOT_ROOTFS_WRITABLE=n ENROOT_LOGIN_SHELL=n
mkdir -p "$ENROOT_DATA_PATH" "$ENROOT_RUNTIME_PATH" "$ENROOT_CACHE_PATH" "$ENROOT_TEMP_PATH" \
    "$ENROOT_CONFIG_PATH" "$TMPDIR" "$runtime_dir/var-tmp" "$runtime_dir/cache/workload"
# Fail before touching the image if site configuration ignores any path override.
effective=$(enroot info) || fail "enroot info is required to validate runtime storage"
for name in ENROOT_DATA_PATH ENROOT_RUNTIME_PATH ENROOT_CACHE_PATH ENROOT_TEMP_PATH ENROOT_CONFIG_PATH; do
    actual=$(sed -n "s/^[[:space:]]*${name}=//p" <<< "$effective")
    [[ "$actual" == "${!name}" ]] || fail "site configuration overrides $name"
done
# Enroot config::fini unsets false values before `info`; an exported value
# (even "0") can enable these flags. Require their absence, not a spelling.
for name in ENROOT_MOUNT_HOME ENROOT_ROOTFS_WRITABLE ENROOT_LOGIN_SHELL; do
    if grep -q "^[[:space:]]*${name}=" <<< "$effective"; then
        fail "site configuration enables $name"
    fi
done
enroot_args=()
for mount in "${user_mounts[@]}"; do enroot_args+=(--mount "$mount"); done
for name in "${forward_names[@]}"; do
    [[ -v "$name" ]] || fail "requested environment variable is absent: $name"
    enroot_args+=(--env "$name")
done
# Slurm supplies rank/topology/GPU visibility at execution time, not submission.
for name in "${!SLURM_@}" RANK LOCAL_RANK WORLD_SIZE MASTER_ADDR MASTER_PORT CUDA_VISIBLE_DEVICES; do
    if [[ -v "$name" ]]; then enroot_args+=(--env "$name"); fi
done
# Respect explicitly forwarded cache roots only when their physical storage is
# Lustre. Otherwise use rank-local defaults on the mounted runtime directory.
cache_names=(XDG_CACHE_HOME XDG_RUNTIME_DIR UV_CACHE_DIR HF_HOME TORCH_HOME
    TORCH_EXTENSIONS_DIR TRITON_CACHE_DIR CUDA_CACHE_PATH NUMBA_CACHE_DIR
    HYBRID_EP_CACHE_DIR MEGATRON_CONFIG_LOCK_DIR HF_HUB_CACHE HUGGINGFACE_HUB_CACHE
    HF_DATASETS_CACHE HF_MODULES_CACHE TRANSFORMERS_CACHE XDG_DATA_HOME XDG_STATE_HOME
    XDG_CONFIG_HOME TORCHINDUCTOR_CACHE_DIR FLASHINFER_WORKSPACE_BASE MPLCONFIGDIR
    NEMO_HOME NEMO_CACHE_DIR NEMORUN_HOME)
for name in "${cache_names[@]}"; do
    if [[ " ${forward_names[*]} " == *" $name "* ]]; then
        path=${!name}
        [[ "$path" == /* && "$path" != *:* && "$path" != *,* && "$path" != *[[:space:]]* ]] || fail "invalid cache path for $name"
        [[ "/${path#/}/" != *"/../"* ]] || fail "cache path must not contain parent traversal: $name"
        [[ -d "$path" && $(stat -f -c %T -- "$path") == lustre ]] || fail "$name must name an existing Lustre directory"
        # Make the selected host cache available at the identical container path.
        enroot_args+=(--mount "$path:$path:none:bind,rw,x-create=dir")
    fi
done
# Resolve HF's dependent defaults only from explicitly forwarded settings, never
# from unrelated host environment values. Validate all explicit paths above
# before creating any dependent directory (the hub alias may appear later).
hf_home="$runtime_dir/cache/workload/HF_HOME"
if [[ " ${forward_names[*]} " == *" HF_HOME "* ]]; then hf_home="$HF_HOME"; fi
hf_hub="$hf_home/hub"
if [[ " ${forward_names[*]} " == *" HUGGINGFACE_HUB_CACHE "* ]]; then hf_hub="$HUGGINGFACE_HUB_CACHE"; fi
if [[ " ${forward_names[*]} " == *" HF_HUB_CACHE "* ]]; then hf_hub="$HF_HUB_CACHE"; fi
for name in "${cache_names[@]}"; do
    if [[ " ${forward_names[*]} " == *" $name "* ]]; then continue; fi
    case "$name" in
        HF_HOME) path="$hf_home" ;;
        HF_HUB_CACHE|HUGGINGFACE_HUB_CACHE|TRANSFORMERS_CACHE) path="$hf_hub" ;;
        HF_MODULES_CACHE) path="$hf_home/modules" ;;
        HF_DATASETS_CACHE) path="$hf_home/datasets" ;;
        *) path="$runtime_dir/cache/workload/$name" ;;
    esac
    if [[ "$path" == "$runtime_dir/"* ]]; then
        mkdir -p "$path"
    elif [[ -e "$path" || -L "$path" ]]; then
        [[ -d "$path" && $(stat -f -c %T -- "$path") == lustre ]] || fail "$name must resolve to Lustre storage"
    fi
    enroot_args+=(--env "$name=$path")
done
# Managed mounts must be LAST: a user mount of an ancestor (e.g. /var or the
# Lustre user root) must not hide the temporary/runtime storage underneath it.
enroot_args+=(--mount "$runtime_dir:$runtime_dir:none:bind,rw,x-create=dir"
    --mount "$runtime_dir/tmp:/tmp:none:bind,rw,x-create=dir"
    --mount "$runtime_dir/var-tmp:/var/tmp:none:bind,rw,x-create=dir")
# Short container-visible paths avoid Unix socket length limits, but are backed
# by the Lustre bind mounts above. Do not create bytecode beside readonly source.
enroot_args+=(--env TMPDIR=/tmp --env TMP=/tmp --env TEMP=/tmp --env PYTHONDONTWRITEBYTECODE=1)
# No recursive cleanup: retain this job's bounded runtime artifacts for diagnosis.
