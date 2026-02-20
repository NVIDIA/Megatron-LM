#! /bin/bash
# Copyright (c) 2018-2022, NVIDIA CORPORATION.  All rights reserved.
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

set -euo pipefail

print_usage() {
    cat << EOF
${0} [options] [--] COMMAND [ARG...]

Control binding policy for each task. Assumes one rank will be launched for each GPU.

Options:
    --cpu=MODE
        * exclusive -- bind each rank to an exclusive set of cores near its GPU
        * exclusive,nosmt -- bind each rank to an exclusive set of cores near its GPU, without hyperthreading
        * node -- bind each rank to all cores in the NUMA node nearest its GPU [default]
	* *.sh -- bind each rank using the bash associative array bind_cpu_cores or bind_cpu_nodes from a file
        * off -- don't bind
    --mem=MODE
        * node -- bind each rank to the nearest NUMA node [default]
	* *.sh -- bind each rank using the bash associative array bind_mem from a file
        * off -- don't bind
    --ib=MODE
        * single -- bind each rank to a single IB device near its GPU
        * off -- don't bind [default]
EOF
}

################################################################################
# Argument parsing
################################################################################

cpu_mode='node'
mem_mode='node'
ib_mode='off'
while [[ "$#" -gt "0" ]]; do
    case "$1" in
        -h|--help) print_usage ; exit 0 ;;
        --cpu=*) cpu_mode="${1/*=/}"; shift ;;
        --cpu)   cpu_mode="$2"; shift 2 ;;
        --mem=*) mem_mode="${1/*=/}"; shift ;;
        --mem)   mem_mode="$2"; shift 2 ;;
        --ib=*) ib_mode="${1/*=/}"; shift ;;
        --ib)   ib_mode="$2"; shift 2 ;;
        --) shift; break ;;
        *) break ;;
    esac
done
if [[ $# -lt 1 ]]; then
    echo 'ERROR: no command given' 2>&1
    print_usage
    exit 1
fi

################################################################################
# Get system params
################################################################################

# LOCAL_RANK is set with an enroot hook for Pytorch containers
# SLURM_LOCALID is set by Slurm
# OMPI_COMM_WORLD_LOCAL_RANK is set by mpirun
readonly local_rank="${LOCAL_RANK:=${SLURM_LOCALID:=${OMPI_COMM_WORLD_LOCAL_RANK:-}}}"
if [[ ! "${local_rank}" ]]; then
    echo 'ERROR: cannot read LOCAL_RANK from env' >&2
    exit 1
fi

num_gpus=$(nvidia-smi -i 0 --query-gpu=count --format=csv,noheader,nounits)
if [[ "${local_rank}" -ge "${num_gpus}" ]]; then
    echo "ERROR: local rank is ${local_rank}, but there are only ${num_gpus} gpus available" >&2
    exit 1
fi

get_lscpu_value() {
    awk -F: "(\$1 == \"${1}\"){gsub(/ /, \"\", \$2); print \$2; found=1} END{exit found!=1}"
}
lscpu_out=$(lscpu)
num_sockets=$(get_lscpu_value 'Socket(s)' <<< "${lscpu_out}")
num_nodes=$(lscpu --parse | grep -v '^#' | cut -f4 -d, | sort --unique | wc --lines)
cores_per_socket=$(get_lscpu_value 'Core(s) per socket' <<< "${lscpu_out}")

echo "num_gpus=${num_gpus} num_sockets = ${num_sockets} num_nodes=${num_nodes} cores_per_socket=${cores_per_socket}"

readonly cores_per_node=$(( (num_sockets * cores_per_socket) / num_nodes ))
if [[ "${num_gpus}" -gt "1" ]] && [[ "${num_gpus}" -ge "${num_nodes}" ]]; then
    readonly gpus_per_node=$(( num_gpus / num_nodes ))
else
    readonly gpus_per_node=1
    num_nodes="${num_gpus}"
fi
readonly cores_per_gpu=$(( cores_per_node / gpus_per_node ))
readonly local_node=$(( local_rank / gpus_per_node ))


declare -a ibdevs=()
if ibstat_out="$(ibv_devinfo --list | tail -n+2 | cut -f2 | grep -v '^$')"; then
    mapfile -t ibdevs <<< "${ibstat_out}"
fi
readonly num_ibdevs="${#ibdevs[@]}"

################################################################################
# Setup for exec
################################################################################

declare -a numactl_args=()

case "${cpu_mode}" in
    exclusive)
        numactl_args+=( "$(printf -- "--physcpubind=%u-%u,%u-%u" \
            $(( local_rank * cores_per_gpu )) \
            $(( (local_rank + 1) * cores_per_gpu - 1 )) \
            $(( local_rank * cores_per_gpu + (cores_per_gpu * gpus_per_node * num_nodes) )) \
            $(( (local_rank + 1) * cores_per_gpu + (cores_per_gpu * gpus_per_node * num_nodes) - 1 )) \
        )" )
        ;;
    exclusive,nosmt)
        numactl_args+=( "$(printf -- "--physcpubind=%u-%u" \
            $(( local_rank * cores_per_gpu )) \
            $(( (local_rank + 1) * cores_per_gpu - 1 )) \
        )" )
        ;;
    node)
        numactl_args+=( "--cpunodebind=${local_node}" )
        ;;
    *.sh)
	# shellcheck source=/dev/null
	source "${cpu_mode}"
	if [[ "${bind_cpu_cores:-}" ]]; then
	    numactl_args+=( "--physcpubind=${bind_cpu_cores[${local_rank}]}" )
	elif [[ "${bind_cpu_nodes:-}" ]]; then
	    numactl_args+=( "--cpunodebind=${bind_cpu_nodes[${local_rank}]}" )
	else
	    echo "ERROR: invalid CPU affinity file ${cpu_mode}." >&2
	    exit 1
	fi
	;;
    off|'')
        ;;
    *)
        echo "ERROR: invalid cpu mode '${cpu_mode}'" 2>&1
        print_usage
        exit 1
        ;;
esac

case "${mem_mode}" in
    node)
        numactl_args+=( "--membind=${local_node}" )
        ;;
    *.sh)
	# shellcheck source=/dev/null
	source "${mem_mode}"
	if [[ ! "${bind_mem:-}" ]]; then
	    echo "ERROR: invalid memory affinity file ${mem_mode}." >&2
	    exit 1
	fi
	numactl_args+=( "--membind=${bind_mem[${local_rank}]}" )
	;;
    off|'')
        ;;
    *)
        echo "ERROR: invalid mem mode '${mem_mode}'" 2>&1
        print_usage
        exit 1
        ;;
esac

case "${ib_mode}" in
    single)
        if [[ "${num_ibdevs}" -eq 0 ]]; then
            echo "WARNING: used '$0 --ib=single', but there are 0 IB devices available; skipping IB binding." 2>&1
	elif (( num_ibdevs > num_gpus)) || (( num_gpus % num_ibdevs != 0 )) ; then
	    echo "ERROR: can't evenly map ${num_gpus} gpus to ${num_ibdevs} ibdevs" 2>&1
	    echo "set MELLANOX_VISIBLE_DEVICES correctly or use --ib=off" 2>&1
	    exit 1
        else
            readonly ibdev="${ibdevs[$(( local_rank * num_ibdevs / num_gpus ))]}"
            export OMPI_MCA_btl_openib_if_include="${OMPI_MCA_btl_openib_if_include-$ibdev}"
            export UCX_NET_DEVICES="${UCX_NET_DEVICES-$ibdev:1}"
        fi
        ;;
    off|'')
        ;;
    *)
        echo "ERROR: invalid ib mode '${ib_mode}'" 2>&1
        print_usage
        exit 1
        ;;
esac

################################################################################
# Exec
################################################################################

if [[ "${#numactl_args[@]}" -gt 0 ]] ; then
    [[ "${DEBUG:-0}" = "1" ]] && set -x
    exec numactl "${numactl_args[@]}" -- "${@}"
else
    exec "${@}"
fi
