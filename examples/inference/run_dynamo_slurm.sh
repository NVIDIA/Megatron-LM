#!/bin/bash
# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

usage() {
    cat <<'EOF'
Launch one externally managed Megatron Dynamo engine in an existing SLURM allocation.

Required environment:
  NPROC_PER_NODE       Megatron processes to launch on each node
  PARENT_EVENT_HOST    Routable hostname or address of this Dynamo parent
  SLURM_NNODES         Number of nodes in the current allocation
  SLURM_JOB_NODELIST   Nodes in the current allocation

Optional environment:
  PYTHON_EXECUTABLE    Python executable visible on every node (default: python)
  MASTER_ADDR          torchrun rendezvous host (default: first allocated node)
  MASTER_PORT          torchrun rendezvous port (default: 29500)
  PARENT_EVENT_PORT    Engine-event port on the Dynamo parent (default: 5557)

Usage:
  run_dynamo_slurm.sh <Dynamo backend args> -- <Megatron args>
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    usage
    exit 0
fi

: "${NPROC_PER_NODE:?NPROC_PER_NODE must be set}"
: "${PARENT_EVENT_HOST:?PARENT_EVENT_HOST must be set to a routable parent address}"
: "${SLURM_NNODES:?Run this script inside a SLURM allocation}"
: "${SLURM_JOB_NODELIST:?Run this script inside a SLURM allocation}"

backend_args=()
while [[ $# -gt 0 && "$1" != "--" ]]; do
    case "$1" in
        --engine-launch-mode|--engine-launch-mode=*|\
        --nproc-per-node|--nproc-per-node=*|\
        --parent-event-host|--parent-event-host=*|\
        --handoff-owner|--handoff-owner=*|\
        --parent-event-port|--parent-event-port=*)
            echo "$(basename "$0") owns $1; configure it through the documented environment" >&2
            exit 2
            ;;
    esac
    backend_args+=("$1")
    shift
done
if [[ $# -eq 0 ]]; then
    echo "Missing '--' before the Megatron arguments" >&2
    exit 2
fi
shift
if [[ $# -eq 0 ]]; then
    echo "Megatron arguments are required after '--'" >&2
    exit 2
fi
megatron_args=("$@")

role=aggregated
coordinator_host=
coordinator_port=
for ((index = 0; index < ${#backend_args[@]}; index++)); do
    argument=${backend_args[$index]}
    case "$argument" in
        --role|--coordinator-host|--coordinator-port)
            if ((index + 1 >= ${#backend_args[@]})); then
                echo "Missing value for $argument" >&2
                exit 2
            fi
            value=${backend_args[$((index + 1))]}
            ((index += 1))
            case "$argument" in
                --role) role=$value ;;
                --coordinator-host) coordinator_host=$value ;;
                --coordinator-port) coordinator_port=$value ;;
            esac
            ;;
        --role=*) role=${argument#*=} ;;
        --coordinator-host=*) coordinator_host=${argument#*=} ;;
        --coordinator-port=*) coordinator_port=${argument#*=} ;;
    esac
done

python_executable=${PYTHON_EXECUTABLE:-python}
handoff_owner=
if [[ "$role" == "decode" ]]; then
    handoff_owner=$("$python_executable" -c 'import uuid; print(uuid.uuid4().hex)')
    backend_args+=(--handoff-owner "$handoff_owner")
    echo "Decode handoff owner: $handoff_owner" >&2
fi
master_port=${MASTER_PORT:-29500}
parent_event_port=${PARENT_EVENT_PORT:-5557}
parent_event_host=$PARENT_EVENT_HOST
readarray -t allocated_hosts < <(scontrol show hostnames "$SLURM_JOB_NODELIST")
master_addr=${MASTER_ADDR:-${allocated_hosts[0]}}

parent_event_address="tcp://${parent_event_host}:${parent_event_port}"
engine_args=(
    --dynamo-parent-event-address "$parent_event_address"
    --role "$role"
)
if [[ -n "$coordinator_host" ]]; then
    engine_args+=(--coordinator-host "$coordinator_host")
fi
if [[ -n "$coordinator_port" ]]; then
    engine_args+=(--coordinator-port "$coordinator_port")
fi

"$python_executable" -m megatron.inference.integrations.dynamo \
    "${backend_args[@]}" \
    --engine-launch-mode external \
    --parent-event-host "$parent_event_host" \
    --parent-event-port "$parent_event_port" \
    -- "${megatron_args[@]}" &
parent_pid=$!

srun \
    --nodes="$SLURM_NNODES" \
    --ntasks="$SLURM_NNODES" \
    --ntasks-per-node=1 \
    --gpus-per-task="$NPROC_PER_NODE" \
    --kill-on-bad-exit=1 \
    bash -c '
        set -euo pipefail
        python_executable=$1
        nnodes=$2
        nproc_per_node=$3
        master_addr=$4
        master_port=$5
        shift 5
        exec "$python_executable" -m torch.distributed.run \
            --nnodes="$nnodes" \
            --nproc-per-node="$nproc_per_node" \
            --node-rank="${SLURM_NODEID:?SLURM_NODEID is not set}" \
            --master-addr="$master_addr" \
            --master-port="$master_port" \
            --module megatron.inference.integrations.dynamo.engine_service \
            "$@"
    ' bash \
    "$python_executable" \
    "$SLURM_NNODES" \
    "$NPROC_PER_NODE" \
    "$master_addr" \
    "$master_port" \
    "${engine_args[@]}" \
    "${megatron_args[@]}" &
engine_step_pid=$!

terminate_children() {
    kill "$parent_pid" "$engine_step_pid" 2>/dev/null || true
}
trap terminate_children INT TERM

while kill -0 "$parent_pid" 2>/dev/null && kill -0 "$engine_step_pid" 2>/dev/null; do
    sleep 1
done

status=0
if ! kill -0 "$parent_pid" 2>/dev/null; then
    wait "$parent_pid" || status=$?
else
    wait "$engine_step_pid" || status=$?
    if [[ $status -eq 0 ]]; then
        echo "Megatron engine step exited before the Dynamo parent" >&2
        status=1
    fi
fi
terminate_children
wait "$parent_pid" 2>/dev/null || true
wait "$engine_step_pid" 2>/dev/null || true
if [[ -n "$handoff_owner" ]]; then
    # Reaping the local srun client alone is not proof that remote GPU ranks
    # have exited. The allocation controller must attest to that separately.
    echo "After confirming this attempt's parent and ALL decode ranks have exited:" >&2
    printf '%q ' "$python_executable" -m megatron.inference.integrations.dynamo.handoff_recovery \
        --coordinator-address '<each-prefill-coordinator>' \
        --confirm-terminated-owner "$handoff_owner" >&2
    echo >&2
fi
exit "$status"
