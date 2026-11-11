#!/bin/bash
# Submit several run.sh configs back to back: each run's server starts after the previous run's
# client ends (any outcome), so a per-user running-job cap (short QoS: 2) never strands a server
# without its client. KEY=VALUE arguments after the configs apply to every run.
#
#   inference-bench/run_chain.sh inference-bench/configs/pd_interference/superv3_{narrow,wide}_{agg,pd}.env
set -euo pipefail
CONFIGS=(); OVERRIDES=()
for arg in "$@"; do
    if [[ "${arg}" == *=* ]]; then OVERRIDES+=("${arg}"); else CONFIGS+=("${arg}"); fi
done
AFTER=${AFTER:-}
for config in "${CONFIGS[@]}"; do
    out=$(AFTER="${AFTER}" inference-bench/run.sh "${config}" "${OVERRIDES[@]}")
    echo "${out}"
    AFTER=$(echo "${out}" | sed -n 's/.*client \([0-9]*\).*/\1/p' | tail -1)
    [ -n "${AFTER}" ] || { echo "could not read the client job id of ${config}"; exit 1; }
done
