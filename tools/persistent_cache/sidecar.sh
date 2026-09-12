#!/bin/bash
# tools/persistent_cache/sidecar.sh — long-running per-node writeback loop.
# Spawn from rank_entrypoint with `&`. One per node (gated by SLURM_LOCALID==0).

set -euo pipefail
# shellcheck disable=SC1091
source "$(dirname "$0")/lib.sh"

is_local_rank_zero || exit 0
# Default OFF (doc comment #1): periodic in-job writebacks add Lustre fan-in
# without measurable benefit beyond the staggered end-of-job consolidation.
# Launchers needing in-job snapshotting must opt in with MCORE_CACHE_SIDECAR_ENABLED=1.
[[ "${MCORE_CACHE_SIDECAR_ENABLED:-0}" == "1" ]] || exit 0

WB="$(dirname "$0")/writeback.sh"

cache_sleep() {
  local seconds="${1:-0}"
  [[ "$seconds" =~ ^[0-9]+$ ]] || seconds=0
  if command -v python >/dev/null 2>&1; then
    python - "$seconds" <<'PY'
import sys
import time

time.sleep(int(sys.argv[1]))
PY
    return
  fi
  if command -v python3 >/dev/null 2>&1; then
    python3 - "$seconds" <<'PY'
import sys
import time

time.sleep(int(sys.argv[1]))
PY
    return
  fi
  sleep "$seconds"
}

_final() {
  # Per-node exit jitter scaled by cluster size. Avoids N-node thundering herd
  # at job teardown.
  local nnodes=${SLURM_JOB_NUM_NODES:-1}
  local window=${MCORE_CACHE_SYNC_EXIT_JITTER_SECONDS:-}
  if [[ -z "$window" ]]; then
    window=$(( nnodes / 20 ))
    (( window < 15 )) && window=15
    (( window > 45 )) && window=45
  fi
  cache_sleep "$(( RANDOM % (window > 0 ? window : 1) ))"
  bash "$WB" --final || true
}
trap '_final' EXIT

# Initial per-node jitter so 64 nodes don't first-rsync at the same wall clock.
cache_sleep "$(( RANDOM % MCORE_CACHE_SYNC_JITTER_SECONDS ))"

_periodic_count=0
while cache_sleep "$MCORE_CACHE_SYNC_FREQUENCY"; do
  if (( MCORE_CACHE_MAX_PERIODIC_WRITEBACKS >= 0 && _periodic_count >= MCORE_CACHE_MAX_PERIODIC_WRITEBACKS )); then
    continue
  fi
  bash "$WB" || true
  _periodic_count=$(( _periodic_count + 1 ))
done
