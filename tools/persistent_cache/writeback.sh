#!/bin/bash
# tools/persistent_cache/writeback.sh — one-shot writeback from NODE_CACHE_BASE/
# to CACHE_WRITE_DIR/. Uses rsync when available; otherwise writes per-node
# snapshots that promotion merges later.
# Args: --final  bypass delay/recency guards (used by EXIT trap and atexit).

set -euo pipefail
# shellcheck disable=SC1091
source "$(dirname "$0")/lib.sh"

is_local_rank_zero || exit 0

force=0
[[ "${1:-}" == "--final" ]] && force=1

now=$(date +%s)
elapsed=$(( now - MCORE_JOB_START_EPOCH ))
had_error=0

for scope in "${SCOPES[@]}"; do
  src="$(scope_local_dir "$scope")"
  dst="$(scope_lustre_write_dir "$scope")"
  [[ -d "$src" ]] || continue
  [[ -n "$(ls -A "$src" 2>/dev/null)" ]] || continue

  if (( ! force )); then
    # Guard 1: delay since job start. Compile is in flight before this.
    if (( elapsed < MCORE_CACHE_SYNC_DELAY_SECONDS )); then
      echo "[CACHE WB] ${scope}: skip (${elapsed}s < ${MCORE_CACHE_SYNC_DELAY_SECONDS}s)"
      continue
    fi
    # Guard 2: file recency — anything modified in last 5m means compile in progress.
    if find "$src" -type f -mmin -5 -print -quit 2>/dev/null | grep -q .; then
      echo "[CACHE WB] ${scope}: skip (compile in flight)"
      continue
    fi
  fi

  if ! rsync_safe "$src" "$dst" "$scope"; then
    had_error=1
  fi
done

exit "$had_error"
