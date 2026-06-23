#!/bin/bash
# writeback_tarball.sh — simplified single-writer tarball writeback for PRETRAINING.
#
# Pretraining is single-role: the Triton/Inductor compile cache is ~identical
# across ranks (high overlap), so we don't need a per-role cache_write directory
# stage, rsync, or union accumulation.
#
# ONE designated writer (global rank 0) tars its node-local cache per scope and
# writes ONE <scope>.tar.zst straight to cache_read/ (atomic tmp+rename via
# tar_safe). Result: a handful of large sequential writes by a single rank ->
# NO per-node dirs, NO millions of small-file creates, NO Lustre MDT storm.
# The read path (bootstrap.sh) already extracts cache_read/*.tar.zst into /tmp.
#
# Compression is large (inductor-dominated), so the single write per scope is
# cheap and storm-free.
#
# Kill switch: MCORE_CACHE_WRITEBACK_DISABLED=1 (use image-baked at >=256 cards).
# Override the writer rank with MCORE_CACHE_WRITER_GLOBAL_RANK (default 0).

set -euo pipefail
# shellcheck disable=SC1091
source "$(dirname "$0")/lib.sh"

[[ "${MCORE_CACHE_WRITEBACK_DISABLED:-0}" == "1" ]] && { echo "[CACHE WB] disabled (MCORE_CACHE_WRITEBACK_DISABLED=1)"; exit 0; }

# Exactly one writer for the whole job — the union is ~one node's cache anyway.
writer_rank="${MCORE_CACHE_WRITER_GLOBAL_RANK:-0}"
if [[ "${SLURM_PROCID:-0}" != "${writer_rank}" ]]; then
  exit 0
fi

# Optional stagger so that, if this is ever run by >1 writer, they don't collide.
if [[ -n "${MCORE_CACHE_SYNC_EXIT_JITTER_SECONDS:-}" ]]; then
  sleep "$(( RANDOM % (MCORE_CACHE_SYNC_EXIT_JITTER_SECONDS + 1) ))" || true
fi

mkdir -p "${CACHE_READ_DIR}"
had_error=0
for scope in "${SCOPES[@]}"; do
  src="$(scope_local_dir "$scope")"
  [[ -d "$src" ]] || continue
  [[ -n "$(ls -A "$src" 2>/dev/null)" ]] || continue
  # tar_safe: atomic tar --zstd to <tmp> then rename, so readers never see a partial tarball.
  if ! tar_safe "$src" "$(scope_lustre_read_tar "$scope")" "$scope"; then
    had_error=1
  fi
done

if (( had_error )); then
  echo "[CACHE WB] one or more scopes failed to tar (rank ${writer_rank})" >&2
fi
exit "$had_error"
