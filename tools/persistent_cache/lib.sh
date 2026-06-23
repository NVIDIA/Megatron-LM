#!/bin/bash
# tools/persistent_cache/lib.sh — shared functions and config. Source me; don't run me.
#
# Required env (caller must set):
#   PERSISTENT_CACHE  Lustre dir holding cache_read/ and cache_write/ subdirs.
#
# Derived (set by this lib if not already):
#   CACHE_READ_DIR, CACHE_WRITE_DIR, NODE_CACHE_BASE,
#   MCORE_CACHE_SYNC_DELAY_SECONDS, MCORE_CACHE_SYNC_FREQUENCY,
#   MCORE_CACHE_SYNC_JITTER_SECONDS, MCORE_CACHE_SYNC_EXIT_JITTER_SECONDS,
#   MCORE_CACHE_MAX_PERIODIC_WRITEBACKS, MCORE_JOB_START_EPOCH, SCOPES.

: "${PERSISTENT_CACHE:?PERSISTENT_CACHE is unset (set in launcher before sourcing)}"
: "${CACHE_READ_DIR:=${PERSISTENT_CACHE}/cache_read}"
: "${CACHE_WRITE_DIR:=${PERSISTENT_CACHE}/cache_write}"
: "${NODE_CACHE_BASE:=/dev/shm/mcore_cache_${SLURM_JOB_ID:-manual}}"

: "${MCORE_CACHE_SYNC_DELAY_SECONDS:=600}"
: "${MCORE_CACHE_SYNC_FREQUENCY:=300}"
: "${MCORE_CACHE_SYNC_JITTER_SECONDS:=120}"
: "${MCORE_CACHE_SYNC_EXIT_JITTER_SECONDS:=}"
# Default 0 = no in-job periodic writebacks (doc comment #1 / Russell). Only
# the staggered end-of-job consolidated writeback runs via sidecar EXIT trap.
# Set ≥1 only at small scale where a mid-job crash leaving the cache stale is
# more costly than the Lustre write storm. At ≥256 cards prefer the image-baked
# path with MCORE_CACHE_WRITEBACK_DISABLED=1.
: "${MCORE_CACHE_MAX_PERIODIC_WRITEBACKS:=0}"
: "${MCORE_JOB_START_EPOCH:=$(date +%s)}"
export MCORE_JOB_START_EPOCH

# shellcheck disable=SC2206
SCOPES=(${MCORE_CACHE_SCOPES:-triton inductor cuda_ptx hybrid_ep cudnn_fe nccl_topo dataset_idx megatron_fused_kernels})

# Multi-threaded zstd by default (decompress 4-8x faster on multi-core nodes).
# Can be overridden by user; tar --zstd reads ZSTD_NBTHREADS for both compress/decompress.
: "${ZSTD_NBTHREADS:=0}"   # 0 = use all available cores
export ZSTD_NBTHREADS

# Probe rsync availability once per process. If absent, writeback uses a
# per-node snapshot fallback instead of concurrent cp into the shared scope.
if command -v rsync >/dev/null 2>&1; then
  MCORE_CACHE_RSYNC_AVAILABLE=1
else
  MCORE_CACHE_RSYNC_AVAILABLE=0
fi
export MCORE_CACHE_RSYNC_AVAILABLE

scope_local_dir()        { echo "${NODE_CACHE_BASE}/$1"; }
scope_lustre_write_dir() { echo "${CACHE_WRITE_DIR}/$1"; }
scope_lustre_read_tar()  { echo "${CACHE_READ_DIR}/$1.tar.zst"; }
cache_node_id() {
  local host
  host="$(hostname -s 2>/dev/null || hostname 2>/dev/null || echo unknown)"
  echo "${SLURM_JOB_ID:-manual}_${host}"
}
scope_node_write_dir() { echo "${CACHE_WRITE_DIR}/_nodes/$(cache_node_id)/$1"; }

is_local_rank_zero() { [[ "${SLURM_LOCALID:-0}" == "0" ]]; }

# Copy src/ into the persistent writeback area.
#
# With rsync, write directly to cache_write/<scope>/ and mark the scope dirty
# only after a clean rc=0. Any nonzero rsync leaves .mcore_cache_partial, which
# promote_tarballs.sh refuses to package.
#
# Without rsync, never cp concurrently into the shared scope. Copy into a
# job+node-private staging dir, then atomically publish the complete snapshot
# under cache_write/_nodes/<job>_<host>/<scope>. Promotion later merges completed
# node snapshots on a CPU node.
rsync_safe() {
  local src="$1" dst="$2" name="$3"
  [[ -d "$src" ]] || return 1
  [[ -n "$(ls -A "$src" 2>/dev/null)" ]] || return 1
  local _err _rc=0 partial
  if command -v rsync >/dev/null 2>&1; then
    mkdir -p "$dst"
    partial="${dst}/.mcore_cache_partial"
    touch "$partial" 2>/dev/null || true
    _err=$(rsync -a --exclude='tmp*' --exclude='.tmp_*' --exclude='.*' \
      "$src/" "$dst/" 2>&1) || _rc=$?
    if (( _rc == 0 )); then
      rm -f "$partial"
      touch "$dst/.mcore_cache_dirty" 2>/dev/null || true
      touch "$dst/.mcore_cache_writeback_stamp" 2>/dev/null || true
      echo "[CACHE] ${name}: rsynced ($(du -sh "$dst" 2>/dev/null | cut -f1))"
      return 0
    fi
    echo "[CACHE] ${name}: rsync failed, left partial marker at ${partial} (rc=${_rc}: ${_err})" >&2
    return 1
  fi

  # Fallback: private per-node snapshot. The shared cache_write/<scope>/ is not
  # touched, so a failed cp cannot leave truncated files in the promotable tree.
  local node_dst tmp parent sibling_partial
  node_dst="$(scope_node_write_dir "$name")"
  parent="$(dirname "$node_dst")"
  tmp="${node_dst}.tmp.$$"
  sibling_partial="${node_dst}.partial"
  mkdir -p "$parent"
  rm -rf "$tmp"
  touch "$sibling_partial" 2>/dev/null || true
  mkdir -p "$tmp"
  shopt -s nullglob dotglob 2>/dev/null
  local entry _bn
  for entry in "$src"/*; do
    _bn="$(basename "$entry")"
    case "$_bn" in
      tmp*|.tmp_*|.*) continue ;;
    esac
    cp -a "$entry" "$tmp/" 2>/dev/null || _rc=$?
  done
  if (( _rc == 0 )); then
    touch "$tmp/.mcore_cache_dirty" 2>/dev/null || true
    touch "$tmp/.mcore_cache_writeback_stamp" 2>/dev/null || true
    if rm -rf "$node_dst" && mv "$tmp" "$node_dst"; then
      rm -f "$sibling_partial"
      echo "[CACHE] ${name}: cp-snapshotted $(cache_node_id) ($(du -sh "$node_dst" 2>/dev/null | cut -f1))"
      return 0
    fi
    _rc=$?
  fi
  rm -rf "$tmp"
  echo "[CACHE] ${name}: cp snapshot failed, left partial marker at ${sibling_partial} (rc=${_rc})" >&2
  return 1
}

has_partial_marker() {
  local src="$1"
  [[ -f "${src}.partial" || -f "${src}/.mcore_cache_partial" ]]
}

source_has_content() {
  local src="$1"
  [[ -d "$src" ]] && [[ -n "$(find "$src" -mindepth 1 \
    ! -name '.mcore_cache_dirty' \
    ! -name '.mcore_cache_writeback_stamp' \
    ! -name '.mcore_cache_partial' \
    ! -name 'tmp*' \
    ! -name '.tmp_*' \
    -print -quit 2>/dev/null)" ]]
}

source_is_dirty_or_newer() {
  local src="$1" tarball="$2"
  [[ -f "${src}/.mcore_cache_dirty" ]] && return 0
  [[ ! -f "$tarball" ]] && return 0
  find "$src" -type f \
    ! -name '.mcore_cache_dirty' \
    ! -name '.mcore_cache_writeback_stamp' \
    ! -name '.mcore_cache_partial' \
    -newer "$tarball" -print -quit 2>/dev/null | grep -q .
}

merge_cache_source() {
  local src="$1" dst="$2" name="$3"
  [[ -d "$src" ]] || return 0
  mkdir -p "$dst"
  local _err _rc=0
  if command -v rsync >/dev/null 2>&1; then
    _err=$(rsync -a --ignore-existing --exclude='tmp*' --exclude='.tmp_*' \
      --exclude='.mcore_cache_*' "$src/" "$dst/" 2>&1) || _rc=$?
    if (( _rc == 0 || _rc == 23 || _rc == 24 )); then
      return 0
    fi
    echo "[CACHE PROMOTE] ${name}: merge rsync failed from ${src} (rc=${_rc}: ${_err})" >&2
    return 1
  fi

  shopt -s nullglob dotglob 2>/dev/null
  local entry _bn
  for entry in "$src"/*; do
    _bn="$(basename "$entry")"
    case "$_bn" in
      tmp*|.tmp_*|.mcore_cache_*) continue ;;
    esac
    cp -an "$entry" "$dst/" 2>/dev/null || _rc=$?
  done
  if (( _rc == 0 )); then
    return 0
  fi
  echo "[CACHE PROMOTE] ${name}: merge cp failed from ${src} (rc=${_rc})" >&2
  return 1
}

# Atomic tar+zstd. tmp+rename so readers never see a partial tarball.
# NOTE: don't use --exclude='.*' — that matches the '.' source arg and emits an empty tar.
tar_safe() {
  local src="$1" tarball="$2" name="$3"
  [[ -d "$src" ]] || return 1
  [[ -n "$(ls -A "$src" 2>/dev/null)" ]] || return 1
  local tmp="${tarball}.tmp.$$"
  if tar --zstd -cf "$tmp" --blocking-factor=8192 \
       -C "$src" --exclude='tmp*' --exclude='.tmp_*' \
       --exclude='.mcore_cache_dirty' --exclude='.mcore_cache_writeback_stamp' . ; then
    mv "$tmp" "$tarball"
    echo "[CACHE] ${name}: tarball ($(du -sh "$tarball" 2>/dev/null | cut -f1))"
    return 0
  fi
  rm -f "$tmp"
  echo "[CACHE] ${name}: tar failed" >&2
  return 1
}
