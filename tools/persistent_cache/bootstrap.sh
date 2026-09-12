#!/bin/bash
# tools/persistent_cache/bootstrap.sh — SOURCE me from rank_entrypoint.sh.
#
# Per node: seed NODE_CACHE_BASE/<scope>/ from CACHE_READ_DIR/<scope>.tar.zst.
# Per rank: export the env vars Megatron-LM's child processes will read.
#
# Sourcing (not executing) is required so the env-var exports stick in the
# caller's shell, where `python pretrain_*.py` will inherit them.

# shellcheck disable=SC1091
source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"

# Sweep stale exec-probe files from crashed prior jobs (>1 day old).
# Best-effort; failures are silent. Bounds disk leak from probe writes.
find /tmp /dev/shm -maxdepth 1 -name '.pc_probe.*' -mtime +1 -delete 2>/dev/null || true

# Probe candidate node-local dirs for write+exec. Some container runtimes mount
# /dev/shm with noexec, which breaks Triton/Inductor (they dlopen compiled .so).
# Probe order: /dev/shm (fastest) → /tmp → per-node lustre subdir (always works).
_pc_can_write_exec() {
  local dir="$1"
  mkdir -p "$dir" 2>/dev/null || return 1
  local probe="${dir}/.pc_probe.$$"
  printf '#!/bin/bash\nexit 0\n' > "$probe" 2>/dev/null || return 1
  if ! chmod +x "$probe" 2>/dev/null; then rm -f "$probe"; return 1; fi
  if ! "$probe" >/dev/null 2>&1;        then rm -f "$probe"; return 1; fi
  rm -f "$probe"
  return 0
}

_pc_select_base() {
  # NODE_CACHE_BASE_OVERRIDE wins if set (for testing).
  if [[ -n "${NODE_CACHE_BASE_OVERRIDE:-}" ]]; then
    if _pc_can_write_exec "${NODE_CACHE_BASE_OVERRIDE}"; then
      echo "${NODE_CACHE_BASE_OVERRIDE}" ; return 0
    fi
    echo "[CACHE BOOTSTRAP] override ${NODE_CACHE_BASE_OVERRIDE} is not write+exec; falling back" >&2
  fi
  local jid="${SLURM_JOB_ID:-manual}"
  local host
  host="$(hostname -s 2>/dev/null || hostname || echo unknown)"
  local cands=(
    "/dev/shm/mcore_cache_${jid}"
    "/tmp/mcore_cache_${jid}"
    "${PERSISTENT_CACHE:-/tmp}/_node/${host}_${jid}"
  )
  for c in "${cands[@]}"; do
    if _pc_can_write_exec "$c"; then
      echo "$c"; return 0
    fi
  done
  return 1
}

NODE_CACHE_BASE="$(_pc_select_base)"
export NODE_CACHE_BASE
echo "[CACHE BOOTSTRAP] node=$(hostname) base=${NODE_CACHE_BASE} (write+exec verified)"
if (( ! MCORE_CACHE_RSYNC_AVAILABLE )); then
  echo "[CACHE BOOTSTRAP] rsync not in PATH; sidecar/writeback will use per-node cp snapshot fallback"
fi

_pc_lock_dir="$(dirname "${NODE_CACHE_BASE}")"
mkdir -p "${_pc_lock_dir}"
_pc_lock_file="${NODE_CACHE_BASE}.bootstrap.lock"
_pc_sentinel="${NODE_CACHE_BASE}/.bootstrapped"

# Per-scope extract (run in parallel for speed).
_pc_extract_one() {
  local scope="$1"
  local local_dir tar_file t0 t1 dur
  local_dir="$(scope_local_dir "${scope}")"
  tar_file="$(scope_lustre_read_tar "${scope}")"
  mkdir -p "${local_dir}"
  if [[ -f "${tar_file}" ]]; then
    t0=$(date +%s.%N)
    # tar --zstd respects ZSTD_NBTHREADS=0 for multi-threaded decompress.
    if tar --zstd -xf "${tar_file}" -C "${local_dir}" 2>/dev/null; then
      t1=$(date +%s.%N)
      dur=$(awk -v a="${t0}" -v b="${t1}" 'BEGIN{printf "%.2f", b-a}')
      echo "[CACHE SEED] ${scope}: ok in ${dur}s ($(du -sh "${local_dir}" 2>/dev/null | cut -f1))"
    else
      echo "[CACHE SEED] ${scope}: extract failed (non-fatal, will compile cold)"
    fi
  else
    echo "[CACHE SEED] ${scope}: no tarball at ${tar_file}"
  fi
}

(
  # Bounded flock: if some other rank dies holding the lock, fall through after 600s
  # rather than block forever. Runs as cold seed (=== compile-from-scratch path).
  if ! flock -x -w 600 9; then
    echo "[CACHE BOOTSTRAP] flock timed out (600s); skipping seed (will compile cold)"
    exit 0
  fi
  if [[ ! -f "${_pc_sentinel}" ]]; then
    # Free-space pre-flight (informational only; probe already proved write).
    _free_gb=$(df -BG "${NODE_CACHE_BASE}" 2>/dev/null | awk 'NR==2{gsub(/G/,"",$4); print $4}' || true)
    echo "[CACHE BOOTSTRAP] ${NODE_CACHE_BASE}: ${_free_gb:-?}G free"

    # Ensure zstd is available.
    command -v zstd >/dev/null 2>&1 || {
      apt-get update -qq && apt-get install -y -qq zstd 2>/dev/null || true
    }

    # Wipe stale cache from a prior failed job on this node.
    rm -rf "${NODE_CACHE_BASE}"
    mkdir -p "${NODE_CACHE_BASE}"

    # Extract scopes in parallel. Each scope writes to its own dir, no conflicts.
    _pc_total_t0=$(date +%s.%N)
    for _scope in "${SCOPES[@]}"; do
      _pc_extract_one "${_scope}" &
    done
    wait
    _pc_total_t1=$(date +%s.%N)
    _pc_total_dur=$(awk -v a="${_pc_total_t0}" -v b="${_pc_total_t1}" 'BEGIN{printf "%.2f", b-a}')
    echo "[CACHE BOOTSTRAP] all scopes extracted in ${_pc_total_dur}s wall (parallel, ZSTD_NBTHREADS=${ZSTD_NBTHREADS})"

    touch "${_pc_sentinel}"
  fi
) 9>"${_pc_lock_file}"

# Every rank exports env vars now that the seed is done (or attempted).
# These are read by torch._inductor / triton / CUDA driver / TE at first import.
export TRITON_CACHE_DIR="$(scope_local_dir triton)"
export TORCHINDUCTOR_CACHE_DIR="$(scope_local_dir inductor)"
export CUDA_CACHE_PATH="$(scope_local_dir cuda_ptx)"
export HYBRID_EP_JIT_DIR="$(scope_local_dir hybrid_ep)"
export CUDNN_FRONTEND_CACHE_DIR="$(scope_local_dir cudnn_fe)"
# Megatron's "compiling and loading fused kernels" step uses
# torch.utils.cpp_extension.load, which checks TORCH_EXTENSIONS_DIR for an
# existing build, saving several seconds on warm restart.
export TORCH_EXTENSIONS_DIR="$(scope_local_dir megatron_fused_kernels)"
mkdir -p "$(scope_local_dir nccl_topo)"
# Dump-side: NCCL writes topology graph here at init. Useful for diagnostics.
export NCCL_TOPO_DUMP_FILE="$(scope_local_dir nccl_topo)/topo.xml"
export NCCL_GRAPH_DUMP_FILE="$(scope_local_dir nccl_topo)/graph.xml"
# NOTE: NCCL_TOPO_FILE read-side is intentionally NOT set. Tried it in v2; the
# dumped XML at 64-GPU scale exceeds NCCL 2.28's 256-node XML parser limit and
# causes init to fail. The dump is informational only for now.

echo "[CACHE BOOTSTRAP] rank=${SLURM_PROCID:-?} ready (NODE_CACHE_BASE=${NODE_CACHE_BASE})"
