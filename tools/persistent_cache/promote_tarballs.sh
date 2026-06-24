#!/bin/bash
# tools/persistent_cache/promote_tarballs.sh — refresh CACHE_READ_DIR/<scope>.tar.zst from
# CACHE_WRITE_DIR/<scope>/. Called from sbatch wrapper as a separate
# `srun -p cpu`, never from a GPU job. Heavy tar work belongs on a CPU node.
#
# Args: scope names to promote. Defaults to all SCOPES.

set -euo pipefail
# shellcheck disable=SC1091
source "$(dirname "$0")/lib.sh"

mkdir -p "${CACHE_READ_DIR}"

# shellcheck disable=SC2206
todo=("${@:-${SCOPES[@]}}")
had_error=0

for scope in "${todo[@]}"; do
  tarball="$(scope_lustre_read_tar "$scope")"
  sources=()
  skipped_partial=0

  src="$(scope_lustre_write_dir "$scope")"
  if source_has_content "$src"; then
    if has_partial_marker "$src"; then
      echo "[CACHE PROMOTE] ${scope}: skip partial shared source ${src}" >&2
      skipped_partial=1
    else
      sources+=("$src")
    fi
  fi

  node_root="${CACHE_WRITE_DIR}/_nodes"
  if [[ -d "$node_root" ]]; then
    while IFS= read -r partial_marker; do
      echo "[CACHE PROMOTE] ${scope}: skip orphan partial node marker ${partial_marker}" >&2
      skipped_partial=1
    done < <(find "$node_root" -mindepth 2 -maxdepth 2 -type f -name "${scope}.partial" 2>/dev/null | sort)

    while IFS= read -r node_src; do
      if has_partial_marker "$node_src"; then
        echo "[CACHE PROMOTE] ${scope}: skip partial node source ${node_src}" >&2
        skipped_partial=1
        continue
      fi
      if source_has_content "$node_src"; then
        sources+=("$node_src")
      fi
    done < <(find "$node_root" -mindepth 2 -maxdepth 2 -type d -name "$scope" 2>/dev/null | sort)
  fi

  if (( ${#sources[@]} == 0 )); then
    if (( skipped_partial )); then
      echo "[CACHE PROMOTE] ${scope}: no complete source to promote (partial sources present)" >&2
      had_error=1
    fi
    continue
  fi

  needs=0
  if [[ ! -f "$tarball" ]]; then
    needs=1
  else
    for src in "${sources[@]}"; do
      if source_is_dirty_or_newer "$src" "$tarball"; then
        needs=1
        break
      fi
    done
  fi

  if (( needs )); then
    stage="${CACHE_WRITE_DIR}/.promote_${scope}.$$"
    rm -rf "$stage"
    mkdir -p "$stage"
    merge_failed=0
    for src in "${sources[@]}"; do
      if ! merge_cache_source "$src" "$stage" "$scope"; then
        merge_failed=1
      fi
    done
    if (( merge_failed )); then
      rm -rf "$stage"
      had_error=1
      continue
    fi
    if tar_safe "$stage" "$tarball" "$scope"; then
      for src in "${sources[@]}"; do
        rm -f "$src/.mcore_cache_dirty"
      done
      rm -rf "$stage"
    else
      rm -rf "$stage"
      had_error=1
    fi
  else
    echo "[CACHE PROMOTE] ${scope}: tarball up to date"
  fi
done

exit "$had_error"
