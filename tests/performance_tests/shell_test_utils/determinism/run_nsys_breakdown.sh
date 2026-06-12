#!/bin/bash
# Wrap a Python entry point under nsys twice (det + nondet) and print the
# per-NVTX-range diff. Entry point reads DETERMINISM_PERF_MODE and must call
# cudaProfilerStart/Stop (Megatron's --profile flag handles this).
# Usage: bash run_nsys_breakdown.sh OUTDIR -- CMD...
set -euo pipefail
OUT_ARG="${1:?usage: $0 OUTDIR -- CMD...}"; shift
[ "${1:-}" = "--" ] || { echo "expected --"; exit 64; }; shift
# mkdir before realpath: realpath fails on missing path under ``set -e``.
mkdir -p "$OUT_ARG"
OUT=$(realpath "$OUT_ARG")

for MODE in det nondet; do
  DETERMINISM_PERF_MODE=$MODE \
    nsys profile -t cuda,nvtx -f true \
      --capture-range=cudaProfilerApi --capture-range-end=stop-shutdown \
      -o "$OUT/nsys-$MODE" "$@"
  nsys stats --force-export=true --report nvtx_sum --format csv "$OUT/nsys-$MODE.nsys-rep" > "$OUT/nsys-$MODE.csv"
done

# LOG_DIR (if set by caller) enables the step-time regression check.
python "$(dirname "$0")/print_nsys_leaderboard.py" "$OUT" ${LOG_DIR:+"$LOG_DIR"}
