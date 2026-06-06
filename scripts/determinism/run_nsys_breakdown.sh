#!/bin/bash
# Run any Python entry point twice and capture per-NVTX-range CUDA time
# under nsys, then print a side-by-side comparison. The two passes are
# distinguished only by the ``DETERMINISM_PERF_MODE=det|nondet`` env var
# we export per iteration — YOUR entry point is responsible for branching
# on that env var (e.g. conditionally setting --deterministic-mode). If
# the entry point ignores the env var, both passes execute identically
# and the leaderboard delta is just measurement noise.
#
# The entry point MUST call cudaProfilerStart/Stop around the iters to
# measure (Megatron's --profile flag does this).
#
# Usage:
#   bash run_nsys_breakdown.sh OUTDIR -- <command-that-reads-DETERMINISM_PERF_MODE>
#
# Example (uses bash -c to re-evaluate the env var per pass):
#   bash run_nsys_breakdown.sh /tmp/out -- \
#     bash -c 'uv run python -m torch.distributed.run --nproc-per-node 8 \
#       pretrain_gpt.py \
#         $([ "$DETERMINISM_PERF_MODE" = det ] && echo --deterministic-mode) \
#         --profile --profile-step-start 4 --profile-step-end 7 <other args ...>'
#
# For a recipe-driven CI invocation, see
# tests/test_utils/recipes/h100/determinism-perf.yaml which inlines its
# own per-mode loop (cleaner than the bash -c indirection above).
set -euo pipefail
OUT=$(realpath "${1:?usage: $0 OUTDIR -- CMD...}"); shift
[ "${1:-}" = "--" ] || { echo "expected --"; exit 64; }; shift
mkdir -p "$OUT"

for MODE in det nondet; do
  DETERMINISM_PERF_MODE=$MODE \
    nsys profile -t cuda,nvtx -f true \
      --capture-range=cudaProfilerApi --capture-range-end=stop-shutdown \
      -o "$OUT/nsys-$MODE" "$@"
  nsys stats --force-export=true --report nvtx_sum --format csv "$OUT/nsys-$MODE.nsys-rep" > "$OUT/nsys-$MODE.csv"
done

python "$(dirname "$0")/print_nsys_leaderboard.py" "$OUT"
