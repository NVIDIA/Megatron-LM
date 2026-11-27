#!/bin/bash
# Det-vs-nondet perf breakdown per determinism test: any op, kernel or layer that a test
# under tests/unit_tests/determinism/kernels replays. Each test gets one NVTX range per
# phase (forward, Backward); the leaderboard compares the two modes range by range.
# Usage: bash tests/performance_tests/shell_test_utils/determinism/kernel_breakdown.sh \
#          LEADERBOARD_DIR [PYTEST_ARGS...]   (default: the whole kernels directory)
# A "det slower/faster" row means the per-call ranges of ONE run do not overlap. Run it again
# with DETERMINISM_PERF_ORDER="nondet det" and keep only rows that agree in both orders.
set -euo pipefail

OUT="${1:?usage: $0 LEADERBOARD_DIR [PYTEST_ARGS...]}"; shift
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
[ $# -gt 0 ] || set -- tests/unit_tests/determinism/kernels

# The entry point run_nsys_breakdown.sh asks for: it brackets the tests with
# cudaProfilerStart/Stop and exits 0, so the nondet pass still runs after a failure. Each
# pass writes its pytest outcome to OUT/pytest-<mode>.{xml,rc}; a det-pass failure fails
# this script after the leaderboard is printed (a nondet failure is a finding, not an error).
mkdir -p "$OUT"
export KERNEL_BREAKDOWN_OUT="$(realpath "$OUT")"
RUN='import os, sys, pytest, torch
out, mode = os.environ["KERNEL_BREAKDOWN_OUT"], os.environ["DETERMINISM_PERF_MODE"]
torch.cuda.profiler.start()
rc = pytest.main(["-q", "-p", "no:cacheprovider", "--no-header", "-o", "faulthandler_timeout=600",
                  f"--junitxml={out}/pytest-{mode}.xml", *sys.argv[1:]])
torch.cuda.synchronize()
torch.cuda.profiler.stop()
open(f"{out}/pytest-{mode}.rc", "w").write(f"{int(rc)}\n")'

# No CPU sampling: for a whole pytest run it makes a report that takes minutes to export.
# Median of the traced calls (3-6 of each workload; see harness._perf_range), all rows.
export NSYS_PROFILE_ARGS="${NSYS_PROFILE_ARGS:---sample=none --cpuctxsw=none}"
export NSYS_LEADERBOARD_COLUMN="${NSYS_LEADERBOARD_COLUMN:-Med}"
export NSYS_LEADERBOARD_TOP_N="${NSYS_LEADERBOARD_TOP_N:-0}"
bash "$SCRIPT_DIR/run_nsys_breakdown.sh" "$OUT" -- python -c "$RUN" "$@"

for MODE in det nondet; do
  echo "pytest $MODE pass: exit $(cat "$KERNEL_BREAKDOWN_OUT/pytest-$MODE.rc" 2>/dev/null || echo missing)"
done
# det: every test must pass. nondet: failed tests (exit 1) are findings, listed in
# pytest-nondet.xml; an interrupted, crashed or empty run (any other code) is an error.
[ "$(cat "$KERNEL_BREAKDOWN_OUT/pytest-det.rc" 2>/dev/null)" = "0" ] || { echo "FAIL: det pass did not pass"; exit 1; }
case "$(cat "$KERNEL_BREAKDOWN_OUT/pytest-nondet.rc" 2>/dev/null)" in
  0|1) ;;
  *) echo "FAIL: nondet pass did not complete"; exit 1 ;;
esac
