#!/usr/bin/env bash
# bisect.sh — called by: git bisect run ./bisect.sh
#
# Bisects gpt3_mcore_te_tp2_pp1_resume_torch_dist_cp2_nondeterministic to find
# the first commit that broke checkpoint-resume reproducibility (lm loss / num-zeros mismatch).
#
# Two-phase approach per step:
#   1. First run  → generates actual golden values for this commit's training
#   2. Copy them  → makes the "run-vs-golden" pytest check auto-pass
#   3. Second run → only the checkpoint-resume (run-1 vs run-2) check can fail
#
# Exit codes: 0=good  1=bad  125=skip (training crashed, no output produced)
#
# Setup (run once inside the pod):
#   git bisect start
#   git bisect bad upstream/main
#   git bisect good 696f164de1076f46c87c904e58ca293108459572
#   git bisect run ./bisect.sh

set -uo pipefail

REPO_ROOT="$(git rev-parse --show-toplevel)"
MODEL="gpt"
TESTCASE="gpt3_mcore_te_tp2_pp1_resume_torch_dist_cp2_nondeterministic"
GOLDEN_VALUES_DIR="${REPO_ROOT}/tests/functional_tests/test_cases/${MODEL}/${TESTCASE}"
GOLDEN_VALUES_FILE="${GOLDEN_VALUES_DIR}/golden_values_dev_dgx_h100.json"
GENERATED_SCRIPT="${REPO_ROOT}/test_cases/${MODEL}/${TESTCASE}.sh"

# This commit carries the correct threshold logic in the three test-infrastructure files.
# It always exists in the git object DB regardless of the currently checked-out commit.
FIX_COMMIT="761547c73129c96ad54e8a1d240ca5e3352044b5"

COMMON_PY="tests/functional_tests/python_test_utils/common.py"
PIPELINE_PY="tests/functional_tests/python_test_utils/test_pretraining_regular_pipeline.py"
RESUME_PY="tests/functional_tests/python_test_utils/test_pretraining_resume_checkpoint_pipeline.py"

LOG_DIR="/tmp/bisect_logs"
CURRENT_COMMIT="$(git rev-parse HEAD)"
LOG_FILE="${LOG_DIR}/${CURRENT_COMMIT}.log"

mkdir -p "${LOG_DIR}"

log() { echo "[bisect $(date +%H:%M:%S)] $*" | tee -a "${LOG_FILE}"; }

log "=== Bisecting commit: ${CURRENT_COMMIT} ==="

# Restore all modified tracked files on exit so git bisect can checkout the next commit cleanly.
cleanup() {
    git -C "${REPO_ROOT}" checkout -- \
        "${COMMON_PY}" \
        "${PIPELINE_PY}" \
        "${RESUME_PY}" \
        "${GOLDEN_VALUES_FILE}" \
        2>/dev/null || true
    log "Restored modified tracked files."
}
trap cleanup EXIT

cd "${REPO_ROOT}"

# ---------------------------------------------------------------
# Overlay fixed test infrastructure
#
# Commits between 696f164de1 and adc69dba2d have broken/incompatible
# versions of these files. We always apply the fixed versions so that
# the only source of failure is the actual checkpoint-resume behaviour.
# ---------------------------------------------------------------
overlay_test_infra() {
    log "Overlaying fixed test infrastructure from ${FIX_COMMIT}..."
    git show "${FIX_COMMIT}:${COMMON_PY}"    > "${REPO_ROOT}/${COMMON_PY}"
    git show "${FIX_COMMIT}:${PIPELINE_PY}"  > "${REPO_ROOT}/${PIPELINE_PY}"
    git show "${FIX_COMMIT}:${RESUME_PY}"    > "${REPO_ROOT}/${RESUME_PY}"
}

# ---------------------------------------------------------------
# Generate (or re-generate) the test launch script.
# Each call produces a fresh UUID in OUTPUT_PATH.
# ---------------------------------------------------------------
generate_test_script() {
    python -m tests.test_utils.python_scripts.generate_local_jobs \
        --environment dev \
        --scope mr \
        --test-case "${TESTCASE}" \
        2>&1 | tee -a "${LOG_FILE}" || {
        log "ERROR: generate_local_jobs failed. Skipping commit."
        exit 125
    }
    # Override N_REPEAT: default is 5 (from recipe); 1 is enough for bisect
    sed -i 's/"N_REPEAT=[0-9]*"/"N_REPEAT=1"/' "${GENERATED_SCRIPT}"
}

overlay_test_infra
generate_test_script

# ===============================================================
# FIRST RUN — produces the actual golden values for this commit
# ===============================================================
log "=== FIRST RUN ==="
FIRST_EXIT=0
bash "${GENERATED_SCRIPT}" \
    2>&1 | tee "/tmp/first_run_${CURRENT_COMMIT}.log" | tee -a "${LOG_FILE}" \
    || FIRST_EXIT=$?
log "First run exit code: ${FIRST_EXIT}"

FIRST_OUTPUT_PATH="$(grep 'This test wrote results into' "/tmp/first_run_${CURRENT_COMMIT}.log" \
    | tail -1 | awk '{print $NF}')"
log "First run output path: ${FIRST_OUTPUT_PATH}"

FIRST_JSON="${FIRST_OUTPUT_PATH}/golden_values_dev_dgx_h100.json"
if [[ -z "${FIRST_OUTPUT_PATH}" || ! -f "${FIRST_JSON}" ]]; then
    log "ERROR: No golden values JSON found — training likely crashed. Skipping commit."
    exit 125
fi

# ---------------------------------------------------------------
# Swap golden values: replace repo's golden values with the actual
# values produced by this commit's run 1.  The second run's
# "run-vs-golden" pytest check will now auto-pass, leaving only the
# checkpoint-resume (run-1 vs run-2) check to determine good/bad.
# ---------------------------------------------------------------
log "Updating golden values from first run output..."
cp "${FIRST_JSON}" "${GOLDEN_VALUES_FILE}"

# Re-overlay (generate_local_jobs doesn't touch these, but be safe)
# and re-generate to get a fresh OUTPUT_PATH UUID for the second run.
overlay_test_infra
generate_test_script

# ===============================================================
# SECOND RUN — good/bad verdict
# Only the checkpoint-resume check (run-1 vs run-2) can fail here.
# ===============================================================
log "=== SECOND RUN ==="
SECOND_EXIT=0
bash "${GENERATED_SCRIPT}" \
    2>&1 | tee "/tmp/second_run_${CURRENT_COMMIT}.log" | tee -a "${LOG_FILE}" \
    || SECOND_EXIT=$?
log "Second run exit code: ${SECOND_EXIT}"

if [[ "${SECOND_EXIT}" -eq 0 ]]; then
    log "=== VERDICT: GOOD (checkpoint-resume reproducible) ==="
else
    log "=== VERDICT: BAD  (checkpoint-resume mismatch) ==="
fi

# EXIT trap fires → restores all 4 modified tracked files
exit "${SECOND_EXIT}"
