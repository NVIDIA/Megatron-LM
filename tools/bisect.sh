#!/usr/bin/env bash
# bisect.sh — checkout a commit, run a functional test to generate golden values,
# copy them into the test case directory, then run the test again to validate.
#
# Usage:
#   ./bisect.sh [--second-run] <COMMIT> <MODEL> <TESTCASE>
#
# Options:
#   --second-run   Also copy the generated golden values and re-run the test
#                  to validate them (steps 6 and 7).  Omit to stop after the
#                  first run (steps 1–5 only).
#
# Example:
#   ./bisect.sh abc1234 gpt gpt3_mcore_te_tp2_pp1_resume_torch_dist_cp2_nondeterministic
#   ./bisect.sh --second-run abc1234 gpt gpt3_mcore_te_tp2_pp1_resume_torch_dist_cp2_nondeterministic

set -uo pipefail

# ---------------------------------------------------------------------------
# Parse --second-run flag (optional, must come first)
# ---------------------------------------------------------------------------
SECOND_RUN=false
if [[ "${1:-}" == "--second-run" ]]; then
    SECOND_RUN=true
    shift
fi

COMMIT="${1:?Usage: $0 [--second-run] <COMMIT> <MODEL> <TESTCASE>}"
MODEL="${2:?Usage: $0 [--second-run] <COMMIT> <MODEL> <TESTCASE>}"
TESTCASE="${3:?Usage: $0 [--second-run] <COMMIT> <MODEL> <TESTCASE>}"
# Commits cherry-picked onto the tested commit before running the test.
# Applied in order; add entries here to layer additional fixes on every bisect step.
CHERRYPICK_SHAS=(
    "bfa0f308aa5f2df76eb24e6b9fb86de5b39b5334"
)
TEST_SCRIPT="test_cases/${MODEL}/${TESTCASE}.sh"
GOLDEN_DIR="tests/functional_tests/test_cases/${MODEL}/${TESTCASE}"
LOG_FILE="log.txt"

REPO_ROOT="$(git rev-parse --show-toplevel)"

log() { echo "[update-golden-values $(date +%H:%M:%S)] $*"; }
die() { log "FATAL: $*"; exit 1; }

cd "${REPO_ROOT}"

#1. Stash any local changes
log "Stashing local changes..."
git stash

#2. Fetch and checkout the target commit
log "Fetching and checking out ${COMMIT}..."
git fetch origin "${COMMIT}"
git checkout "${COMMIT}"

#3. Cherry-pick each patch commit onto the tested commit, in order
log "Cherry-picking ${#CHERRYPICK_SHAS[@]} commit(s)..."
for sha in "${CHERRYPICK_SHAS[@]}"; do
   log "  cherry-pick ${sha}..."
   git fetch origin "${sha}"
   git cherry-pick "${sha}"
done

python -m tests.test_utils.python_scripts.generate_local_jobs --environment dev --scope mr

# 4. Run the test and tee output to log.txt
log "Running ${TEST_SCRIPT} (output → ${LOG_FILE})..."
bash "${TEST_SCRIPT}" 2>&1 | tee -a "${LOG_FILE}"

# 5. Parse OUTPUT_PATH from the last few lines of log.txt
log "Extracting OUTPUT_PATH from ${LOG_FILE}..."
OUTPUT_PATH="$(tail -20 "${LOG_FILE}" \
    | grep 'This test wrote results into' \
    | tail -1 \
    | sed 's/.*This test wrote results into //')"

[[ -n "${OUTPUT_PATH}" ]] || die "Could not extract OUTPUT_PATH from ${LOG_FILE}"
log "OUTPUT_PATH=${OUTPUT_PATH}"

if [[ "${SECOND_RUN}" == false ]]; then
    log "Stopping after first run (pass --second-run to copy golden values and re-validate)."
    exit 0
fi

# 6. Copy golden_values*.json into the test case directory
log "Copying golden values from ${OUTPUT_PATH} → ${GOLDEN_DIR}..."
mkdir -p "${GOLDEN_DIR}"
cp "${OUTPUT_PATH}"/golden_values*.json "${GOLDEN_DIR}/"
log "Copied: $(ls "${GOLDEN_DIR}"/golden_values*.json)"

# 7. Run the test again to validate against the new golden values
log "Re-running ${TEST_SCRIPT} to validate..."
bash "${TEST_SCRIPT}" 2>&1 | tee -a "${LOG_FILE}"

log "Done."
