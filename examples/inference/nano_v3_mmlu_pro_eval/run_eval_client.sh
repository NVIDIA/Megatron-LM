#!/usr/bin/env bash
# Run the MMLU-Pro client against one already-running server.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/config.sh"

MODE="${1:?usage: run_eval_client.sh MODE [BASE_URL] [RUN_OUTPUT_DIR] [LIMIT] [BATCH_SIZE]}"
BASE_URL="${2:-http://127.0.0.1:${SERVER_PORT}}"
RUN_OUTPUT_DIR="${3:-${EVAL_OUTPUT_BASE}/manual}"
if (( $# >= 4 )); then
    if [[ "$4" == "__FULL__" ]]; then
        EVAL_LIMIT=""
    else
        EVAL_LIMIT="$4"
    fi
fi
if (( $# >= 5 )); then
    BATCH_SIZE="$5"
fi
MODE_OUTPUT_DIR="${RUN_OUTPUT_DIR}/${MODE}"
MODEL_NAME="${SERVED_MODEL_NAME}"
LM_EVAL_PYTHON="${LM_EVAL_HARNESS_PATH}/env/bin/python"

[[ -x "${LM_EVAL_PYTHON}" ]] || {
    echo "lm-eval Python environment not found: ${LM_EVAL_PYTHON}" >&2
    echo "Install the adlr/nemo5 harness as described in README.md or override LM_EVAL_HARNESS_PATH." >&2
    exit 1
}

mkdir -p "${MODE_OUTPUT_DIR}" "${HF_DATASETS_CACHE}"

ARGS=(
    --model notok-completions
    --model_args "base_url=${BASE_URL},model=${MODEL_NAME}"
    --batch_size="${BATCH_SIZE}"
    --tasks mmlu_pro_cot_mini_5_shot_base
    --num_fewshot=5
    --log_samples
    --output_path "${MODE_OUTPUT_DIR}"
)
# Dynamo currently rejects four explicit OpenAI stop strings. The task's
# fourth string is </s>, which this Megatron TikToken tokenizer already uses
# as its default EOD/termination token, so keep the other three explicit.
if [[ "${MODE}" == "dynamo" ]]; then
    # Preserve the quotes through lm-eval's two shlex.split() passes. Without
    # all three backslashes, OmegaConf parses Q: and Question: as mappings.
    ARGS+=(--override 'generation_kwargs.until=[\\\"Q:\\\",\\\"Question:\\\",\\\"<|im_end|>\\\"]')
fi
if [[ -n "${EVAL_LIMIT}" ]]; then
    ARGS+=(--limit="${EVAL_LIMIT}")
fi

echo "[$(date --iso-8601=seconds)] Running MMLU-Pro: mode=${MODE}, target=${BASE_URL}"
(
    # lm-eval records source provenance with git. Run from its checkout so an
    # SSH client's arbitrary working directory cannot emit a misleading
    # "not a git repository" message after a successful evaluation.
    cd "${LM_EVAL_HARNESS_PATH}"
    "${LM_EVAL_PYTHON}" -m lm_eval "${ARGS[@]}"
) 2>&1 | tee "${MODE_OUTPUT_DIR}/lm_eval.log"
