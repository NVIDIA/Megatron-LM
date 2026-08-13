#!/bin/bash
# Reproduce LRU prefix-cache exhaustion while a block-aligned request resumes.
#
# This launcher targets the nemo_minitron-0.5b checkpoint used by the
# gpt_dynamic_inference_tp1_pp1_583m_prefix_caching_lru functional test.
#
# On an unpatched main checkout, the third request exits with:
#   AssertionError: active_request_count == 0 with no hidden chunked prefill.
#
# On a checkout containing the resume accounting fix, all four requests finish.
#
# Usage:
#   bash examples/inference/advanced/run_prefix_cache_lru_resume_repro.sh \
#     --checkpoint /path/to/nemo_minitron-0.5b/v1 \
#     --tokenizer-model /path/to/multiMixV8.gpt4o_nc_sd.500000.128k.vocab.json

set -euo pipefail

CHECKPOINT=""
TOKENIZER_MODEL=""
OUTPUT_PATH="${TMPDIR:-/tmp}/prefix-cache-lru-resume-results.json"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --checkpoint)
            CHECKPOINT="$2"
            shift 2
            ;;
        --tokenizer-model)
            TOKENIZER_MODEL="$2"
            shift 2
            ;;
        --output)
            OUTPUT_PATH="$2"
            shift 2
            ;;
        -h|--help)
            sed -n '2,15p' "$0"
            exit 0
            ;;
        *)
            echo "Unknown argument: $1" >&2
            echo "Run with --help for usage." >&2
            exit 1
            ;;
    esac
done

if [[ -z "$CHECKPOINT" ]]; then
    echo "Error: --checkpoint is required." >&2
    exit 1
fi
if [[ -z "$TOKENIZER_MODEL" ]]; then
    echo "Error: --tokenizer-model is required." >&2
    exit 1
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO_ROOT"

if [[ -z "${PYTHON_BIN:-}" ]]; then
    if [[ -x /opt/venv/bin/python ]]; then
        PYTHON_BIN=/opt/venv/bin/python
    else
        PYTHON_BIN=python
    fi
fi

REPRO_TMPDIR="$(mktemp -d "${TMPDIR:-/tmp}/prefix-cache-lru-resume.XXXXXX")"
trap 'rm -rf "$REPRO_TMPDIR"' EXIT
PROMPT_FILE="$REPRO_TMPDIR/prompts.jsonl"

# Each prompt is unique and round-trips through the production tokenizer to
# exactly one 256-token KV block.
"$PYTHON_BIN" - "$TOKENIZER_MODEL" "$PROMPT_FILE" <<'PY'
import json
import sys

from megatron.core.tokenizers.text.libraries.tiktoken_tokenizer import TikTokenTokenizer

tokenizer = TikTokenTokenizer(sys.argv[1], pattern="v2")
with open(sys.argv[2], "w", encoding="utf-8") as prompt_file:
    for request_idx in range(4):
        source = f"unique prefix-cache request {request_idx}: " + "hi " * 1024
        token_ids = tokenizer.text_to_ids(source)[:256]
        text = tokenizer.ids_to_text(token_ids)
        assert len(tokenizer.text_to_ids(text)) == 256
        prompt_file.write(json.dumps({"text": text}) + "\n")
PY

export CUDA_DEVICE_MAX_CONNECTIONS=1
export NVTE_ALLOW_NONDETERMINISTIC_ALGO=0
export NCCL_ALGO=Ring
export CUBLAS_WORKSPACE_CONFIG=:4096:8

"$PYTHON_BIN" -m torch.distributed.run \
    --nproc-per-node 1 \
    examples/inference/advanced/gpt_dynamic_inference.py \
    --tiktoken-pattern v2 \
    --use-mcore-models \
    --tokenizer-type TikTokenizer \
    --tokenizer-model "$TOKENIZER_MODEL" \
    --auto-detect-ckpt-format \
    --max-tokens-to-oom 3600000 \
    --inference-max-seq-length 4096 \
    --attention-backend flash \
    --use-checkpoint-args \
    --micro-batch-size 1 \
    --no-load-optim \
    --no-use-tokenizer-model-from-checkpoint-args \
    --timing-log-level 0 \
    --load "$CHECKPOINT" \
    --distributed-backend nccl \
    --log-interval 1 \
    --transformer-impl transformer_engine \
    --tensor-model-parallel-size 1 \
    --pipeline-model-parallel-size 1 \
    --deterministic-mode \
    --ckpt-format torch_dist \
    --bf16 \
    --num-layers 24 \
    --hidden-size 1152 \
    --num-attention-heads 16 \
    --max-position-embeddings 1024 \
    --seq-length 1024 \
    --temperature 1.0 \
    --top_k 1 \
    --num-tokens-to-generate 2 \
    --termination-id -1 \
    --inference-dynamic-batching-buffer-size-gb 0.1 \
    --inference-dynamic-batching-block-size 256 \
    --inference-dynamic-batching-max-requests 1 \
    --inference-dynamic-batching-max-tokens 256 \
    --inference-dynamic-batching-prefix-caching \
    --inference-dynamic-batching-prefix-caching-eviction-policy lru \
    --dist-ckpt-strictness log_unexpected \
    --inference-ckpt-non-strict \
    --output-path "$OUTPUT_PATH" \
    --prompt-file "$PROMPT_FILE" \
    --prompt-file-num-truncate 4 \
    --incoming-requests-per-step 1 \
    --inference-repeat-n 1 \
    --inference-logging-step-interval 1 \
    --drain-between-batches \
    --batch-boundaries 0,1,2,3
