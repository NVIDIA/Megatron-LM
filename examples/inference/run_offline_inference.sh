#!/bin/bash
# Offline inference launcher for the Megatron high-level API examples.
#
# Requires `simpy` (used by examples/inference/utils.py for synthetic request
# arrival simulation). If it is not already installed:
#   pip install simpy
#
# Required CLI args:
#   --hf-token <token>     Hugging Face token for tokenizer downloads.
#   --checkpoint <path>    Path to the Megatron checkpoint passed as --load.
#
# Optional CLI args:
#   --mode sync|async      Selects MegatronLLM vs MegatronAsyncLLM (default: sync).
#   --use-coordinator      Run in coordinator mode (default: direct).
#   --nproc <n>            Number of processes (default: 8).
#
# Examples:
#   sync + direct (defaults):
#     bash run_offline_inference.sh --hf-token hf_xxx --checkpoint /path/to/ckpt
#   sync + coordinator:
#     bash run_offline_inference.sh --hf-token hf_xxx --checkpoint /path/to/ckpt --use-coordinator
#   async + direct:
#     bash run_offline_inference.sh --hf-token hf_xxx --checkpoint /path/to/ckpt --mode async
#   async + coordinator:
#     bash run_offline_inference.sh --hf-token hf_xxx --checkpoint /path/to/ckpt --mode async --use-coordinator

HF_TOKEN=""
CHECKPOINT=""
MODE="sync"
USE_COORDINATOR=0
NPROC=8

while [[ $# -gt 0 ]]; do
    case "$1" in
        --hf-token)
            HF_TOKEN="$2"
            shift 2
            ;;
        --checkpoint)
            CHECKPOINT="$2"
            shift 2
            ;;
        --mode)
            MODE="$2"
            shift 2
            ;;
        --use-coordinator)
            USE_COORDINATOR=1
            shift
            ;;
        --nproc)
            NPROC="$2"
            shift 2
            ;;
        -h|--help)
            sed -n '2,26p' "$0"
            exit 0
            ;;
        *)
            echo "Unknown argument: $1" >&2
            echo "Run with -h for usage." >&2
            exit 1
            ;;
    esac
done

if [[ -z "$HF_TOKEN" ]]; then
    echo "Error: --hf-token is required" >&2
    exit 1
fi
if [[ -z "$CHECKPOINT" ]]; then
    echo "Error: --checkpoint is required" >&2
    exit 1
fi
if [[ "$MODE" != "sync" && "$MODE" != "async" ]]; then
    echo "Invalid --mode='$MODE'; expected 'sync' or 'async'." >&2
    exit 1
fi

export HF_TOKEN

EXTRA_ARGS=""
if [[ "$USE_COORDINATOR" == "1" ]]; then
    EXTRA_ARGS="$EXTRA_ARGS --use-coordinator"
fi
if [[ "$MODE" == "async" ]]; then
    EXTRA_ARGS="$EXTRA_ARGS --async-mode"
fi

torchrun --nproc-per-node "$NPROC" \
    -m examples.inference.offline_inference $EXTRA_ARGS \
    --load "$CHECKPOINT" \
    --bf16 \
    --tensor-model-parallel-size 1 \
    --micro-batch-size 64 \
    --dist-ckpt-strictness log_unexpected \
    --inference-rng-tracker \
    --cuda-graph-impl local \
    --decode-only-cuda-graphs \
    --tokenizer-type HuggingFaceTokenizer \
    --tokenizer-model Qwen/Qwen2.5-1.5B \
    --no-use-tokenizer-model-from-checkpoint-args \
    --num-layers 28 \
    --hidden-size 1536 \
    --num-attention-heads 12 \
    --max-position-embeddings 32768 \
    --num-query-groups 2 \
    --group-query-attention \
    --swiglu \
    --normalization RMSNorm \
    --disable-bias-linear \
    --position-embedding-type rope \
    --rotary-percent 1.0 \
    --rotary-base 1000000 \
    --seq-length 32768 \
    --ffn-hidden-size 8960
