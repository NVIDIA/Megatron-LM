#!/bin/bash
# OpenAI-compatible inference server launcher for the Megatron high-level API.
#
# Required CLI args:
#   --hf-token <token>   Hugging Face token for tokenizer downloads.
#   --hf-home <path>     Hugging Face cache directory.
#   --checkpoint <path>  Path to the Megatron checkpoint passed as --load.
#
# Optional CLI args:
#   --nproc <n>          Number of processes (default: 8).
#
# Example:
#   bash run_inference_server.sh \
#     --hf-token hf_xxx \
#     --hf-home /path/to/hf_home \
#     --checkpoint /path/to/ckpt

HF_TOKEN=""
HF_HOME=""
CHECKPOINT=""
NPROC=8

while [[ $# -gt 0 ]]; do
    case "$1" in
        --hf-token)
            HF_TOKEN="$2"
            shift 2
            ;;
        --hf-home)
            HF_HOME="$2"
            shift 2
            ;;
        --checkpoint)
            CHECKPOINT="$2"
            shift 2
            ;;
        --nproc)
            NPROC="$2"
            shift 2
            ;;
        -h|--help)
            sed -n '2,16p' "$0"
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
if [[ -z "$HF_HOME" ]]; then
    echo "Error: --hf-home is required" >&2
    exit 1
fi
if [[ -z "$CHECKPOINT" ]]; then
    echo "Error: --checkpoint is required" >&2
    exit 1
fi

export HF_TOKEN
export HF_HOME
# Required by Megatron when using tensor or context parallelism.
export CUDA_DEVICE_MAX_CONNECTIONS=1

torchrun --nproc-per-node "$NPROC" \
    -m examples.inference.launch_inference_server \
    --tensor-model-parallel-size 2 \
    --expert-tensor-parallel-size 1 \
    --expert-model-parallel-size 8 \
    --sequence-parallel \
    --pipeline-model-parallel-size 1 \
    --inference-max-seq-length 4096 \
    --load "$CHECKPOINT" \
    --micro-batch-size 1 \
    --moe-router-dtype fp32 \
    --moe-token-dispatcher-type alltoall \
    --use-checkpoint-args \
    --bf16 \
    --attention-backend flash \
    --transformer-impl inference_optimized \
    --te-rng-tracker \
    --inference-rng-tracker \
    --cuda-graph-impl "local" \
    --dist-ckpt-strictness log_unexpected \
    --inference-dynamic-batching-buffer-size-gb 20 \
    --model-provider hybrid \
    --inference-dynamic-batching-max-tokens 2048 \
    --enable-chunked-prefill \
    --inference-logging-step-interval 50 \
    --inference-dynamic-batching-num-cuda-graphs -1 \
    --cuda-graph-scope full_iteration_inference \
    --inference-dynamic-batching-max-requests 256
