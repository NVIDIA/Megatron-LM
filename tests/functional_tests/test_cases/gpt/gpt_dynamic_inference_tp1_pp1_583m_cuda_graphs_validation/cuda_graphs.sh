# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

set -u

# Libraries.
uv pip install simpy
uv pip install tiktoken

# Environment variables.
export CUDA_DEVICE_MAX_CONNECTIONS=1
export NVTE_ALLOW_NONDETERMINISTIC_ALGO=0
export NCCL_ALGO=Ring
export CUBLAS_WORKSPACE_CONFIG=:4096:8

# Required variables.
: ${CHECKPOINT_DIR:?"CHECKPOINT_DIR is not set."}
: ${TOKENIZER_MODEL:?"TOKENIZER_MODEL is not set."}
: ${NUM_CUDA_GRAPHS:?"NUM_CUDA_GRAPHS is not set."}
: ${OUTPUT_PATH:?"OUTPUT_PATH is not set."}

# Prompts.
: ${NUM_TOKENS_TO_PROMPT="4 16"}
: ${NUM_TOKENS_TO_GENERATE=256}
: ${INCOMING_REQUESTS_DURATION=10.}
: ${INCOMING_REQUESTS_PER_SEC=200.} # used only for generating 10*200=2000 synthetic requests
: ${INCOMING_REQUESTS_PER_STEP:?"INCOMING_REQUESTS_PER_STEP is not set."}

# Dynamic context.
: ${BUFFER_SIZE_GB=50.}
: ${BUFFER_OVERFLOW_FACTOR=1.}
: ${BUFFER_GUARANTEED_FRACTION=0.05}

# Cuda graphs.
: ${NUM_CUDA_GRAPHS=16}

# Arguments.
ARGS=" \
  --tiktoken-pattern v2 \
  --use-mcore-models \
  --tokenizer-type TikTokenizer \
  --tokenizer-model ${TOKENIZER_MODEL} \
  --auto-detect-ckpt-format \
  --max-tokens-to-oom 3600000 \
  --inference-max-seq-length 4096 \
  --attention-backend flash \
  --use-checkpoint-args \
  --micro-batch-size 1 \
  --no-load-optim \
  --no-use-tokenizer-model-from-checkpoint-args \
  --timing-log-level 2 \
  --load ${CHECKPOINT_DIR} \
  --distributed-backend nccl \
  --log-interval 1 \
  --transformer-impl transformer_engine \
  --tensor-model-parallel-size 1 \
  --pipeline-model-parallel-size 1 \
  --deterministic-mode \
  --ckpt-format torch_dist \
  --bf16 \
  --log-memory-to-tensorboard \
  --log-num-zeros-in-grad \
  --log-validation-ppl-to-tensorboard \
  --log-timers-to-tensorboard \
  --num-layers 24 \
  --hidden-size 1152 \
  --num-attention-heads 16 \
  --max-position-embeddings 1024 \
  --seq-length 1024 \
  --temperature 1.0 \
  --top_k 1 \
  --return-log-probs \
  --dist-ckpt-strictness log_unexpected \
  --inference-ckpt-non-strict \
  \
  --inference-dynamic-batching \
  --inference-dynamic-batching-buffer-size-gb ${BUFFER_SIZE_GB} \
  --inference-dynamic-batching-buffer-overflow-factor ${BUFFER_OVERFLOW_FACTOR} \
  --inference-dynamic-batching-buffer-guaranteed-fraction ${BUFFER_GUARANTEED_FRACTION} \
  \
  --num-tokens-to-prompt ${NUM_TOKENS_TO_PROMPT} \
  --num-tokens-to-generate ${NUM_TOKENS_TO_GENERATE} \
  --incoming-requests-duration ${INCOMING_REQUESTS_DURATION} \
  --incoming-requests-per-sec ${INCOMING_REQUESTS_PER_SEC} \
  --incoming-requests-per-step ${INCOMING_REQUESTS_PER_STEP} \
  \
  --output-path ${OUTPUT_PATH} \
  --output-every-n-results 512 \
"

# Enable cuda graphs.
if [ "${NUM_CUDA_GRAPHS}" != "0" ]; then
    ARGS+="  \
      --enable-cuda-graph \
      --inference-dynamic-batching-num-cuda-graphs ${NUM_CUDA_GRAPHS} \
    "
fi

# Command.
CMD="python -m examples.inference.gpt.gpt_dynamic_inference ${ARGS}"
echo "~~~"
echo "CMD ...${CMD}."
echo "~~~"
eval ${CMD}
