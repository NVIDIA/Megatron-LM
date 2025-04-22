# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

# Run dynamic batching inference on the 12B GPT model.

set -u

pip install simpy
pip install sentencepiece
pip install tiktoken

export CUDA_DEVICE_MAX_CONNECTIONS=1

: ${CHECKPOINT_DIR:?"CHECKPOINT_DIR is not set"}
: ${TOKENIZER_MODEL:?"TOKENIZER_MODEL is not set"}

: ${NUM_TOKENS_TO_PROMPT="8 32"}
: ${NUM_TOKENS_TO_GENERATE=256}
: ${INCOMING_REQUESTS_DURATION=10.}
: ${INCOMING_REQUESTS_PER_SEC=100.}

: ${INFERENCE_DYNAMIC_BATCHING_BUFFER_SIZE_GB=50.}
: ${INFERENCE_DYNAMIC_BATCHING_BUFFER_OVERFLOW_FACTOR=1.}
: ${INFERENCE_DYNAMIC_BATCHING_BUFFER_GUARANTEED_FRACTION=0.05}

: ${ENGINE=dynamic}
# NSIGHT_PREFIX=/path/to/nsight/profile

# --inference-rng-tracker \ # ... re-add after bugfix.
ARGS=" \
    --no-persist-layer-norm \
    --apply-layernorm-1p \
    --no-position-embedding \
    --group-query-attention \
    --num-query-groups 8 \
    --load ${CHECKPOINT_DIR} \
    --use-checkpoint-args \
    --untie-embeddings-and-output-weights \
    --disable-bias-linear \
    --use-rotary-position-embeddings \
    --position-embedding-type rope \
    --rotary-base 1000000 \
    --rotary-percent 1.0 \
    --swiglu \
    --normalization RMSNorm \
    --attention-dropout 0.0 \
    --hidden-dropout 0.0 \
    --exit-duration-in-mins 5740 \
    --tensor-model-parallel-size 1 \
    --pipeline-model-parallel-size 1 \
    --num-layers 40 \
    --hidden-size 5120 \
    --ffn-hidden-size 14336 \
    --num-attention-heads 32 \
    --kv-channels 128 \
    --seq-length 1024 \
    --max-position-embeddings 1024 \
    --micro-batch-size 64 \
    --bf16 \
    --tokenizer-type TikTokenizer \
    --tiktoken-pattern v2 \
    --tokenizer-model ${TOKENIZER_MODEL} \
    --distributed-timeout-minutes 2400 \
    --transformer-impl local \
    --use-flash-attn \
    \
    --inference-dynamic-batching \
    --inference-dynamic-batching-buffer-size-gb ${INFERENCE_DYNAMIC_BATCHING_BUFFER_SIZE_GB} \
    --inference-dynamic-batching-buffer-overflow-factor ${INFERENCE_DYNAMIC_BATCHING_BUFFER_OVERFLOW_FACTOR} \
    --inference-dynamic-batching-buffer-guaranteed-fraction ${INFERENCE_DYNAMIC_BATCHING_BUFFER_GUARANTEED_FRACTION} \
    \
    --enable-cuda-graph \
"

if [[ -v PROMPTS ]]; then
    ARGS+=" --prompts ${PROMPTS}"
else
    ARGS+=" \
        --num-tokens-to-prompt ${NUM_TOKENS_TO_PROMPT} \
        --num-tokens-to-generate ${NUM_TOKENS_TO_GENERATE} \
        --incoming-requests-duration ${INCOMING_REQUESTS_DURATION} \
        --incoming-requests-per-sec ${INCOMING_REQUESTS_PER_SEC} \
    "
fi

CMD="python -m examples.inference.gpt.gpt_${ENGINE}_inference ${ARGS}"
if [[ -v NSIGHT_PREFIX ]]; then
    CMD="nsys profile -t cuda,nvtx,mpi -s none --wait=primary --show-output=true --force-overwrite=true --export=sqlite -o ${NSIGHT_PREFIX} ${CMD}"
fi

eval ${CMD}
