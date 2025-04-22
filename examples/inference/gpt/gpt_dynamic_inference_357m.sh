# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

# Run dynamic batching inference on the 357M GPT model.

set -u

pip install simpy
pip install sentencepiece
pip install tiktoken

export CUDA_DEVICE_MAX_CONNECTIONS=1

: ${CHECKPOINT_DIR:?"CHECKPOINT_DIR is not set"}
: ${VOCAB_FILE:?"VOCAB_FILE is not set"}
: ${MERGE_FILE:?"MERGE_FILE is not set"}

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
    --exit-on-missing-checkpoint \
    --transformer-impl local \
    --load ${CHECKPOINT_DIR} \
    --tokenizer-type GPT2BPETokenizer \
    --vocab-file ${VOCAB_FILE} \
    --merge-file ${MERGE_FILE} \
    --exit-on-missing-checkpoint \
    --max-position-embeddings 2048 \
    --seq-length 2048 \
    --tensor-model-parallel-size 1 \
    --pipeline-model-parallel-size 1 \
    --num-layers 24 \
    --num-attention-heads 16 \
    --hidden-size 1024 \
    --bf16 \
    --micro-batch-size 1 \
    --attention-dropout 0.0 \
    --hidden-dropout 0.0 \
    --seed 42 \
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
