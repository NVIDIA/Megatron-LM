# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

# Run dynamic batching inference on the 357M GPT model.

set -u

# Libraries.
pip install simpy
pip install sentencepiece
pip install tiktoken

# Environment variables.
export CUDA_DEVICE_MAX_CONNECTIONS=1

# Checkpoint.
: ${CHECKPOINT_DIR:?"CHECKPOINT_DIR is not set"}
: ${VOCAB_FILE:?"VOCAB_FILE is not set"}
: ${MERGE_FILE:?"MERGE_FILE is not set"}

# Prompts.
: ${NUM_TOKENS_TO_PROMPT="8 32"}
: ${NUM_TOKENS_TO_GENERATE=256}
: ${INCOMING_REQUESTS_DURATION=10.}
: ${INCOMING_REQUESTS_PER_SEC=100.}

# Dynamic context.
: ${BUFFER_SIZE_GB=50.}
: ${BUFFER_OVERFLOW_FACTOR=1.}
: ${BUFFER_GUARANTEED_FRACTION=0.05}

# Cuda graphs.
: ${CUDA_GRAPH_IMPL=local}
: ${NUM_CUDA_GRAPHS=16}
: ${CUDA_GRAPH_SHARE_IO_BUFFERS=1}

# Miscellaneous.
: ${USE_COORDINATOR=0}
: ${ENGINE=dynamic}
: ${EXTRA_ARGS=""}
# NSIGHT_PREFIX=/path/to/nsight/profile

# Arguments.
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
    --inference-rng-tracker \
    \
    --inference-dynamic-batching \
    --inference-dynamic-batching-buffer-size-gb ${BUFFER_SIZE_GB} \
    --inference-dynamic-batching-buffer-overflow-factor ${BUFFER_OVERFLOW_FACTOR} \
    --inference-dynamic-batching-buffer-guaranteed-fraction ${BUFFER_GUARANTEED_FRACTION} \
    \
    ${EXTRA_ARGS} \
"

# Cuda graphs.
if [ "${NUM_CUDA_GRAPHS}" != "0" ]; then
    ARGS+=" \
        --cuda-graph-impl local \
        --inference-dynamic-batching-num-cuda-graphs ${NUM_CUDA_GRAPHS} \
    "
fi

# Prompts.
if [[ -v PROMPTS ]]; then
    ARGS+=" \
        --prompts ${PROMPTS} \
        --num-tokens-to-generate ${NUM_TOKENS_TO_GENERATE} \
    "
else
    ARGS+=" \
        --num-tokens-to-prompt ${NUM_TOKENS_TO_PROMPT} \
        --num-tokens-to-generate ${NUM_TOKENS_TO_GENERATE} \
        --incoming-requests-duration ${INCOMING_REQUESTS_DURATION} \
        --incoming-requests-per-sec ${INCOMING_REQUESTS_PER_SEC} \
    "
fi

# Command.
if [[ "${USE_COORDINATOR}" == "0" ]]; then
    CMD="python -m examples.inference.gpt.gpt_${ENGINE}_inference ${ARGS}"
else
    CMD="python -um examples.inference.gpt.gpt_${ENGINE}_inference_with_coordinator ${ARGS}"
fi

if [[ -v NSIGHT_PREFIX ]]; then
    CMD="nsys profile -s none -t nvtx,cuda --cudabacktrace=all --cuda-graph-trace=node --python-backtrace=cuda --wait all -o ${NSIGHT_PREFIX} --force-overwrite true --capture-range=cudaProfilerApi --capture-range-end=stop ${CMD}"
fi

echo "~~~"
echo "CMD ... ${CMD}."
echo "~~~"
eval ${CMD}
