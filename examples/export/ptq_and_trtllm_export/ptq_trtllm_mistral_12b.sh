#!/bin/bash
set -e

DEFAULT_NAME="/checkpoints/Mistral-NeMo-12B-Base"
NAME="${1:-$DEFAULT_NAME}"

DEFAULT_QUANT_CFG="fp8"
QUANT_CFG="${2:-$DEFAULT_QUANT_CFG}"

# CHANGE THE FOLLOWING IF YOU MOUNT YOUR DATA AND CHECKPOINTS DIFFERENTLY IN THE CONTAINER.
TP="8"
INFERENCE_TP=${TP}
DECODER_TYPE="llama"
CHECKPOINT_LOAD_DIR="${NAME}"

if [ "$QUANT_CFG" = "int4_awq" ]; then
    INFERENCE_TP="1"
fi

additional_options=" \
    --export-quant-cfg ${QUANT_CFG} \
    --export-legacy-megatron \
    --export-te-mcore-model \
    --calib-batch-size 8 \
    --decoder ${DECODER_TYPE} \
    --export-dir /tmp/trtllm_ckpt \
    --inference-tensor-parallel ${INFERENCE_TP} "

# DO NOT CHANGE THE SETTING BELOW UNLESS YOU KNOW WHAT YOU ARE DOING!!!
export CUDA_DEVICE_MAX_CONNECTIONS=1

options=" \
    --untie-embeddings-and-output-weights \
    --attention-backend unfused \
    --disable-bias-linear \
    --use-rotary-position-embeddings \
    --rotary-percent 1.0 \
    --attention-dropout 0.0 \
    --hidden-dropout 0.0 \
    --tensor-model-parallel-size ${TP} \
    --pipeline-model-parallel-size 1 \
    --num-layers 40 \
    --hidden-size 5120 \
    --ffn-hidden-size 14336 \
    --num-attention-heads 32 \
    --seq-length 8192 \
    --kv-channels 128 \
    --normalization RMSNorm \
    --swiglu \
    --num-query-groups 8 \
    --group-query-attention \
    --position-embedding-type rope \
    --max-position-embeddings 8192 \
    --micro-batch-size 1 \
    --tokenizer-type HuggingFaceTokenizer \
    --tiktoken-pattern v2 \
    --tokenizer-model mistralai/Mistral-Nemo-Base-2407 \
    --save-interval 1000000 \
    --load ${CHECKPOINT_LOAD_DIR} \
    --fp16 \
    --rotary-base 1000000 \
    --use-dist-ckpt"

# Precompile CUDA extentions
python -c "import modelopt.torch.quantization.extensions as ext; print(ext.cuda_ext); print(ext.cuda_ext_fp8)"

# Acquire launch configuration where variable launch_config will be set
launch_config="--nproc_per_node=${TP}"

# Launch multi-process with torchrun
torchrun ${launch_config} examples/export/ptq_and_trtllm_export/text_generation_ptq.py ${options} ${additional_options}
