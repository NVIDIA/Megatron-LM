#!/bin/bash
set -e

DEFAULT_NAME="/checkpoints/Mistral-NeMo-12B-Base"
NAME="${1:-$DEFAULT_NAME}"

DEFAULT_QUANT_CFG="fp8"
QUANT_CFG="${2:-$DEFAULT_QUANT_CFG}"

# NOTE: UNFUSED ATTENTION MUST BE USED TO AVOID ADDITIONAL STATE_DICT KEY MISMATCH.
export NVTE_FLASH_ATTN=0
export NVTE_FUSED_ATTN=0
export NVTE_UNFUSED_ATTN=1

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
    --no-masked-softmax-fusion \
    --no-position-embedding \
    --use-mcore-models \
    --disable-bias-linear \
    --rotary-percent 1.0 \
    --attention-dropout 0.0 \
    --hidden-dropout 0.0 \
    --tensor-model-parallel-size ${TP} \
    --pipeline-model-parallel-size 1 \
    --num-layers 32 \
    --hidden-size 4096 \
    --ffn-hidden-size 14336 \
    --num-attention-heads 32 \
    --seq-length 4096 \
    --kv-channels 128 \
    --normalization RMSNorm \
    --swiglu \
    --num-query-groups 8 \
    --num-experts 8 \
    --moe-router-topk 2 \
    --moe-aux-loss-coeff 1e-2 \
    --moe-router-load-balancing-type aux_loss \
    --group-query-attention \
    --position-embedding-type rope \
    --no-rope-fusion \
    --max-position-embeddings 32768 \
    --micro-batch-size 1 \
    --tokenizer-type HuggingFaceTokenizer \
    --tiktoken-pattern v2 \
    --tokenizer-model mistralai/Mixtral-8x7B-Instruct-v0.1 \
    --save-interval 1000000 \
    --load ${CHECKPOINT_LOAD_DIR} \
    --bf16 \
    --rotary-base 1000000 \
    --use-dist-ckpt"

# Precompile CUDA extentions
python -c "import modelopt.torch.quantization.extensions as ext; print(ext.cuda_ext); print(ext.cuda_ext_fp8)"

# Acquire launch configuration where variable launch_config will be set
launch_config="--nproc_per_node=${TP}"

# Launch multi-process with torchrun
torchrun ${launch_config} examples/export/ptq_and_trtllm_export/text_generation_ptq.py ${options} ${additional_options}


