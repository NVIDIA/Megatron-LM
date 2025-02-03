#!/bin/bash
set -e

DEFAULT_NAME="/checkpoints/nemotron3-8b_v0.3.0"
NAME="${1:-$DEFAULT_NAME}"

DEFAULT_QUANT_CFG="fp8"
QUANT_CFG="${2:-$DEFAULT_QUANT_CFG}"

# CHANGE THE FOLLOWING IF YOU MOUNT YOUR DATA AND CHECKPOINTS DIFFERENTLY IN THE CONTAINER.
TP="8"
INFERENCE_TP=${TP}
DECODER_TYPE="gptnext"
CHECKPOINT_LOAD_DIR="${NAME}/nemo"

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
    --apply-layernorm-1p \
    --attn-attention unfused \
    --untie-embeddings-and-output-weights \
    --disable-bias-linear \
    --no-rope-fusion \
    --no-position-embedding \
    --use-rotary-position-embeddings \
    --rotary-percent 0.5 \
    --squared-relu \
    --attention-dropout 0.0 \
    --hidden-dropout 0.0 \
    --tensor-model-parallel-size ${TP} \
    --pipeline-model-parallel-size 1 \
    --num-layers 32 \
    --hidden-size 4096 \
    --ffn-hidden-size 16384 \
    --group-query-attention \
    --num-attention-heads 48 \
    --kv-channels 128 \
    --seq-length 4096 \
    --num-query-groups 8 \
    --max-position-embeddings 4096 \
    --micro-batch-size 4 \
    --tokenizer-type HuggingFaceTokenizer \
    --tokenizer-model nvidia/Minitron-8B-Base \
    --save-interval 1000000 \
    --load ${CHECKPOINT_LOAD_DIR} \
    --bf16 \
    --use-dist-ckpt"

# Precompile CUDA extentions
python -c "import modelopt.torch.quantization.extensions as ext; print(ext.cuda_ext); print(ext.cuda_ext_fp8)"

# Acquire launch configuration where variable launch_config will be set
launch_config="--nproc_per_node=${TP}"

# Launch multi-process with torchrun
torchrun ${launch_config} examples/export/ptq_and_trtllm_export/text_generation_ptq.py ${options} ${additional_options}
