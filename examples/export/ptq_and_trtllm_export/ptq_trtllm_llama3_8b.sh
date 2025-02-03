#!/bin/bash
set -e

DEFAULT_NAME="/checkpoints/llama-3_1-8b-nemo_v1.0"
NAME="${1:-$DEFAULT_NAME}"

DEFAULT_QUANT_CFG="int8_sq"
QUANT_CFG="${2:-$DEFAULT_QUANT_CFG}"


# CHANGE THE FOLLOWING IF YOU MOUNT YOUR DATA AND CHECKPOINTS DIFFERENTLY IN THE CONTAINER.
TP="1"
INFERENCE_TP=${TP}
DECODER_TYPE="llama"
CHECKPOINT_LOAD_DIR="${NAME}"

# LLaMA2 text 7b has ffn_hidden_size 11008. int4_awq requires a block_size of 128 as a result the TP can at most be 2
if [ "$QUANT_CFG" = "int4_awq" ]; then
    INFERENCE_TP="2"
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
    --disable-bias-linear \
    --attention-backend unfused \
    --swiglu \
    --no-rope-fusion \
    --untie-embeddings-and-output-weights \
    --use-rotary-position-embeddings \
    --normalization RMSNorm \
    --rotary-percent 1.0 \
    --hidden-dropout 0.0 \
    --attention-dropout 0.0 \
    --no-bias-gelu-fusion \
    --no-bias-dropout-fusion \
    --no-async-tensor-model-parallel-allreduce \
    --tensor-model-parallel-size ${TP} \
    --pipeline-model-parallel-size 1 \
    --num-layers 32 \
    --hidden-size 4096 \
    --group-query-attention \
    --num-query-groups 8 \
    --ffn-hidden-size 14336 \
    --num-attention-heads 32 \
    --seq-length 8192 \
    --max-position-embeddings 8192 \
    --micro-batch-size 4 \
    --make-vocab-size-divisible-by 128 \
    --tokenizer-type HuggingFaceTokenizer \
    --tokenizer-model meta-llama/Meta-Llama-3-8B \
    --save-interval 1000000 \
    --use-dist-ckpt \
    --load ${CHECKPOINT_LOAD_DIR} \
    --rotary-base 500000 \
    --fp16"

# Precompile CUDA extentions
python -c "import modelopt.torch.quantization.extensions as ext; print(ext.cuda_ext); print(ext.cuda_ext_fp8)"

# Acquire launch configuration where variable launch_config will be set
launch_config="--nproc_per_node=${TP}"

# Launch multi-process with torchrun
torchrun ${launch_config} examples/export/ptq_and_trtllm_export/text_generation_ptq.py ${options} ${additional_options}
