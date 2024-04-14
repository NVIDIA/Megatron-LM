#!/bin/bash
DEFAULT_NAME="/checkpoints/llama2-text-7b_v0.2.0"
NAME="${1:-$DEFAULT_NAME}"

DEFAULT_QUANT_CFG="int8_sq"
QUANT_CFG="${2:-$DEFAULT_QUANT_CFG}"

# CHANGE THE FOLLOWING IF YOU MOUNT YOUR DATA AND CHECKPOINTS DIFFERENTLY IN THE CONTAINER.
TP="8"
PP=1
INFERENCE_TP=${TP}
DECODER_TYPE="llama"
CHECKPOINT_LOAD_DIR="${NAME}"
TOKENIZER_MODEL="${CHECKPOINT_LOAD_DIR}/hf/tokenizer.model"

# LLaMA2 text 7b has ffn_hidden_size 11008. int4_awq requires a block_size of 128 as a result the TP can at most be 2
if [ "$QUANT_CFG" = "int4_awq" ]; then
    INFERENCE_TP="2"
fi

additional_options=" \
    --ammo-quant-cfg ${QUANT_CFG} \
    --ammo-load-classic-megatron-to-mcore \
    --decoder ${DECODER_TYPE} \
    --engine-dir /tmp/ammo \
    --max-input-len 2048 \
    --max-output-len 512 \
    --max-batch-size 8 \
    --inference-tensor-parallel ${INFERENCE_TP} "

trtllm_options=" \
    --engine-dir /tmp/ammo \
    --tokenizer ${CHECKPOINT_LOAD_DIR}/hf \
    --max-output-len 512 "

# DO NOT CHANGE THE SETTING BELOW UNLESS YOU KNOW WHAT YOU ARE DOING!!!
export CUDA_DEVICE_MAX_CONNECTIONS=1

options=" \
    --disable-bias-linear \
    --swiglu \
    --untie-embeddings-and-output-weights \
    --use-rotary-position-embeddings \
    --normalization RMSNorm \
    --norm-epsilon 1e-5 \
    --no-position-embedding \
    --no-masked-softmax-fusion \
    --no-bias-gelu-fusion \
    --no-bias-dropout-fusion \
    --no-async-tensor-model-parallel-allreduce \
    --tensor-model-parallel-size ${TP} \
    --pipeline-model-parallel-size 1 \
    --num-layers 32 \
    --hidden-size 4096 \
    --ffn-hidden-size 11008 \
    --num-attention-heads 32 \
    --seq-length 2048 \
    --max-position-embeddings 4096 \
    --micro-batch-size 1 \
    --make-vocab-size-divisible-by 1 \
    --tokenizer-type Llama2Tokenizer \
    --tokenizer-model ${TOKENIZER_MODEL} \
    --save-interval 1000000 \
    --bf16 \
    --use-mcore-models "

set +x

# Precompile CUDA extentions
python -c "import ammo.torch.quantization.extensions as ext; print(ext.cuda_ext); print(ext.cuda_ext_fp8)"

# Acquire launch configuration where variable launch_config will be set
launch_config="--nproc_per_node=${TP}"

# Launch multi-process with torchrun
torchrun ${launch_config} examples/inference/text_generation_ptq.py ${options} ${additional_options} --load ${CHECKPOINT_LOAD_DIR}

# This script is using mpi4py which will fork multiple processes.
python examples/inference/trtllm_text_generation.py ${trtllm_options}
