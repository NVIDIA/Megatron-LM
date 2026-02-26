#!/bin/bash

# --- User Configuration ---
# Path to your Megatron-LM repository
MEGATRON_PATH="/lustre/fsw/portfolios/coreai/users/shanmugamr/Megatron-LM/"
# Path for saving the checkpoint and logs
OUTPUT_PATH="/lustre/fsw/portfolios/coreai/users/shanmugamr/Megatron-LM/deepseek_mtp_dummy_ckpt"

# Path to a dummy data file (can be a simple text file)
# Example: echo "hello world" > dummy_data.txt

# --- Script ---
mkdir -p ${OUTPUT_PATH}/checkpoints
mkdir -p ${OUTPUT_PATH}/tensorboard

# These arguments define a very small DeepSeek-like model with MTP heads.
# Model size is reduced for quick checkpoint creation.
PRETRAIN_ARGS=(
    # Parallelism
    --tensor-model-parallel-size 1
    --pipeline-model-parallel-size 1
    --use-mcore-models

    # Model Architecture (Small)
    --num-layers 16
    --hidden-size 256
    --ffn-hidden-size 1024
    --num-attention-heads 8
    --seq-length 1024
    --max-position-embeddings 1024
    --position-embedding-type rope
    --normalization RMSNorm
    --swiglu
    --untie-embeddings-and-output-weights

    # MTP Head Configuration
    # These arguments are taken from the deepseek example script
    --mtp-num-layers 3
    --mtp-loss-scaling-factor 0.1

    # Training Configuration (Minimal)
    --micro-batch-size 1
    --global-batch-size 1
    --train-iters 1 # Run for only 1 iteration to create the checkpoint
    --lr 1e-4
    --lr-decay-style cosine

    # Data and Tokenizer
    --tokenizer-type HuggingFaceTokenizer
    --tokenizer-model deepseek-ai/deepseek-coder-6.7b-base # or another HF tokenizer
    --mock-data
    --split 100,0,0
    --no-create-attention-mask-in-dataloader

    # Checkpointing
    --save ${OUTPUT_PATH}/checkpoints
    --save-interval 1 # Save after the first iteration
    --eval-interval 1

    # Other settings
    --use-flash-attn
    --disable-bias-linear
    --bf16
    --log-interval 1
    --tensorboard-dir ${OUTPUT_PATH}/tensorboard
)

# --- Execution ---
cd ${MEGATRON_PATH}
export PYTHONPATH=${MEGATRON_PATH}:${PYTHONPATH}

python ${MEGATRON_PATH}/pretrain_gpt.py ${PRETRAIN_ARGS[@]}

echo "---"
echo "Dummy checkpoint created in: ${OUTPUT_PATH}/checkpoints"
echo "---"
