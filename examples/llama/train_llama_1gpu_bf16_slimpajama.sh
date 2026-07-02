#!/bin/bash

# Runs a small Llama-3.x style model for training from scratch

export CUDA_DEVICE_MAX_CONNECTIONS=1

# Configure Wandb to use scratch space to avoid home directory space issues
export WANDB_CACHE_DIR=/home/scratch.sagdesai_wwfo/.cache/wandb
export WANDB_CONFIG_DIR=/home/scratch.sagdesai_wwfo/.config/wandb
export WANDB_DATA_DIR=/home/scratch.sagdesai_wwfo/.local/share/wandb

GPUS_PER_NODE=1
# Change for multinode config
MASTER_ADDR=localhost
MASTER_PORT=6000
NUM_NODES=1
NODE_RANK=0
WORLD_SIZE=$(($GPUS_PER_NODE*$NUM_NODES))

CHECKPOINT_PATH=/home/scratch.sagdesai_wwfo/megatron-lm-fp8/checkpoints
TENSORBOARD_LOGS_PATH=/home/scratch.sagdesai_wwfo/megatron-lm-fp8/tensorboard_logs
TOKENIZER_MODEL=/home/scratch.sagdesai_wwfo/datasets/slimpajama/tokenizer
DATA_PATH=/home/scratch.sagdesai_wwfo/datasets/slimpajama/slimpajama_processed/example_train_0_text_document

DISTRIBUTED_ARGS=(
    --nproc_per_node $GPUS_PER_NODE 
    --nnodes $NUM_NODES 
    --master_addr $MASTER_ADDR 
    --master_port $MASTER_PORT
)

# Llama-style model arguments (small scale for 1 GPU training)
LLAMA_MODEL_ARGS=(
    --num-layers 12
    --hidden-size 2048
    --num-attention-heads 16
    --seq-length 2048
    --max-position-embeddings 2048
    --attention-backend auto
    --normalization RMSNorm
    --position-embedding-type rope
    --untie-embeddings-and-output-weights
    --disable-bias-linear
    --transformer-impl transformer_engine
    --group-query-attention
    --num-query-groups 4
    --attention-dropout 0.0
    --hidden-dropout 0.0
    --rotary-base 500000
    --rotary-percent 1.0
    --swiglu
    --ffn-hidden-size 5504
    --no-masked-softmax-fusion
    --attention-softmax-in-fp32
)

TRAINING_ARGS=(
    --micro-batch-size 16
    --global-batch-size 64 
    --train-iters 500000 
    --weight-decay 0.1 
    --adam-beta1 0.9 
    --adam-beta2 0.95 
    --init-method-std 0.006 
    --clip-grad 1.0 
    --bf16
    --lr 6.0e-5 
    --lr-decay-style cosine 
    --min-lr 6.0e-6
    --lr-warmup-fraction .001 
    --lr-decay-iters 430000
)

MODEL_PARALLEL_ARGS=(
	--tensor-model-parallel-size 1 
	--pipeline-model-parallel-size 1 
)

DATA_ARGS=(
    --data-path $DATA_PATH
    --tokenizer-type HuggingFaceTokenizer
    --tokenizer-model $TOKENIZER_MODEL
    --split 949,50,1
)

EVAL_AND_LOGGING_ARGS=(
    --log-interval 100
    --tensorboard-log-interval 1
    --save-interval 10000 
    --eval-interval 1000 
    --save $CHECKPOINT_PATH 
    --load $CHECKPOINT_PATH 
    --eval-iters 10
    --tensorboard-dir $TENSORBOARD_LOGS_PATH 
    --wandb-project "megatron-lm-training"
    --wandb-exp-name "llama3x-small-slimpajama-bf16"
    --wandb-save-dir $CHECKPOINT_PATH/wandb
    --log-params-norm
    --log-num-zeros-in-grad
    --log-timers-to-tensorboard
)

/bin/python -m torch.distributed.run ${DISTRIBUTED_ARGS[@]} pretrain_gpt.py \
    ${LLAMA_MODEL_ARGS[@]} \
    ${TRAINING_ARGS[@]} \
    ${MODEL_PARALLEL_ARGS[@]} \
    ${DATA_ARGS[@]} \
    ${EVAL_AND_LOGGING_ARGS[@]}
