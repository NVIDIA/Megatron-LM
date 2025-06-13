#!/bin/bash

export CUDA_DEVICE_MAX_CONNECTIONS=1
# export GLOO_SOCKET_IFNAME=bond1
# export NCCL_SOCKET_IFNAME=bond1

readonly GPUS_PER_NODE=8
readonly NODE_RANK=0
readonly NNODES=1
readonly MASTER_PORT=65535
export MASTER_ADDR=localhost

CHECKPOINT_PATH=${PWD}/ckpt
VOCAB_FILE=../oscar-data/gpt2-vocab.json
MERGE_FILE=../oscar-data/gpt2-merges.txt
DATA_PATH=../oscar-data/my-gpt2_text_document

DISTRIBUTED_ARGS=(
    --nproc_per_node $GPUS_PER_NODE 
    --nnodes $NNODES 
    --node_rank $NODE_RANK
    --master_addr $MASTER_ADDR 
    --master_port $MASTER_PORT
)

readonly TRAINING_YAML=$1 # e.g. "examples/cli-arg-yaml-cfgs/trainer.yaml"
readonly MODEL_YAML=$2    # e.g. "examples/cli-arg-yaml-cfgs/gpt.yaml"

MODEL_PARALLEL_ARGS=(
	--tensor-model-parallel-size 2
	--pipeline-model-parallel-size 2
)

DATA_ARGS=(
    --data-path $DATA_PATH 
    --vocab-file $VOCAB_FILE 
    --merge-file $MERGE_FILE 
    --split 949,50,1
)

EVAL_AND_LOGGING_ARGS=(
    --save-interval 10000 
    --eval-interval 1000 
    --save $CHECKPOINT_PATH 
    --load $CHECKPOINT_PATH 
)

torchrun ${DISTRIBUTED_ARGS[@]} pretrain_gpt.py \
    --cli-arg-yaml-cfgs $TRAINING_YAML $MODEL_YAML \
    ${MODEL_PARALLEL_ARGS[@]} \
    ${DATA_ARGS[@]} \
    ${EVAL_AND_LOGGING_ARGS[@]}
