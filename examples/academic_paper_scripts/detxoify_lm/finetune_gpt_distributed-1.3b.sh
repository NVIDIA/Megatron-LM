#! /bin/bash

# Change for multinode config
GPUS_PER_NODE=16
MASTER_ADDR=localhost
MASTER_PORT=$(($RANDOM + 1024))
NNODES=1
NODE_RANK=0
WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES))

# input
DATA_PATH=$1
SHARE_DATA=$PWD                       # current work dir
FINETUNED_PATH="$SHARE_DATA/$2"
lr=$3
bs=$4
iter=$5
CHECKPOINT_PATH=$6

# vocab
VOCAB_FILE=gpt2-vocab.json           # Your gpt-2 vocab
MERGE_FILE=gpt2-merges.txt           # Your gpt-2 merge file

# tensorboard
TENSORBOARD_DIR="$SHARE_DATA/tensorboard/$2"
mkdir -p ${TENSORBOARD_DIR}

DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE --nnodes $NNODES --node_rank $NODE_RANK --master_addr $MASTER_ADDR --master_port $MASTER_PORT"

python -m torch.distributed.run $DISTRIBUTED_ARGS \
     examples/detxoify_lm/finetune_gpt.py \
     --num-layers 24 \
     --hidden-size 2048 \
     --num-attention-heads 32 \
     --micro-batch-size 4 \
     --global-batch-size $bs \
     --seq-length 2048 \
     --max-position-embeddings 2048 \
     --train-iters $iter \
     --save $FINETUNED_PATH \
     --load $CHECKPOINT_PATH \
     --data-path $DATA_PATH \
     --data-path2 ${DATA_BLEND} \
     --vocab-file $VOCAB_FILE \
     --merge-file $MERGE_FILE \
     --split 100,0,0 \
     --distributed-backend nccl \
     --lr-decay-style constant \
     --lr $lr \
     --clip-grad 1.0 \
     --weight-decay 0.1 \
     --adam-beta1 0.9 \
     --adam-beta2 0.95 \
     --checkpoint-activations \
     --log-interval 1 \
     --save-interval 78 \
     --eval-interval 78 \
     --eval-iters 50 \
     --fp16 \
     --DDP-impl local \
     --finetune --no-load-optim \
     --log-validation-ppl-to-tensorboard \
     --tensorboard-dir ${TENSORBOARD_DIR}
