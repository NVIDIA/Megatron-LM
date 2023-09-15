#! /bin/bash

DATA_PATH=$1
CHECKPOINT_PATH=$2
TENSORBOARD_DIR=$3
TP_SIZE=$4
PP_SIZE=$5
NNODES=$6

GPUS_PER_NODE=8
# Change for multinode config
MASTER_ADDR=localhost
MASTER_PORT=6000
NODE_RANK=0
WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES))
export CUDA_DEVICE_MAX_CONNECTIONS=1


# Runs the "345M" parameter model
DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE --nnodes $NNODES"

# Run for 100 iterations
torchrun $DISTRIBUTED_ARGS \
       pretrain_bert.py \
       --use-checkpoint-args \
       --use-checkpoint-opt_param-scheduler \
       --num-layers 24 \
       --hidden-size 1024 \
       --num-attention-heads 16 \
       --log-params-norm \
       --log-num-zeros-in-grad \
       --log-validation-ppl-to-tensorboard \
       --log-timers-to-tensorboard \
       --tensorboard-dir ${TENSORBOARD_DIR} \
       --micro-batch-size 4 \
       --global-batch-size 128 \
       --seq-length 512 \
       --max-position-embeddings 512 \
       --train-iters 100 \
       --timing-log-level 2 \
       --lr-decay-iters 990000 \
       --save $CHECKPOINT_PATH \
       --load $CHECKPOINT_PATH \
       --data-path $DATA_PATH \
       --vocab-file /workspace/data/bert_data/vocab.txt \
       --split 949,50,1 \
       --distributed-backend nccl \
       --lr 0.0001 \
       --min-lr 0.00001 \
       --lr-warmup-fraction 0.01 \
       --log-interval 1 \
       --save-interval 50 \
       --eval-interval 1000 \
       --eval-iters 10 \
       --tensor-model-parallel-size $TP_SIZE \
       --pipeline-model-parallel-size $PP_SIZE \
       --no-gradient-accumulation-fusion \
       --fp16

echo 50 > $CHECKPOINT_PATH/latest_checkpointed_iteration.txt

# Resume from 50th iteration ckpt and continue to 100 iterations
torchrun $DISTRIBUTED_ARGS \
       pretrain_bert.py \
       --use-checkpoint-args \
       --use-checkpoint-opt_param-scheduler \
       --num-layers 24 \
       --hidden-size 1024 \
       --num-attention-heads 16 \
       --log-params-norm \
       --log-num-zeros-in-grad \
       --log-validation-ppl-to-tensorboard \
       --log-timers-to-tensorboard \
       --tensorboard-dir ${TENSORBOARD_DIR} \
       --micro-batch-size 4 \
       --global-batch-size 128 \
       --seq-length 512 \
       --max-position-embeddings 512 \
       --train-iters 100 \
       --timing-log-level 2 \
       --lr-decay-iters 990000 \
       --save $CHECKPOINT_PATH \
       --load $CHECKPOINT_PATH \
       --data-path $DATA_PATH \
       --vocab-file /workspace/data/bert_data/vocab.txt \
       --split 949,50,1 \
       --distributed-backend nccl \
       --lr 0.0001 \
       --min-lr 0.00001 \
       --lr-warmup-fraction 0.01 \
       --log-interval 1 \
       --save-interval 10000 \
       --eval-interval 1000 \
       --eval-iters 10 \
       --tensor-model-parallel-size $TP_SIZE \
       --pipeline-model-parallel-size $PP_SIZE \
       --no-gradient-accumulation-fusion \
       --fp16