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
DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE --nnodes $NNODES --node_rank $NODE_RANK --master_addr $MASTER_ADDR --master_port $MASTER_PORT"

# Run for 100 iterations and save checkpoint at 50
python -m torch.distributed.launch $DISTRIBUTED_ARGS \
       pretrain_gpt.py \
       --use-checkpoint-args \
       --use-checkpoint-opt_param-scheduler \
       --num-layers 12 \
       --hidden-size 512 \
       --num-attention-heads 8 \
       --log-params-norm \
       --log-num-zeros-in-grad \
       --log-validation-ppl-to-tensorboard \
       --log-timers-to-tensorboard \
       --tensorboard-dir ${TENSORBOARD_DIR} \
       --micro-batch-size 4 \
       --global-batch-size 32 \
       --seq-length 1024 \
       --max-position-embeddings 1024 \
       --train-iters 100 \
       --timing-log-level 2 \
       --lr-decay-iters 320000 \
       --save $CHECKPOINT_PATH \
       --load $CHECKPOINT_PATH \
       --data-path $DATA_PATH \
       --vocab-file /workspace/data/gpt3_data/gpt2-vocab.json \
       --merge-file /workspace/data/gpt3_data/gpt2-merges.txt \
       --data-impl mmap \
       --split 949,50,1 \
       --distributed-backend nccl \
       --lr 0.00015 \
       --lr-decay-style cosine \
       --min-lr 1.0e-5 \
       --weight-decay 1e-2 \
       --clip-grad 1.0 \
       --lr-warmup-fraction .01 \
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
python -m torch.distributed.launch $DISTRIBUTED_ARGS \
       pretrain_gpt.py \
       --use-checkpoint-args \
       --use-checkpoint-opt_param-scheduler \
       --num-layers 12 \
       --hidden-size 512 \
       --num-attention-heads 8 \
       --log-params-norm \
       --log-num-zeros-in-grad \
       --log-validation-ppl-to-tensorboard \
       --log-timers-to-tensorboard \
       --tensorboard-dir ${TENSORBOARD_DIR} \
       --micro-batch-size 4 \
       --global-batch-size 32 \
       --seq-length 1024 \
       --max-position-embeddings 1024 \
       --train-iters 100 \
       --timing-log-level 2 \
       --lr-decay-iters 320000 \
       --save $CHECKPOINT_PATH \
       --load $CHECKPOINT_PATH \
       --data-path $DATA_PATH \
       --vocab-file /workspace/data/gpt3_data/gpt2-vocab.json \
       --merge-file /workspace/data/gpt3_data/gpt2-merges.txt \
       --data-impl mmap \
       --split 949,50,1 \
       --distributed-backend nccl \
       --lr 0.00015 \
       --lr-decay-style cosine \
       --min-lr 1.0e-5 \
       --weight-decay 1e-2 \
       --clip-grad 1.0 \
       --lr-warmup-fraction .01 \
       --log-interval 1 \
       --save-interval 10000 \
       --eval-interval 1000 \
       --eval-iters 10 \
       --tensor-model-parallel-size $TP_SIZE \
       --pipeline-model-parallel-size $PP_SIZE \
       --no-gradient-accumulation-fusion \
       --fp16