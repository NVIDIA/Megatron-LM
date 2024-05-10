#!/bin/bash

NNODES=$SLURM_NNODES # number of nodes used for training
GPUS_PER_NODE=$SLURM_GPUS_PER_NODE # number of gpus per node
MASTER_PORT=6002
WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES))
NODE_RANK=$SLURM_NODEID

VOCAB_FILE=  # Path to vocab.json
MERGE_FILE=  # Path to merges.txt
DATA_PATH=  # Path to Pile Dedupulicated dataset
TENSORBOARD_DIR=  # path to store tensorboard log
WANDB_DIR=  # path to store wandb log

# Parallel Setting
TENSOR_PARALLEL_SIZE=8
PIPELINE_PARALLEL_SIZE=1
MICRO_BATCH_SIZE=1
# Calculate GLOBAL_BATCH_SIZE based on Accumulation Step=1
GLOBAL_BATCH_SIZE=$(($WORLD_SIZE / ($TENSOR_PARALLEL_SIZE * $PIPELINE_PARALLEL_SIZE) * $MICRO_BATCH_SIZE))

# Distributed training args
DISTRIBUTED_ARGS="
    --nproc_per_node $GPUS_PER_NODE \
    --nnodes $NNODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT \
"

# 13B Model
MODEL_ARGS="
    --num-layers 40 \
    --hidden-size 5120 \
    --num-attention-heads 40 \
    --seq-length 2048 \
    --max-position-embeddings 2048 \
"

# learning rate, training type and other optimizer arguments
OPTIMIZER_ARGS="
    --lr 0.0001 \
    --lr-decay-iters 70000 \
    --lr-decay-style cosine \
    --min-lr 0.00001 \
    --adam-beta1 0.9 \
    --adam-beta2 0.95 \
    --adam-eps 1e-08 \
    --weight-decay .1 \
    --lr-warmup-fraction 0.01 \
    --clip-grad 1.0 \
    --loss-scale 0 \
    --loss-scale-window 1000 \
    --hysteresis 2 \
    --min-loss-scale 1 \
    --bf16 \
    --use-distributed-optimizer \
"

# Parallel size, global batch and others, (micro batch size depeneds on gpu memory)
TRAINING_ARGS="
    --tensor-model-parallel-size $TENSOR_PARALLEL_SIZE \
    --pipeline-model-parallel-size $PIPELINE_PARALLEL_SIZE \
    --micro-batch-size $MICRO_BATCH_SIZE \
    --global-batch-size $GLOBAL_BATCH_SIZE \
    --train-iters 80000 \
"

# Data path, --mock-data is using fake data (to avoid IO overhead)
DATA_ARGS="
    --data-path $DATA_PATH \
    --vocab-file $VOCAB_FILE \
    --merge-file $MERGE_FILE \
    --mock-data \
"

OUTPUT_ARGS="
    --log-interval 20 \
    --timing-log-level 2 \
    --save-interval 5002 \
    --eval-interval 100 \
    --eval-iters 10 \
    --log-timers-to-tensorboard \
    --log-validation-ppl-to-tensorboard \
    --log-throughput \
    --tensorboard-dir ${TENSORBOARD_DIR} \
    --tensorboard-log-interval 1 \
    --wandb-project NeurIPS-Speed \
    --wandb-save-dir ${WANDB_DIR} \
    --wandb-exp-name 13B-Baseline-E2E \
"

QUANTIZE_ARGS="
    --no-async-tensor-model-parallel-allreduce \
    --recompute-activations \
    --recompute-granularity selective \
"

torchrun $DISTRIBUTED_ARGS pretrain_gpt.py \
    $MODEL_ARGS \
    $TRAINING_ARGS \
    $OPTIMIZER_ARGS \
    $DATA_ARGS \
    $OUTPUT_ARGS \
    $QUANTIZE_ARGS \
    --distributed-backend nccl \
    --save $CHECKPOINT_PATH \
    --load $CHECKPOINT_PATH