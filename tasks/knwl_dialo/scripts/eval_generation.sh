#!/bin/bash

WORLD_SIZE=1

DISTRIBUTED_ARGS="--nproc_per_node $WORLD_SIZE \
                  --nnodes 1 \
                  --node_rank 0 \
                  --master_addr localhost \
                  --master_port 6000"

OUTPUT_PATH=<Speicifc path for the output generation>
GROUND_TRUTH_PATH=<Speicifc path for the ground truth>

python -m torch.distributed.launch $DISTRIBUTED_ARGS ./tasks/main.py \
        --num-layers 24 \
        --hidden-size 1024 \
        --num-attention-heads 16 \
        --seq-length 2048 \
        --max-position-embeddings 2048 \
        --micro-batch-size 4 \
        --task KNWL-DIALO-EVAL-F1 \
        --guess-file ${OUTPUT_PATH} \
        --answer-file ${GROUND_TRUTH_PATH}
