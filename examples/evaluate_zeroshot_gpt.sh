#!/bin/bash

WORLD_SIZE=8

DISTRIBUTED_ARGS="--nproc_per_node $WORLD_SIZE \
                  --nnodes 1 \
                  --node_rank 0 \
                  --master_addr localhost \
                  --master_port 6000"

TASK="LAMBADA"

VALID_DATA=<lambada path>
VOCAB_FILE=gpt2-vocab.json
MERGE_FILE=gpt2-merges.txt
CHECKPOINT=checkpoints/gpt2_345m


python -m torch.distributed.launch $DISTRIBUTED_ARGS ./tasks/main.py \
               --task $TASK \
               --valid-data $VALID_DATA \
               --tokenizer-type GPT2BPETokenizer \
               --strict-lambada \
               --vocab-file $VOCAB_FILE \
               --merge-file $MERGE_FILE \
               --load $CHECKPOINT \
               --tensor-model-parallel-size 1 \
               --num-layers 24 \
               --hidden-size 1024 \
               --num-attention-heads 16 \
               --batch-size 8 \
               --activations-checkpoint-method uniform \
               --seq-length 1024 \
               --max-position-embeddings 1024 \
               --log-interval 10 \
               --fp16 \
               --no-load-optim \
               --no-load-rng
