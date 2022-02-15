#! /bin/bash

# Runs the "345M" parameter model

RANK=0
WORLD_SIZE=1

# DATA_PATH=<Specify path and file prefix>_text_document
# CHECKPOINT_PATH=<Specify path>
CHECKPOINT_PATH=checkpoints/gpt_base
DATA_PATH=/workspace/data/libai_dataset/loss_compara_content_sentence

python3 pretrain_gpt.py \
       --num-layers 6 \
       --hidden-size 384 \
       --num-attention-heads 16 \
       --micro-batch-size 4 \
       --global-batch-size 8 \
       --seq-length 256 \
       --max-position-embeddings 256 \
       --train-iters 500000 \
       --lr-decay-iters 320000 \
       --save $CHECKPOINT_PATH \
       --load $CHECKPOINT_PATH \
       --data-path $DATA_PATH \
       --num-workers 0 \
       --vocab-file /workspace/data/gpt_dataset/gpt2-vocab.json \
       --merge-file /workspace/data/gpt_dataset/gpt2-merges.txt \
       --data-impl mmap \
       --split 949,50,1 \
       --distributed-backend nccl \
       --lr 0.00015 \
       --min-lr 1.0e-5 \
       --lr-decay-style cosine \
       --weight-decay 1e-2 \
       --clip-grad 1.0 \
       --lr-warmup-fraction .01 \
       --activations-checkpoint-method uniform \
       --log-interval 100 \
       --save-interval 10000 \
       --eval-interval 1000 \
       --eval-iters 10 \
       --no-bias-dropout-fusion \
       # --fp16

       # --seq-length 1024 \
       # --max-position-embeddings 1024 \