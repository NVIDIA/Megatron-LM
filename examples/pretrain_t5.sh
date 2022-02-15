#!/bin/bash

RANK=0
WORLD_SIZE=1
DATA_PATH=/home/wang/workspace/Megatron-LM/examples
VOCAB_FILE=/home/wang/workspace/Megatron-LM/examples/vocab_file.txt
CHECKPOINT_PATH=/home/wang/workspace/Megatron-LM/examples

CHECKPOINT_PATH=checkpoints/t5_base
VOCAB_FILE=/home/wang/data/t5/dataset/bert-base-chinese-vocab.txt
DATA_PATH=/home/wang/data/t5/dataset/loss_compara_content_sentence

VOCAB_FILE=/workspace/data/libai_dataset/bert-base-chinese-vocab.txt
DATA_PATH=/workspace/data/libai_dataset/loss_compara_content_sentence

python3 pretrain_t5.py \
       --num-layers 6 \
       --hidden-size 384 \
       --num-attention-heads 12 \
       --kv-channels 32 \
       --ffn-hidden-size 1536 \
       --encoder-seq-length 512 \
       --decoder-seq-length 128 \
       --micro-batch-size 16 \
       --global-batch-size 16 \
       --max-position-embeddings 512 \
       --train-iters 1000 \
       --lr-decay-iters 1000000 \
       --save $CHECKPOINT_PATH \
       --load $CHECKPOINT_PATH \
       --data-path $DATA_PATH \
       --vocab-file $VOCAB_FILE \
       --data-impl mmap \
       --split 949,50,1 \
       --lr 0.0001 \
       --min-lr 0.00001 \
       --lr-decay-style constant \
       --lr-warmup-fraction .00 \
       --weight-decay 1e-2 \
       --clip-grad 1.0 \
       --log-interval 1 \
       --save-interval 10000 \
       --eval-interval 1000 \
       --eval-iters 10 \
       --vocab-extra-ids 100 \
       --num-workers 0 \
       # --fp16 \
