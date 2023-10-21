#!/bin/bash

TENSOR_MODEL_PARALLEL_SIZE=2

VOCAB_FILE=bert-vocab.txt
CHECKPOINT_PATH=checkpoints/bert_345m

WORLD_SIZE=$TENSOR_MODEL_PARALLEL_SIZE python tools/merge_mp_partitions.py \
                                --model-type BERT \
                                --tensor-model-parallel-size $TENSOR_MODEL_PARALLEL_SIZE \
                                --tokenizer-type BertWordPieceLowerCase \
                                --vocab-file $VOCAB_FILE \
                                --num-layers 24 \
                                --hidden-size 1024 \
                                --num-attention-heads 16 \
                                --seq-length 512 \
                                --max-position-embeddings 512 \
                                --load $CHECKPOINT_PATH
