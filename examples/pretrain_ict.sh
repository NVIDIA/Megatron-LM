#! /bin/bash

# Runs the "217M" parameter biencoder model for ICT retriever

RANK=0
WORLD_SIZE=1

PRETRAINED_BERT_PATH=<Specify path of pretrained BERT model>
TEXT_DATA_PATH=<Specify path and file prefix of the text data>
TITLE_DATA_PATH=<Specify path and file prefix od the titles>
CHECKPOINT_PATH=<Specify path>


python pretrain_ict.py \
        --num-layers 12 \
        --hidden-size 768 \
        --num-attention-heads 12 \
        --tensor-model-parallel-size 1 \
        --micro-batch-size 32 \
        --seq-length 256 \
        --max-position-embeddings 512 \
        --train-iters 100000 \
        --vocab-file bert-vocab.txt \
        --tokenizer-type BertWordPieceLowerCase \
        --DDP-impl torch \
        --bert-load ${PRETRAINED_BERT_PATH} \
        --log-interval 100 \
        --eval-interval 1000 \
        --eval-iters 10 \
        --retriever-report-topk-accuracies 1 5 10 20 100 \
        --retriever-score-scaling \
        --load $CHECKPOINT_PATH \
        --save $CHECKPOINT_PATH \
        --data-path ${TEXT_DATA_PATH} \
        --titles-data-path ${TITLE_DATA_PATH} \
        --lr 0.0001 \
        --lr-decay-style linear \
        --weight-decay 1e-2 \
        --clip-grad 1.0 \
        --lr-warmup-fraction 0.01 \
        --save-interval 4000 \
        --exit-interval 8000 \
        --query-in-block-prob 0.1 \
        --fp16
