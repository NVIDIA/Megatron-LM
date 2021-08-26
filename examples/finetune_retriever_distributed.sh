#!/bin/bash

# Finetune a BERT or pretrained ICT model using Google natural question data 
# Datasets can be downloaded from the following link:
# https://github.com/facebookresearch/DPR/blob/master/data/download_data.py

WORLD_SIZE=8

DISTRIBUTED_ARGS="--nproc_per_node $WORLD_SIZE \
                  --nnodes 1 \
                  --node_rank 0 \
                  --master_addr localhost \
                  --master_port 6000"

CHECKPOINT_PATH=<Specify path for the finetuned retriever model>

# Load either of the below
BERT_LOAD_PATH=<Path of BERT pretrained model>
PRETRAINED_CHECKPOINT=<Path of Pretrained ICT model>

python -m torch.distributed.launch $DISTRIBUTED_ARGS ./tasks/main.py \
        --task RET-FINETUNE-NQ \
        --train-with-neg \
        --train-hard-neg 1 \
        --pretrained-checkpoint ${PRETRAINED_CHECKPOINT} \
        --num-layers 12 \
        --hidden-size 768 \
        --num-attention-heads 12 \
        --tensor-model-parallel-size 1 \
        --tokenizer-type BertWordPieceLowerCase \
        --train-data nq-train.json \
        --valid-data nq-dev.json \
        --save ${CHECKPOINT_PATH} \
        --load ${CHECKPOINT_PATH} \
        --vocab-file bert-vocab.txt \
        --bert-load ${BERT_LOAD_PATH} \
        --save-interval 5000 \
        --log-interval 10 \
        --eval-interval 20000 \
        --eval-iters 100 \
        --indexer-log-interval 1000 \
        --faiss-use-gpu \
        --DDP-impl torch \
        --fp16 \
        --retriever-report-topk-accuracies 1 5 10 20 100 \
        --seq-length 512 \
        --retriever-seq-length 256 \
        --max-position-embeddings 512 \
        --retriever-score-scaling \
        --epochs 80 \
        --micro-batch-size 8 \
        --eval-micro-batch-size 16 \
        --indexer-batch-size 128 \
        --lr 2e-5 \
        --lr-warmup-fraction 0.01 \
        --weight-decay 1e-1
