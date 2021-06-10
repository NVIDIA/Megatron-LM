#!/bin/bash

# Evaluate natural question test data given Wikipedia embeddings and pretrained
# ICT model or a finetuned model for Natural Question task

# Datasets can be downloaded from the following link:
# https://github.com/facebookresearch/DPR/blob/master/data/download_data.py

EVIDENCE_DATA_DIR=<Specify path of Wikipedia dataset>
EMBEDDING_PATH=<Specify path of the embeddings>
CHECKPOINT_PATH=<Specify path of pretrained ICT model or finetuned model>

QA_FILE=<Path of the natural question dev or test dataset>

python tasks/main.py \
    --task RETRIEVER-EVAL \
    --tokenizer-type BertWordPieceLowerCase \
    --num-layers 12 \
    --hidden-size 768 \
    --num-attention-heads 12 \
    --tensor-model-parallel-size 1 \
    --micro-batch-size 128 \
    --checkpoint-activations \
    --seq-length 512 \
    --max-position-embeddings 512 \
    --load ${CHECKPOINT_PATH} \
    --evidence-data-path ${EVIDENCE_DATA_DIR} \
    --embedding-path ${EMBEDDING_PATH} \
    --retriever-seq-length 256 \
    --vocab-file  bert-vocab.txt\
    --qa-data-test ${QA_FILE} \
    --faiss-use-gpu \
    --retriever-report-topk-accuracies 1 5 20 100 \
    --fp16 \
    --indexer-log-interval 1000 \
    --indexer-batch-size 128


