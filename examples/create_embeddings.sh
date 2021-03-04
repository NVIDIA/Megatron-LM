#!/bin/bash

# Compute embeddings for each entry of a given dataset (e.g. Wikipedia)

RANK=0
WORLD_SIZE=1

# Wikipedia data can be downloaded from the following link:
# https://github.com/facebookresearch/DPR/blob/master/data/download_data.py
EVIDENCE_DATA_DIR=<Specify path of Wikipedia dataset>
EMBEDDING_PATH=<Specify path to store embeddings>
CHECKPOINT_PATH=<Specify path of pretrained ICT model>

python tools/create_doc_index.py \
    --num-layers 12 \
    --hidden-size 768 \
    --num-attention-heads 12 \
    --tensor-model-parallel-size 1 \
    --micro-batch-size 128 \
    --checkpoint-activations \
    --seq-length 512 \
    --retriever-seq-length 256 \
    --max-position-embeddings 512 \
    --load ${CHECKPOINT_PATH} \
    --evidence-data-path ${EVIDENCE_DATA_DIR} \
    --embedding-path ${EMBEDDING_PATH} \
    --indexer-log-interval 1000 \
    --indexer-batch-size 128 \
    --vocab-file bert-vocab.txt \
    --num-workers 2 \
    --fp16

