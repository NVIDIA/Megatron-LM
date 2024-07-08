#!/bin/bash
# This example will start serving the 345M model that is partitioned 8 way tensor parallel
DISTRIBUTED_ARGS="--nproc_per_node 1 \
                  --nnodes 1 \
                  --node_rank 0 \
                  --master_addr localhost \
                  --master_port 6000"

CHECKPOINT=/gpfs/scratch/ehpc12/pre-training/ducttape_output/train_cosine_megatronlm/euro_llm-models/1b/iter_0020000
TOKENIZER_PATH=/gpfs/scratch/ehpc12/pre-training/ducttape_output/train_cosine_megatronlm/euro_llm-models/1b/tokenizer.model

python $DISTRIBUTED_ARGS tools/run_text_generation_server.py   \
       --tensor-model-parallel-size 1  \
       --pipeline-model-parallel-size 1  \
       --num-layers 12  \
       --hidden-size 4128  \
       --load ${CHECKPOINT}  \
       --num-attention-heads 12  \
       --max-position-embeddings 4096  \
       --tokenizer-type SentencePieceTokenizer  \
       --tokenizer-model ${TOKENIZER_PATH} \
       --fp16  \
       --micro-batch-size 1  \
       --seq-length 4096  \
       --seed 42
