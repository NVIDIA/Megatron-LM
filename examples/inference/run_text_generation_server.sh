#!/bin/bash
# This example will start serving the 345M model that is partitioned 8 way tensor parallel
DISTRIBUTED_ARGS="--nproc_per_node 1 \
                  --nnodes 1 \
                  --node_rank 0 \
                  --master_addr localhost \
                  --master_port 6000"

CHECKPOINT=/gpfs/scratch/ehpc12/pre-training/ducttape_output/train_cosine_megatronlm/euro_llm-models/1b/
TOKENIZER_PATH=/gpfs/projects/ehpc12/pre-training/tokenizer/tokenizer.model

CUDA_DEVICE_MAX_CONNECTIONS=1 torchrun $DISTRIBUTED_ARGS tools/run_text_generation_server.py   \
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
       --seed 42 \
       --transformer-impl local \
       --use-legacy-models \
       --swiglu \
       --normalization RMSNorm \
       --disable-bias-linear \
       --use-rotary-position-embeddings \
       --group-query-attention \
       --num-query-groups 8 \
       --num-layers 24 \
       --hidden-size 2048 \
       --ffn-hidden-size 5632 \
       --num-attention-heads 16 \
       --attention-dropout 0 \
       --hidden-dropout 0 \
       --untie-embeddings-and-output-weights \
       --use-flash-attn \
