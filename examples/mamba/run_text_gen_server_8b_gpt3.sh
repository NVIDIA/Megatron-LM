#!/bin/bash

# Use: ./run_text_gen_server_8b_gpt3.sh <checkpoint-path> <tokenizer-path>
# To launch the client: python ../../tools/text_generation_cli.py <URL-provided-by-server>

CHECKPOINT_PATH=$1
TOKENIZER_PATH=$2

DISTRIBUTED_ARGS="--nproc_per_node 1 \
                  --nnodes 1 \
                  --node_rank 0 \
                  --master_addr localhost \
                  --master_port 6000"

export NCCL_IB_SL=1
export CUDA_DEVICE_MAX_CONNECTIONS=1
export NCCL_IB_TIMEOUT=19
export NCCL_IB_QPS_PER_CONNECTION=4

torchrun $DISTRIBUTED_ARGS ../../tools/run_text_generation_server.py \
       --tensor-model-parallel-size 1  \
       --pipeline-model-parallel-size 1  \
       --use-flash-attn \
       --apply-layernorm-1p \
       --untie-embeddings-and-output-weights \
       --num-layers 32  \
       --hidden-size 4096  \
       --load ${CHECKPOINT_PATH}  \
       --num-attention-heads 32  \
       --attention-dropout 0.0 \
       --hidden-dropout 0.0 \
       --disable-bias-linear \
       --seq-length 4096  \
       --max-position-embeddings 4096  \
       --position-embedding-type rope \
       --rotary-percent 0.5 \
       --squared-relu \
       --tokenizer-type GPTSentencePieceTokenizer  \
       --tokenizer-model ${TOKENIZER_PATH} \
       --distributed-backend nccl \
       --distributed-timeout-minutes 1440 \
       --bf16  \
       --micro-batch-size 1  \
       --use-mcore-models \
       --transformer-impl local \
       --seed 42
