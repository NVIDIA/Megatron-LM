#!/bin/bash

# Use: ./run_text_gen_server_8b.sh <checkpoint-path> <tokenizer-path>
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

export TRITON_CACHE_DIR="./triton-cache/"
export TRITON_CACHE_MANAGER="megatron.core.ssm.triton_cache_manager:ParallelFileCacheManager"

torchrun $DISTRIBUTED_ARGS ../../tools/run_mamba_text_generation_server.py \
       --tensor-model-parallel-size 1  \
       --pipeline-model-parallel-size 1  \
       --untie-embeddings-and-output-weights \
       --num-layers 56  \
       --hidden-size 4096  \
       --load ${CHECKPOINT_PATH}  \
       --num-attention-heads 32  \
       --group-query-attention \
       --num-query-groups 8 \
       --hybrid-attention-ratio 0.08 \
       --hybrid-mlp-ratio 0.5 \
       --attention-dropout 0.0 \
       --hidden-dropout 0.0 \
       --disable-bias-linear \
       --normalization RMSNorm \
       --seq-length 4096  \
       --max-position-embeddings 4096  \
       --position-embedding-type none \
       --tokenizer-type GPTSentencePieceTokenizer  \
       --tokenizer-model ${TOKENIZER_PATH} \
       --distributed-backend nccl \
       --distributed-timeout-minutes 1440 \
       --bf16  \
       --micro-batch-size 1  \
       --use-mcore-models \
       --spec megatron.core.models.mamba.mamba_layer_specs mamba_stack_spec \
       --seed 42
