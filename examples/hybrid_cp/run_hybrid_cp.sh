#!/bin/bash

export NCCL_IB_SL=1
export TOKENIZERS_PARALLELISM="false"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

MCORE_PATH="../"
OUTPUT_BASE="./output"
SEQ_LEN=16384

HYBRID_CP_ARGS=" \
    --hybrid-context-parallel \
    --sequence-packing \
    --calculate-per-token-loss \
    --max-seqlen-per-dp-cp-rank 4096 \
"

ARGS=" \
    --sft \
    --legacy-tokenizer \
    --tokenizer-type NullTokenizer \
    --vocab-size 131072 \
    --mock-data \
    --sft-mock-dataset-config-json {\"mode\":\"distribution\",\"type\":\"lognormal\",\"min_seq_len\":1024,\"max_seq_len\":16384,\"mean_seq_len\":8192,\"lognormal_sigma\":1.1} \
    --use-distributed-optimizer \
    --disable-bias-linear \
    --transformer-impl transformer_engine \
    --normalization RMSNorm \
    --norm-epsilon 1e-06 \
    --attention-dropout 0.0 \
    --hidden-dropout 0.0 \
    --untie-embeddings-and-output-weights \
    --position-embedding-type rope \
    --rotary-percent 1.0 \
    --rotary-base 1000000 \
    --swiglu \
    --tensor-model-parallel-size 1  \
    --pipeline-model-parallel-size 1 \
    --rerun-mode disabled \
    --num-layers 4 \
    --hidden-size 2048 \
    --ffn-hidden-size 8192 \
    --add-qkv-bias \
    --num-attention-heads 16 \
    --num-workers 8 \
    --exit-duration-in-mins 230 \
    --seq-length ${SEQ_LEN} \
    --max-position-embeddings ${SEQ_LEN} \
    --train-samples 100000 \
    --lr-warmup-samples 20000 \
    --micro-batch-size 4 \
    --global-batch-size 256 \
    --lr 2e-5 \
    --min-lr 0.0 \
    --lr-decay-style cosine \
    --log-interval 1 \
    --eval-iters 10 \
    --eval-interval 999999 \
    --save-interval 1000 \
    --use-mcore-models \
    --no-create-attention-mask-in-dataloader \
    --no-mmap-bin-files \
    --clip-grad 1.0 \
    --weight-decay 0.05 \
    --adam-beta1 0.9 \
    --adam-beta2 0.999 \
    --init-method-std 0.014 \
    --bf16 \
    --distributed-timeout-minutes 60 \
    --attention-backend flash \
    --disable-gloo-process-groups \
    --use-dist-ckpt \
"

torchrun --nproc_per_node 8 ${MCORE_PATH}/pretrain_gpt.py ${ARGS} ${HYBRID_CP_ARGS}
