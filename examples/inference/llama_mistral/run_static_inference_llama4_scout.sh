#!/bin/bash
export CUDA_DEVICE_MAX_CONNECTIONS=1
export NVTE_APPLY_QK_LAYER_SCALING=0

DISTRIBUTED_ARGS="--nproc_per_node 8 \
                  --nnodes 1 \
                  --node_rank 0 \
                  --master_addr 0.0.0.0 \
                  --master_port 6000"

# Fill in checkpoint path to Llama 4 Scout to run
CHECKPOINT=<Path to Scout checkpoint>
PROMPTS="What is the capital of France?"
TOKENS_TO_GENERATE=4
MAX_BATCH_SIZE=2

MODEL_ARGS=" \
    --micro-batch-size 1 \
    --bf16 \
    --no-masked-softmax-fusion \
    --disable-bias-linear \
    --untie-embeddings-and-output-weights \
    --position-embedding-type rope \
    --no-rope-fusion \
    --normalization RMSNorm \
    --swiglu \
    --num-layers 48 \
    --hidden-size 5120 \
    --ffn-hidden-size 16384 \
    --num-attention-heads 40 \
    --group-query-attention \
    --num-query-groups 8 \
    --qk-layernorm \
    --num-experts 16 \
    --moe-ffn-hidden-size 8192 \
    --moe-router-score-function sigmoid \
    --moe-router-topk 1 \
    --moe-router-topk-scaling-factor 1.0 \
    --moe-shared-expert-intermediate-size 8192 \
    --moe-aux-loss-coeff 1e-3 \
    --moe-token-dispatcher-type alltoall \
    --moe-token-drop-policy probs \
    --moe-router-load-balancing-type seq_aux_loss \
    --seq-length 4096 \
    --max-position-embeddings 4096 \
    --tokenizer-type HuggingFaceTokenizer \
    --make-vocab-size-divisible-by 128 \
    --use-mcore-models \
    --rotary-interleaved \
    --rotary-percent 1.0 \
    --rotary-base 500000 \
    --rope-scaling-factor 8.0 \
    --use-rope-scaling \
    --no-bias-swiglu-fusion \
    --qk-l2-norm \
    --moe-apply-probs-on-input \
    --moe-router-dtype fp64 \
"

torchrun $DISTRIBUTED_ARGS -m examples.inference.gpt.gpt_static_inference   \
      --load ${CHECKPOINT} \
      --tokenizer-model unsloth/Llama-4-Scout-17B-16E-Instruct \
      --dist-ckpt-strictness log_unexpected \
      --tensor-model-parallel-size 8 \
      --prompts ${PROMPTS} \
      --num-tokens-to-generate ${TOKENS_TO_GENERATE}  \
      --max-batch-size ${MAX_BATCH_SIZE} \
      ${MODEL_ARGS}
