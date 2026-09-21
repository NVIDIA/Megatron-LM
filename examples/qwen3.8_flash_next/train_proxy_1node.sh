#!/usr/bin/env bash
# Qwen3.8-Flash-Next proxy, one node (4 GPUs), EP4, BF16, from scratch.
#
# This proxy keeps every per-layer, per-expert and PLE dimension of the 48-layer model and
# shrinks only what sets the total size, so per-rank shapes on 4 GPUs match the 64-GPU model:
# 8 layers instead of 48, 32 experts at EP4 instead of 512 at EP64 (8 local either way), and a
# 1 M n-gram base instead of 20 M. It is a functional and recipe-development proxy, not a
# memory proxy for EP64. See docs/models/qwen3.8_flash_next/training/proxy_single_node.md.
#
# Runs on Megatron's mock data with NullTokenizer at the real vocabulary size, so nothing has to
# be downloaded. Status: this configuration has been run.
#
#   ./train_proxy_1node.sh [extra megatron args...]
#
# Later flags win in Megatron's argparse, so pass overrides as arguments rather than editing the
# block below.
set -uo pipefail

MLM="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$MLM" || exit 2

export CUDA_DEVICE_MAX_CONNECTIONS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python -m torch.distributed.run --nproc_per_node 4 --nnodes 1 pretrain_hybrid.py \
    --hybrid-layer-pattern GEGEGEQEGEGEGEQE/QE \
    --spec megatron.core.models.hybrid.hybrid_layer_specs gated_residual_hybrid_stack_spec \
    --tokenizer-type NullTokenizer --vocab-size 248320 --make-vocab-size-divisible-by 1940 \
    --hidden-size 2560 --num-attention-heads 24 --kv-channels 256 --max-position-embeddings 262144 \
    --group-query-attention --num-query-groups 2 --qk-layernorm --attention-output-gate \
    --position-embedding-type rope --rotary-percent 0.25 --rotary-base 10000000 \
    --qsa-indexer-n-heads 4 --qsa-indexer-head-dim 128 --qsa-indexer-budget 2048 \
    --qsa-indexer-compress-ratio 4 --qsa-indexer-loss-coeff 0.01 \
    --linear-conv-kernel-dim 4 --linear-key-head-dim 128 --linear-value-head-dim 128 \
    --linear-num-key-heads 16 --linear-num-value-heads 48 --gdn-output-gate-activation sigmoid \
    --enable-mhc-connections --mhc-connection-variant gated_residual \
    --mhc-num-residual-streams 4 --hc-lowrank 320 \
    --num-experts 32 --moe-ffn-hidden-size 640 --moe-shared-expert-intermediate-size 640 \
    --moe-shared-expert-gate --moe-router-topk 10 --moe-router-score-function softmax \
    --moe-router-load-balancing-type aux_loss --moe-aux-loss-coeff 1e-3 --moe-router-dtype fp32 \
    --moe-token-dispatcher-type alltoall --moe-grouped-gemm --moe-permute-fusion \
    --engram-variant qwen --engram-layer-ids 3 --engram-vocab-sizes 1000000 1000000 \
    --engram-max-ngram-order 3 --engram-num-hash-heads 8 --engram-memory-dim 1280 \
    --engram-kernel-size 4 --engram-unigram-vocab-size 248320 \
    --engram-hash-seed 1234 --engram-eos-token-id 248044 \
    --mtp-num-layers 1 --mtp-loss-scaling-factor 0.1 \
    --normalization RMSNorm --apply-layernorm-1p --norm-epsilon 1e-6 --swiglu \
    --disable-bias-linear --untie-embeddings-and-output-weights \
    --attention-dropout 0.0 --hidden-dropout 0.0 \
    --no-weight-decay-cond-type apply_wd_to_qk_layernorm \
    --mock-data --seq-length 4096 --moe-router-force-load-balancing \
    --tensor-model-parallel-size 1 --pipeline-model-parallel-size 1 \
    --expert-model-parallel-size 4 --context-parallel-size 1 --expert-tensor-parallel-size 1 \
    --use-distributed-optimizer --overlap-grad-reduce --overlap-param-gather \
    --micro-batch-size 1 --global-batch-size 16 \
    --bf16 --transformer-impl transformer_engine --enable-experimental \
    --cross-entropy-loss-fusion --cross-entropy-fusion-impl native \
    --train-iters 200 --lr 3.0e-4 --min-lr 3.0e-5 --lr-decay-style cosine \
    --lr-warmup-iters 20 --lr-warmup-init 0.0 --weight-decay 0.1 --clip-grad 1.0 \
    --adam-beta1 0.9 --adam-beta2 0.95 --init-method-std 0.02 --seed 1234 \
    --log-interval 1 --log-memory-interval 1 --log-throughput \
    --eval-iters 0 --eval-interval 100 \
    "$@"

# Notes that cost a job each to learn:
#   * --eval-interval must stay set; the iteration-based data sizing divides by it.
#   * --mock-data needs --moe-router-force-load-balancing for meaningful timing: random tokens
#     give a badly skewed router. Drop both together when switching to real data
#     (--data-path ... --split 99,1,0 --tokenizer-type HuggingFaceTokenizer
#      --tokenizer-model Qwen/Qwen3.8-Flash-Next).
#   * Do NOT set NVTE_NORM_FWD_USE_CUDNN / NVTE_NORM_BWD_USE_CUDNN: measured 5-10x step time
#     on this model.
