#!/usr/bin/env bash
# Qwen3.8-Flash-Next proxy, one node (4 GPUs), EP4, BF16, from scratch.
#
# This proxy keeps every per-layer, per-expert and PLE dimension of the 48-layer model and
# shrinks only what sets the total size, so per-rank shapes on 4 GPUs match the 64-GPU model:
# 8 layers instead of 48, 32 experts at EP4 instead of 512 at EP64 (8 local either way), and a
# 1 M n-gram base instead of 20 M. It is a functional and recipe-development proxy, not a
# memory proxy for EP64. See examples/qwen4exp/docs/training/proxy_single_node.md.
#
# Two layouts (LAYOUT=...):
#   thd32k (default)  packed THD, one 32K-token document per rank (the layout of training), with
#                     the training kernels: TE CUDA graphs over attention and the MoE router
#                     (Engram lookup run eagerly in front of its layer's graph), the MoK expert
#                     megakernel, cuDNN GDN with the pre-gated-delta-rule fusion, the CuTe DSL
#                     sparse QSA path and fused indexer loss. Vocabulary 40832 (see below).
#                     Needs the image of examples/qwen4exp/Dockerfile.gb300 (MoK, causal-conv1d).
#   bshd4k            unpacked BSHD at seq 4096 with the real 248320 vocabulary and stock kernels;
#                     the baseline that examples/qwen4exp/docs/validation/support_matrix.md was
#                     measured on, and the one that runs in any container with TE, fla, Triton
#                     and emerging-optimizers.
#
# Runs on Megatron's mock data with NullTokenizer, so nothing has to be downloaded. Optimizer:
# Muon on the layer-wise distributed optimizer (the 2-D matrices; the remaining parameters go to
# its Adam siblings, which the --adam-* flags configure), with the Muon defaults; needs
# emerging-optimizers. Pass --optimizer adam for Adam everywhere. Status: thd32k has been run with
# Muon, and with Adam plus --use-precision-aware-optimizer (40 iterations each, 4 x GB300);
# bshd4k has been run with --optimizer adam.
#
# Performance, LAYOUT=thd32k on one node of 4 x GB300, with the dependencies pinned in
# examples/qwen4exp/Dockerfile.gb300; mean of iterations 21-40 of a 40-iteration run, TFLOP/s as
# reported by --log-throughput (model FLOPs only):
#   default (--optimizer muon)                       2677 ms/step  384 TFLOP/s/GPU  126.9 GiB peak
#   --optimizer adam --use-precision-aware-optimizer \
#     --main-grads-dtype fp32 --main-params-dtype fp32 \
#     --exp-avg-dtype bf16 --exp-avg-sq-dtype bf16   2394 ms/step  430 TFLOP/s/GPU  122.3 GiB peak
# Muon's Newton-Schulz work is not counted as model FLOPs, so it shows up as lower TFLOP/s.
# Back-to-back runs on one node agree within 0.1 %; across nodes within 0.5 %.
#
#   [LAYOUT=bshd4k] ./train_proxy_1node.sh [extra megatron args...]
#
# Later flags win in Megatron's argparse, so pass overrides as arguments rather than editing the
# blocks below.
set -uo pipefail

MLM="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$MLM" || exit 2

export CUDA_DEVICE_MAX_CONNECTIONS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Megatron asserts lr_warmup_steps < lr_decay_steps, and lr_decay_steps follows
# --train-iters. The defaults below are 200/20; a short smoke run such as
#   ./train_proxy_1node.sh --train-iters 3
# would therefore trip that assertion on an argument the caller never touched. Scale the
# warmup with the requested iteration count so the documented smoke command works, while
# still letting an explicit --lr-warmup-iters later on the command line win.
TRAIN_ITERS=200
LR_WARMUP_ITERS=20
_prev=""
for _arg in "$@"; do
    if [ "$_prev" = "--train-iters" ]; then TRAIN_ITERS="$_arg"; fi
    _prev="$_arg"
done
if [ "$TRAIN_ITERS" -le "$LR_WARMUP_ITERS" ]; then
    LR_WARMUP_ITERS=$(( TRAIN_ITERS > 1 ? TRAIN_ITERS - 1 : 0 ))
fi

LAYOUT="${LAYOUT:-thd32k}"
case "$LAYOUT" in
  thd32k)
    # FLA_DISABLE_TENSOR_CACHE: fla's tensor cache would freeze chunk metadata across
    # CUDA-graph replays, and GDN refuses THD attention capture without it. The rest is the
    # environment the recorded runs used.
    export FLA_DISABLE_TENSOR_CACHE=1 NVTE_FUSED_ATTN=1 NCCL_NVLS_ENABLE=0 NCCL_GRAPH_REGISTER=0
    export NUM_OF_TOKENS_PER_CHUNK_DISPATCH_API=128 NUM_OF_TOKENS_PER_CHUNK_COMBINE_API=128
    export NUM_OF_TOKENS_PER_CHUNK_PREPROCESSING_API=128
    # Vocabulary 40832 (= 319 x 128, about 1/6 of the real 248320): scaled down by the same 1/6
    # as the layers (8 of 48), so the embedding / output-layer / cross-entropy share of the step
    # matches the 48-layer model at 32K tokens per rank; at the real vocabulary the vocab-sized
    # tensors would dominate this 8-layer proxy. The PLE unigram vocabulary follows it, and the
    # EOS id is its last token. NullTokenizer + mock data, so no tokenizer depends on it.
    LAYOUT_ARGS=(
      --vocab-size 40832 --make-vocab-size-divisible-by 128
      --engram-unigram-vocab-size 40832 --engram-eos-token-id 40831
      --seq-length 32768 --eval-iters 1 --split 98,2,0
      --sft --pad-packed-seq-alignment max --max-seqlen-per-dp-cp-rank 32768
      --thd-max-packed-sequences 1
      --sft-mock-dataset-config-json '{"mode":"distribution","type":"lognormal","format":"thd","min_seq_len":32768,"max_seq_len":32768,"mean_seq_len":32768,"lognormal_sigma":1.1}'
      --cuda-graph-impl transformer_engine --cuda-graph-modules attn moe_router
      --cuda-graph-warmup-steps 2 --cuda-graph-capture-engram-host --te-rng-tracker
      --moe-megakernel-backend mok
      --moe-megakernel-backend-config '{"fwd_num_comm_sms":48,"bwd_num_comm_sms":56,"minibatch_size":8192,"macrobatch_size":65536,"schedule_capacity_multiplier":0.0625,"all_gather_top_experts_chunk_bytes":2048}'
      --moe-router-fusion
      --gdn-kernel-backend cudnn --gdn-pre-gated-delta-rule-fusion
      --qsa-use-sparse-attention --qsa-indexer-loss-impl fused --qsa-indexer-loss-chunk 1024
      --manual-gc --manual-gc-interval 100 --rerun-mode disabled
      --no-check-for-nan-in-loss-and-grad --num-workers 6 --log-device-memory-used
    )
    ;;
  bshd4k)
    LAYOUT_ARGS=(
      --vocab-size 248320 --make-vocab-size-divisible-by 1940
      --engram-unigram-vocab-size 248320 --engram-eos-token-id 248044
      --seq-length 4096 --eval-iters 0
    )
    ;;
  *)
    echo "LAYOUT must be thd32k or bshd4k, got '$LAYOUT'" >&2
    exit 2
    ;;
esac

python -m torch.distributed.run --nproc_per_node 4 --nnodes 1 pretrain_hybrid.py \
    --hybrid-layer-pattern GEGEGEQEGEGEGEQE/QE \
    --spec megatron.core.models.hybrid.hybrid_layer_specs gated_residual_hybrid_stack_spec \
    --tokenizer-type NullTokenizer \
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
    --engram-kernel-size 4 --engram-hash-seed 1234 \
    --mtp-num-layers 1 --mtp-loss-scaling-factor 0.1 \
    --normalization RMSNorm --apply-layernorm-1p --norm-epsilon 1e-6 --swiglu \
    --disable-bias-linear --untie-embeddings-and-output-weights \
    --attention-dropout 0.0 --hidden-dropout 0.0 \
    --apply-wd-to-qk-layernorm \
    --mock-data --moe-router-force-load-balancing \
    --tensor-model-parallel-size 1 --pipeline-model-parallel-size 1 \
    --expert-model-parallel-size 4 --context-parallel-size 1 --expert-tensor-parallel-size 1 \
    --optimizer muon --use-distributed-optimizer --overlap-grad-reduce --overlap-param-gather \
    --micro-batch-size 1 --global-batch-size 16 \
    --bf16 --transformer-impl transformer_engine --enable-experimental \
    --cross-entropy-loss-fusion --cross-entropy-fusion-impl native \
    --train-iters "$TRAIN_ITERS" --lr 3.0e-4 --min-lr 3.0e-5 --lr-decay-style cosine \
    --lr-warmup-iters "$LR_WARMUP_ITERS" --lr-warmup-init 0.0 --weight-decay 0.1 --clip-grad 1.0 \
    --adam-beta1 0.9 --adam-beta2 0.95 --init-method-std 0.02 --seed 1234 \
    --log-interval 1 --log-memory-interval 1 --log-throughput \
    --eval-interval 100 \
    "${LAYOUT_ARGS[@]}" \
    "$@"

# Notes that cost a job each to learn:
#   * --eval-interval must stay set; the iteration-based data sizing divides by it.
#   * --sft builds a validation loader, so thd32k needs --eval-iters >= 1 ("no sample to consume").
#   * --mock-data needs --moe-router-force-load-balancing for meaningful timing: random tokens
#     give a badly skewed router. Drop both together when switching to real data
#     (--data-path ... --split 99,1,0 --tokenizer-type HuggingFaceTokenizer
#      --tokenizer-model Qwen/Qwen3.8-Flash-Next, and the real vocabulary of bshd4k).
#   * Under THD with attention captured, --gdn-kernel-backend internal falls back to fla and
#     --qsa-attn-backend cudnn needs --qsa-cudnn-bwd-impl qsa_api.
#   * Do NOT set NVTE_NORM_FWD_USE_CUDNN / NVTE_NORM_BWD_USE_CUDNN: a large step-time
#     regression on this model.
