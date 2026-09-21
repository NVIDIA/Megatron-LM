#!/usr/bin/env bash
# Qwen3.8-Flash-Next, the full 48-layer model on 64 GPUs (16 x 4 GB300), EP64, BF16.
#
# ---------------------------------------------------------------------------------------
# STATUS
#
# The argument block below is the configuration that was run end to end on 2026-09-14 for a
# 50-step weights-only resume: 16 s/step, 204.4 GiB peak allocated per rank, 248.6 GiB device
# memory used, 29 GiB headroom. See docs/models/qwen3.8_flash_next/training/full_model_64gpu.md.
#
# **This wrapper script itself has not been executed.** It is a faithful transcription of that
# argument list, not a separately validated artifact -- treat the first run as a validation run.
# ---------------------------------------------------------------------------------------
#
#   MASTER_ADDR=<host> LOAD=<dist_ckpt_dir> ./train_full_model_64gpu.sh [extra megatron args...]
#
# Launch one instance per node (16 nodes x 4 GPUs). On SLURM, resolve MASTER_ADDR *outside* the
# container -- `scontrol` is generally not installed inside it:
#
#   MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)
#   srun --nodes=16 --ntasks-per-node=1 ... bash examples/qwen3.8_flash_next/train_full_model_64gpu.sh
#
# LOAD is optional: set it to continue from a converted HF checkpoint (weights only), leave it
# empty to train from scratch.
set -uo pipefail

MLM="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$MLM" || exit 2

export CUDA_DEVICE_MAX_CONNECTIONS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

NNODES="${NNODES:-16}"
NPROC="${NPROC:-4}"
MASTER_ADDR="${MASTER_ADDR:?set MASTER_ADDR to the rank-0 host}"
MASTER_PORT="${MASTER_PORT:-29500}"
NODE_RANK="${NODE_RANK:-${SLURM_NODEID:-0}}"

# 12 x GEGEGEQE = 48 layers, plus a /QE MTP segment.
PATTERN=""
for _ in $(seq 12); do PATTERN="${PATTERN}GEGEGEQE"; done
PATTERN="${PATTERN}/QE"

LOAD_ARGS=()
if [ -n "${LOAD:-}" ]; then
  # The converted checkpoint holds weights only, so the optimizer and RNG state are absent and
  # the iteration counter restarts at 0.
  LOAD_ARGS=(--load "$LOAD" --no-load-optim --no-load-rng --finetune --ckpt-format torch_dist)
fi

python -m torch.distributed.run \
    --nproc_per_node "$NPROC" --nnodes "$NNODES" --node_rank "$NODE_RANK" \
    --master_addr "$MASTER_ADDR" --master_port "$MASTER_PORT" \
    pretrain_hybrid.py \
    --tokenizer-type HuggingFaceTokenizer --tokenizer-model Qwen/Qwen3.8-Flash-Next \
    --make-vocab-size-divisible-by 1940 \
    --hybrid-layer-pattern "$PATTERN" \
    --spec megatron.core.models.hybrid.hybrid_layer_specs gated_residual_hybrid_stack_spec \
    --hidden-size 2560 --num-attention-heads 24 --kv-channels 256 \
    --max-position-embeddings 262144 \
    --group-query-attention --num-query-groups 2 --qk-layernorm --attention-output-gate \
    --position-embedding-type rope --rotary-percent 0.25 --rotary-base 10000000 \
    --qsa-indexer-n-heads 4 --qsa-indexer-head-dim 128 --qsa-indexer-budget 2048 \
    --qsa-indexer-compress-ratio 4 --qsa-indexer-loss-coeff 0.01 \
    --linear-conv-kernel-dim 4 --linear-key-head-dim 128 --linear-value-head-dim 128 \
    --linear-num-key-heads 16 --linear-num-value-heads 48 --gdn-output-gate-activation sigmoid \
    --enable-mhc-connections --mhc-connection-variant gated_residual \
    --mhc-num-residual-streams 4 --hc-lowrank 320 \
    --num-experts 512 --moe-ffn-hidden-size 640 --moe-shared-expert-intermediate-size 640 \
    --moe-shared-expert-gate --moe-router-topk 10 --moe-router-score-function softmax \
    --moe-router-load-balancing-type aux_loss --moe-aux-loss-coeff 1e-3 --moe-router-dtype fp32 \
    --moe-token-dispatcher-type alltoall --moe-grouped-gemm --moe-permute-fusion \
    --engram-variant qwen --engram-layer-ids 3 --engram-vocab-sizes 20000000 20000000 \
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
    --expert-model-parallel-size 64 --context-parallel-size 1 --expert-tensor-parallel-size 1 \
    --use-distributed-optimizer --overlap-grad-reduce --overlap-param-gather \
    --micro-batch-size 1 --global-batch-size 512 \
    --bf16 --transformer-impl transformer_engine --enable-experimental \
    --cross-entropy-loss-fusion --cross-entropy-fusion-impl native \
    --lr 1e-5 --min-lr 1e-6 --lr-decay-style constant --weight-decay 0.1 --clip-grad 1.0 \
    --train-iters 50 --log-interval 1 --log-memory-interval 1 \
    --eval-iters 0 --eval-interval 1000 \
    "${LOAD_ARGS[@]}" \
    "$@"

# For a real pretraining run, append (values are placeholders for ~200 B tokens at 2 M/step):
#   --train-iters 100000 --lr 3.0e-4 --min-lr 3.0e-5 --lr-decay-style cosine \
#   --lr-warmup-iters 2000 --lr-warmup-init 0.0 --adam-beta1 0.9 --adam-beta2 0.95 \
#   --init-method-std 0.02 --seed 1234 \
#   --save <ckpt_dir> --load <ckpt_dir> --save-interval 500 --ckpt-format torch_dist \
#   --dist-ckpt-optim-fully-reshardable --exit-duration-in-mins 230
#
# Sizing, for changing the layout: activations are ~130 GiB at any EP; parameters + optimizer
# are 5.05 B replicated + 172 B / EP sharded, at 16 bytes per sharded parameter because
# expert-DP is 1 at this layout. EP32 measured 251 GiB and does not fit once the ~42 GiB of
# NCCL/all-to-all/context memory outside the allocator is counted.
