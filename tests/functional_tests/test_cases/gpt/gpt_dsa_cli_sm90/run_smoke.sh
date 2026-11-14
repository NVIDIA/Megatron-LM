#!/usr/bin/env bash
# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

# Run from the repository root in the standard Megatron GPU development container.
# Uses synthetic data, multiple microbatches, and checkpoint/resume; no golden values are fabricated.
set -euo pipefail

export CUDA_DEVICE_MAX_CONNECTIONS=1
smoke_dir=${CLI_SMOKE_DIR:-$(mktemp -d -t megatron-cli-sm90-XXXXXX)}
mkdir -p "${smoke_dir}"

common=(
    --num-layers 4 --hidden-size 512 --ffn-hidden-size 2048 --num-attention-heads 32
    --multi-latent-attention --q-lora-rank 128 --kv-lora-rank 512
    --qk-head-dim 128 --qk-pos-emb-head-dim 64 --v-head-dim 128
    --experimental-attention-variant dsa --disable-bias-linear
    --dsa-kernel-backend cudnn --dsa-indexer-n-heads 32 --dsa-indexer-head-dim 128
    --dsa-indexer-topk 256 --dsa-indexer-topk-freq 2 --dsa-indexer-skip-topk-offset 1
    --dsa-indexer-use-sparse-loss --dsa-indexer-loss-coeff 0.001
    --dsa-indexer-cross-layer-distillation
    --recompute-granularity full --recompute-method uniform --recompute-num-layers 1
    --tensor-model-parallel-size 1 --pipeline-model-parallel-size 1
    --micro-batch-size 1 --global-batch-size 4 --seq-length 512 --max-position-embeddings 512
    --attention-dropout 0 --hidden-dropout 0 --bf16
    --transformer-impl transformer_engine --no-gradient-accumulation-fusion
    --mock-data --tokenizer-type NullTokenizer --vocab-size 1024
    --lr 0.0001 --min-lr 0.00001 --lr-decay-iters 4
    --log-interval 1 --eval-iters 0 --save-interval 2
    --save "${smoke_dir}/checkpoint" --load "${smoke_dir}/checkpoint"
)

python -m torch.distributed.run --nproc-per-node 1 pretrain_gpt.py \
    "${common[@]}" --train-iters 2 2>&1 | tee "${smoke_dir}/initial.log"
python -m torch.distributed.run --nproc-per-node 1 pretrain_gpt.py \
    "${common[@]}" --train-iters 4 2>&1 | tee "${smoke_dir}/resume.log"
printf 'CLI smoke logs and checkpoints: %s\n' "${smoke_dir}"
