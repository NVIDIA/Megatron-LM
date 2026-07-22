#!/bin/bash

# Two-GPU MFSDP v2 + expert-parallel smoke example. Routed experts are
# partitioned by EP and use singleton expert-DP meshes; all other parameters
# are fully sharded over the two-rank dense DP mesh.
set -euo pipefail

unset CUDA_DEVICE_MAX_CONNECTIONS

"${PYTHON:-python}" -m torch.distributed.run --standalone --nproc-per-node 2 pretrain_hybrid.py \
    --spec megatron.core.models.hybrid.hybrid_layer_specs hybrid_stack_spec \
    --hybrid-layer-pattern ME \
    --num-layers 2 \
    --hidden-size 128 \
    --ffn-hidden-size 256 \
    --moe-ffn-hidden-size 256 \
    --num-attention-heads 2 \
    --mamba-num-heads 2 \
    --mamba-head-dim 64 \
    --mamba-num-groups 1 \
    --mamba-state-dim 64 \
    --position-embedding-type none \
    --normalization RMSNorm \
    --swiglu \
    --disable-bias-linear \
    --untie-embeddings-and-output-weights \
    --attention-dropout 0.0 \
    --hidden-dropout 0.0 \
    --num-experts 2 \
    --moe-router-topk 1 \
    --moe-router-pre-softmax \
    --moe-router-load-balancing-type aux_loss \
    --moe-aux-loss-coeff 1.0e-2 \
    --moe-token-dispatcher-type alltoall \
    --moe-grouped-gemm \
    --tensor-model-parallel-size 1 \
    --pipeline-model-parallel-size 1 \
    --context-parallel-size 1 \
    --expert-model-parallel-size 2 \
    --expert-tensor-parallel-size 1 \
    --use-megatron-fsdp \
    --megatron-fsdp-version 2 \
    --use-distributed-optimizer \
    --data-parallel-sharding-strategy optim_grads_params \
    --outer-dp-sharding-strategy no_shard \
    --ckpt-format fsdp_dtensor \
    --no-gradient-accumulation-fusion \
    --bf16 \
    --micro-batch-size 1 \
    --global-batch-size 2 \
    --seq-length 128 \
    --max-position-embeddings 128 \
    --train-iters 5 \
    --lr 1.0e-4 \
    --min-lr 1.0e-5 \
    --lr-decay-style cosine \
    --lr-warmup-fraction 0.01 \
    --weight-decay 0.0 \
    --clip-grad 1.0 \
    --mock-data \
    --tokenizer-type NullTokenizer \
    --vocab-size 1024 \
    --split 99,1,0 \
    --num-workers 1 \
    --no-create-attention-mask-in-dataloader \
    --eval-iters 0 \
    --eval-interval 100 \
    --log-interval 1 \
    --distributed-backend nccl
