#!/bin/bash

export UB_TIMEOUT=720
export CUDA_DEVICE_MAX_CONNECTIONS=1
export NVTE_FWD_LAYERNORM_SM_MARGIN=16
export NVTE_BWD_LAYERNORM_SM_MARGIN=16
export NCCL_P2P_NET_CHUNKSIZE=2097152
export NCCL_DEBUG=WARN
export TORCHINDUCTOR_WORKER_START=fork
export NCCL_NVLS_ENABLE=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# export NVTE_FUSED_ATTN=0  # Disable cuDNN fused attention.
#export NVTE_FLASH_ATTN=1

if [ -n "$SLURM_JOB_USER" ]; then
    CACHE_DIR="/lustre/fsw/portfolios/coreai/users/$SLURM_JOB_USER/cache/"
else
    CACHE_DIR="/workspace/cache"
fi

export TRITON_CACHE_DIR="/lustre/fs1/portfolios/coreai/users/sheliang/megatron_artifacts/triton_cache"
export TRITON_HOME=$TRITON_CACHE_DIR

DATA_CACHE_PATH="${CACHE_DIR}/nm5-hybrid"

BLEND_PATH="/lustre/fsw/portfolios/llmservice/users/akhattar/data/phase2a-updatedcode-toolcall.json"
#     --moe-enable-deepep \
#     --moe-token-dispatcher-type flex \
COMMON_ARGS=" \
    --ckpt-format torch_dist \
    --attention-backend flash \
    --is-hybrid-model \
    --sequence-parallel \
    --tiktoken-pattern v2 \
    --distributed-timeout-minutes 20 \
    --use-mcore-models \
    --data-cache-path ${DATA_CACHE_PATH} \
    --no-mmap-bin-files \
    --untie-embeddings-and-output-weights \
    --disable-bias-linear \
    --init-method-std 0.0173 \
    --position-embedding-type none \
    --squared-relu \
    --normalization RMSNorm \
    --attention-dropout 0.0 \
    --hidden-dropout 0.0 \
    --exit-duration-in-mins 5750 \
    --train-samples 1024 \
    --lr-decay-samples 1024 \
    --lr-warmup-samples 0 \
    --lr-wsd-decay-samples 800 \
    --lr-wsd-decay-style minus_sqrt \
    --override-opt_param-scheduler \
    --lr 1e-3 \
    --min-lr 1e-5 \
    --weight-decay 0.1 \
    --clip-grad 1.0 \
    --lr-decay-style WSD \
    --log-interval 1 \
    --eval-iters 2 \
    --eval-interval 50 \
    --save-interval 100 \
    --save-retain-interval 10000 \
    --use-persistent-ckpt-worker \
    --ckpt-fully-parallel-load \

    --log-progress  \
    --timing-log-option minmax \
    --log-params-norm \
    --log-num-zeros-in-grad \
    --log-throughput \
    --bf16 \
    --adam-beta1 0.9 \
    --adam-beta2 0.95 \
    --use-distributed-optimizer \
    --no-create-attention-mask-in-dataloader \
    --manual-gc \
    --num-workers 8 \
    --logging-level 20 \
    --log-memory-interval 500 \
    --log-energy \
    --ddp-num-buckets 8 \
    --ddp-pad-buckets-for-high-nccl-busbw \
    --overlap-grad-reduce \
    --overlap-param-gather \
    --disable-gloo-process-groups \
    --log-straggler \
    --disable-straggler-on-startup \
    --straggler-minmax-count 16 \
    --check-weight-hash-across-dp-replicas-interval 20000 \
"

# Disabled args
#    --save-retain-interval 10000 \
#    --ckpt-format torch_dist \
#    --ckpt-fully-parallel-save \
#    --async-save \
