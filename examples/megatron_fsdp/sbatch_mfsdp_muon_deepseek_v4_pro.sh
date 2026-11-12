#!/bin/bash
#SBATCH --job-name=mfsdp-muon-deepseek-v4-pro
#SBATCH --nodes=16
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=36
#SBATCH --exclusive
#SBATCH --time=00:45:00

# DeepSeek-V4 Pro width and expert layout with 7 decoder blocks, trained with Megatron-FSDP v2
# (ZeRO-3) and the Muon optimizer on 64 GB200 GPUs (16 nodes x 4): EP64, MXFP8, HybridEP,
# mock data, 12 iterations. See README.md for the required kernels and how to read the
# optimizer step time.
#
# Usage (set --account and --partition for your cluster; on NVL72, --segment=16 keeps the job
# in one NVLink domain):
#   MEGATRON_PATH=/path/to/Megatron-LM CONTAINER_IMAGE=<image> CONTAINER_MOUNTS=/lustre:/lustre \
#   OUTPUT_PATH=/path/to/output sbatch --account=<account> --partition=<partition> \
#       examples/megatron_fsdp/sbatch_mfsdp_muon_deepseek_v4_pro.sh

set -euo pipefail

MEGATRON_PATH=${MEGATRON_PATH:?Set MEGATRON_PATH to the Megatron-LM checkout}
CONTAINER_IMAGE=${CONTAINER_IMAGE:?Set CONTAINER_IMAGE to the training container}
CONTAINER_MOUNTS=${CONTAINER_MOUNTS:?Set CONTAINER_MOUNTS to mounts that include MEGATRON_PATH and OUTPUT_PATH}
OUTPUT_PATH=${OUTPUT_PATH:?Set OUTPUT_PATH to a shared output directory}
PYTHON=${PYTHON:-python}  # A Python inside the container that has the kernels in README.md.
PROFILE=${PROFILE:-0}  # 1: Nsight Systems on PROFILE_RANKS for steps 9-11.
PROFILE_RANKS=${PROFILE_RANKS:-"0 32 48 63"}
EXTRA_ARGS=${EXTRA_ARGS:-}  # Extra training flags, e.g. "--train-iters 20".

RUN_DIR="$OUTPUT_PATH/mfsdp-muon-deepseek-v4-pro-${SLURM_JOB_ID:?Submit this script with sbatch}"
mkdir -p "$RUN_DIR/logs" "$RUN_DIR/nsys" "$RUN_DIR/cache/dataset" "$RUN_DIR/cache/triton"

export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)
export MASTER_PORT=${MASTER_PORT:-6000}
export WORLD_SIZE=$SLURM_NTASKS
export PYTHONPATH="$MEGATRON_PATH${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME=${HF_HOME:-"$OUTPUT_PATH/cache/huggingface"}
export TORCH_EXTENSIONS_DIR="$RUN_DIR/cache/torch-extensions"
export RUN_DIR PYTHON PROFILE PROFILE_RANKS

# Blackwell does not need CUDA_DEVICE_MAX_CONNECTIONS, and Megatron-FSDP must not use 1.
unset CUDA_DEVICE_MAX_CONNECTIONS
export TORCH_NCCL_AVOID_RECORD_STREAMS=0
export NVTE_ALLOW_NONDETERMINISTIC_ALGO=1
export PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True'
export NCCL_NVLS_ENABLE=0
export NVTE_FUSED_ATTN=1
export NVTE_NORM_FWD_USE_CUDNN=1
export NVTE_NORM_BWD_USE_CUDNN=1
export NVTE_CUTEDSL_FUSED_GROUPED_MLP=1
export NUM_OF_TOKENS_PER_CHUNK_COMBINE_API=128
export NUM_OF_STAGES_DISPATCH_API=10
export NUM_OF_IN_FLIGHT_S2G_DISPATCH_API=8
# Compile Inductor kernels in-process: on GB200, ranks have hung in the first step waiting for
# a compile-pool worker.
export TORCHINDUCTOR_COMPILE_THREADS=1

# H: HCA, C: CSA, W: window attention, E: MoE; "/WE" is one MTP depth.
MODEL_ARGS=(
    --tokenizer-type HuggingFaceTokenizer
    --tokenizer-model unsloth/DeepSeek-V3
    --hybrid-layer-pattern "HEHECEHECEHECE/WE"
    --spec megatron.core.models.hybrid.hybrid_layer_specs hybrid_dsv4_stack_spec
    --hidden-size 7168
    --num-attention-heads 128
    --kv-channels 512
    --max-position-embeddings 4096
    --normalization RMSNorm
    --norm-epsilon 1e-6
    --swiglu
    --disable-bias-linear
    --untie-embeddings-and-output-weights
    --position-embedding-type rope
    --rotary-base 10000
    --make-vocab-size-divisible-by 3232
    --multi-latent-attention
    --q-lora-rank 1536
    --qk-pos-emb-head-dim 64
    --v-head-dim 512
    --rotary-scaling-factor 4
    --mscale 1.0
    --mscale-all-dim 1.0
    --qk-layernorm
    --output-projection-groups 16
    --output-projection-lora-rank 1024
    --original-max-position-embeddings 65536
    --experimental-attention-variant dsv4_hybrid
    --csa-window-size 128
    --csa-compress-ratios "([128,128,4]+[128,4]*2+[0])"
    --csa-compress-rotary-base 40000
    --dsa-indexer-n-heads 64
    --dsa-indexer-head-dim 128
    --dsa-indexer-topk 1024
    --dsa-indexer-loss-coeff 1e-2
    --dsa-indexer-use-sparse-loss
    --num-experts 384
    --moe-num-hash-layers 3
    --vocab-size 128815
    --moe-ffn-hidden-size 3072
    --moe-shared-expert-intermediate-size 3072
    --moe-router-load-balancing-type seq_aux_loss
    --moe-router-topk 6
    --moe-aux-loss-coeff 1e-4
    --moe-router-topk-scaling-factor 2.5
    --moe-router-score-function sqrtsoftplus
    --moe-router-enable-expert-bias
    --moe-router-bias-update-rate 1e-3
    --activation-func-clamp-value 10.0
    --enable-mhc-connections
    --mhc-num-residual-streams 4
    --mhc-sinkhorn-iterations 20
    --use-fused-mhc
    --mtp-num-layers 1
    --mtp-loss-scaling-factor 0.1
    --attention-dropout 0.0
    --hidden-dropout 0.0
)

# 64 GPUs: dense DP64 with Megatron-FSDP v2 ZeRO-3; EP64, so every rank holds whole experts.
PARALLEL_ARGS=(
    --distributed-timeout-minutes 60
    --tensor-model-parallel-size 1
    --pipeline-model-parallel-size 1
    --expert-model-parallel-size 64
    --context-parallel-size 1
    --expert-tensor-parallel-size 1
    --use-megatron-fsdp
    --megatron-fsdp-version 2
    --data-parallel-sharding-strategy optim_grads_params
    --suggested-communication-unit-size 0
    --no-gradient-accumulation-fusion
    --sequence-parallel
    --overlap-grad-reduce
    --overlap-param-gather
)

RUNTIME_ARGS=(
    --moe-token-dispatcher-type flex
    --moe-flex-dispatcher-backend hybridep
    --moe-hybridep-num-sms 32
    --moe-grouped-gemm
    --moe-permute-fusion
    --moe-router-fusion
    --moe-router-dtype fp32
    --moe-router-force-load-balancing
    --use-transformer-engine-op-fuser
    --moe-mlp-glu-interleave-size 32
    --moe-expert-rank-capacity-factor 1.5
    --moe-paged-stash
    --moe-paged-stash-buffer-size-factor-cuda 1.2
    --cuda-graph-impl none
)

TRAIN_ARGS=(
    --optimizer muon
    --muon-momentum 0.95
    --muon-coefficient-type deepseekv4
    --muon-num-ns-steps 10
    --muon-no-split-qkv
    --ckpt-format fsdp_dtensor
    --mock-data
    --use-mcore-models
    --use-flash-attn
    --transformer-impl transformer_engine
    --micro-batch-size 1
    --global-batch-size 64
    --seq-length 4096
    --train-iters 12
    --exit-duration-in-mins 220
    --no-save-optim
    --no-check-for-nan-in-loss-and-grad
    --cross-entropy-loss-fusion
    --cross-entropy-fusion-impl native
    --no-create-attention-mask-in-dataloader
    --num-workers 0
    --num-dataset-builder-threads 1
    --manual-gc
    --manual-gc-interval 10
    --lr 2.0e-4
    --min-lr 2.0e-5
    --lr-warmup-init 2.0e-5
    --lr-decay-style cosine
    --lr-decay-iters 12
    --lr-warmup-iters 0
    --weight-decay 0.1
    --clip-grad 1.0
    --adam-beta1 0.9
    --adam-beta2 0.95
    --bf16
    --fp8-recipe mxfp8
    --fp8-format e4m3
    --fp8-param-gather
    --moe-router-padding-for-quantization
    --init-method-std 0.02
    --eval-iters 0
    --eval-interval 200
    --no-load-optim
    --no-load-rng
    --auto-detect-ckpt-format
    --save-interval 500
    --dist-ckpt-strictness log_all
    --enable-experimental
)

LOG_ARGS=(
    --timing-log-level 0
    --log-throughput
    --log-interval 1
    --log-memory-interval 1
    --data-cache-path "$RUN_DIR/cache/dataset"
)
if [[ $PROFILE == 1 ]]; then
    # Megatron starts and stops the CUDA profiler on these ranks; NVTX ranges mark each step's
    # forward, backward, and optimizer phases.
    read -ra profile_ranks <<< "$PROFILE_RANKS"
    LOG_ARGS+=(--profile --profile-step-start 8 --profile-step-end 11
               --profile-ranks "${profile_ranks[@]}" --nvtx-ranges)
fi

read -ra extra_args <<< "$EXTRA_ARGS"
LOG_ARGS+=("${extra_args[@]}")

# One task per GPU, so Nsight Systems wraps only the profiled ranks.
RANK_COMMAND='
set -euo pipefail
export RANK=$SLURM_PROCID LOCAL_RANK=$SLURM_LOCALID
export TRITON_CACHE_DIR="$RUN_DIR/cache/triton/$RANK"
if [[ $PROFILE == 1 && " $PROFILE_RANKS " == *" $RANK "* ]]; then
    exec nsys profile --trace=cuda,nvtx --sample=none --cpuctxsw=none \
        --capture-range=cudaProfilerApi --capture-range-end=stop --force-overwrite=true \
        --output="$RUN_DIR/nsys/rank$RANK" "$PYTHON" -u "$@"
fi
exec "$PYTHON" -u "$@"
'

# Every task sees the node's four GPUs and picks its own by LOCAL_RANK.
srun --label --export=ALL --kill-on-bad-exit=1 --cpu-bind=none --gpu-bind=none \
    --container-image="$CONTAINER_IMAGE" --container-mounts="$CONTAINER_MOUNTS" \
    --container-workdir="$MEGATRON_PATH" --no-container-mount-home \
    --output="$RUN_DIR/logs/rank-%t.log" \
    bash -c "$RANK_COMMAND" rank "$MEGATRON_PATH/pretrain_hybrid.py" \
    "${MODEL_ARGS[@]}" "${PARALLEL_ARGS[@]}" "${RUNTIME_ARGS[@]}" "${TRAIN_ARGS[@]}" "${LOG_ARGS[@]}"
