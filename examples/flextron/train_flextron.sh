#!/bin/bash
# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# Reference launcher for Flextron training on an 8xH100 node.
#
# Usage:
#   RUN_DIR=<output-root> \
#   BLEND_PATH=<data-blend.json> \
#   TEACHER=<teacher-checkpoint-dir> \
#   TOKENIZER_MODEL=<tokenizer-path> \
#   bash examples/flextron/train_flextron.sh
#
# All four paths must be supplied — example defaults below will not work
# outside their original environment and exist only as placeholders.

export UB_TIMEOUT=720
export CUDA_DEVICE_MAX_CONNECTIONS=1
export NVTE_FWD_LAYERNORM_SM_MARGIN=16
export NVTE_BWD_LAYERNORM_SM_MARGIN=16
export NCCL_P2P_NET_CHUNKSIZE=2097152
export NCCL_DEBUG=WARN
export TORCHINDUCTOR_WORKER_START=fork

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

########################################################
#### User-configurable paths (override via env vars) ####
########################################################

# Root for run artifacts (logs / checkpoints / tensorboard / triton cache).
RUN_DIR=${RUN_DIR:-./runs/flextron}

# Training data blend (GPT-style .bin/.idx or blend-config JSON).
BLEND_PATH=${BLEND_PATH:-/path/to/your/data_blend.json}

# Teacher checkpoint used for distillation + as the init for flextron finetuning.
TEACHER=${TEACHER:-/path/to/teacher/checkpoint}

# Tokenizer (SFTTokenizer path or TikToken vocab file depending on BLEND_NAME).
TOKENIZER_MODEL=${TOKENIZER_MODEL:-/path/to/tokenizer}

########################################################
#### CHANGES SHOULD NOT BE NEEDED BEYOND THIS POINT ####
########################################################

DATETIME=`date +'date_%y-%m-%d_time_%H-%M-%S'`
NAME="interactive"
REPO_DIR=`pwd`

LOGS_DIR="${RUN_DIR}/logs/${NAME}"
CHECKPOINT_DIR="${RUN_DIR}/checkpoints/${NAME}"
TENSORBOARD_DIR="${RUN_DIR}/tensorboard/${NAME}"
DATACACHE_DIR="${RUN_DIR}/data_cache"

mkdir -p ${LOGS_DIR}
mkdir -p ${CHECKPOINT_DIR}
mkdir -p ${DATACACHE_DIR}
mkdir -p ${TENSORBOARD_DIR}

# Mamba triton cache - using rank-specific directories to avoid conflicts.
export TRITON_CACHE_DIR="${RUN_DIR}/triton_cache"
export TRITON_HOME=$TRITON_CACHE_DIR

# CRITICAL: Disable Dynamo/Inductor compilation for PP2 stability
export TORCH_COMPILE_DISABLE=1
export TORCHDYNAMO_DISABLE=1
export TORCH_INDUCTOR_DISABLE=1

# Teacher checkpoint is also used as the starting point for flextron finetuning.
LOAD_CHECKPOINT_DIR=${TEACHER}

# Iterations with rare tokens that we should skip.
# ITERATIONS_TO_SKIP=""

# Optional args that can be incorporated later.
# --iterations-to-skip ${ITERATIONS_TO_SKIP} \
# --result-rejected-tracker-filename ${CHECKPOINT_DIR}/result_rejected_tracker.txt \
# --rerun-mode validate_results \

# --manual-gc-interval 10 \
# --recompute-granularity selective \
# --recompute-modules layernorm (moe_act) \
# --moe-token-dispatcher-type alltoall \

# 200 step
# 40 wsd
# LR search 
# Constant warmup of 40-50 steps


# 5% -> 1e-5
# 10% -> 5e-5
# 20% -> 5e-5
# 30% -> 1e-4
# LR decay style cosine



GBS=${GBS:-32}
LR_WARMUP_STEPS=${LR_WARMUP_STEPS:-40}
NUM_STEPS=${NUM_STEPS:-300}
LR=${LR:-1e-4}
MIN_LR=${MIN_LR:-1e-5}
TRAIN_SAMPLES=10000 #Hardcoded to reuse data cache
LR_WARMUP_SAMPLES=$((GBS * LR_WARMUP_STEPS))
LR_DECAY_SAMPLES=$((TRAIN_SAMPLES - LR_WARMUP_SAMPLES))
LR_WSD_DECAY_SAMPLES=$(awk "BEGIN {printf \"%.0f\", ${TRAIN_SAMPLES} * 0.2}") # 20% of the training samples
ORIGINAL_MODEL_SAMPLE_PROB=${ORIGINAL_MODEL_SAMPLE_PROB:-0.0}
LINEAR_SCALER_START=${LINEAR_SCALER_START:-1.0}
LINEAR_SCALER_END=${LINEAR_SCALER_END:-10.0}
TAU_INIT=${TAU_INIT:-1.0}
TAU_DECAY=${TAU_DECAY:-0.9997}
LOSS_ALPHA=${LOSS_ALPHA:-1.0}
ROUTER_STD=${ROUTER_STD:-0.1}
ROUTER_LR=${ROUTER_LR:-1e-2}
LR_MULT_ROUTER=${LR_MULT_ROUTER:-$(awk "BEGIN {print ${ROUTER_LR} / ${LR}}")}
ROUTER_GBS=${ROUTER_GBS:-2}

BUDGET=${BUDGET:-"1.0 0.697"}
SAMPLE_PROBS=${SAMPLE_PROBS:-"1.0 1.0"}

PP=${PP:-1}
TP=${TP:-8}
EP=${EP:-8}
CP=${CP:-1}
SEQLEN=${SEQLEN:-8192}

if [ "$PP" == "1" ]; then
    HYBRID_OVERRIDE_PATTERN='MEMEM*EMEMEM*EMEMEM*EMEMEM*EMEMEM*EMEMEMEM*EMEMEMEME'
elif [ "$PP" == "2" ]; then
    # CRITICAL: Escape the pipe character to prevent shell interpretation
    HYBRID_OVERRIDE_PATTERN='MEMEM*EMEMEM*EMEMEM*EMEMEM|*EMEMEM*EMEMEMEM*EMEMEMEME'
fi

# Paths specific to phase 1.
# NAME="fex_LR_${LR}_MIN_LR_${MIN_LR}_GBS_${GBS}_ROUTER_GBS_${ROUTER_GBS}_ROUTER_LR_${ROUTER_LR}_TAU_INIT_${TAU_INIT}_TAU_DECAY_${TAU_DECAY}_LOSS_ALPHA_${LOSS_ALPHA}_ROUTER_STD_${ROUTER_STD}"



# TODO: verify parallelism/mbs 25T tokens etc.
# per-split-data-args-path ${BLEND_PATH}
# MEMEM*EMEMEM*EMEMEM*EMEMEM*EMEMEM*EMEMEMEM*EMEMEMEME 


options=" \
    --moe-token-dispatcher-type alltoall \
    --moe-shared-expert-overlap \
    --enable-experimental \
    --moe-permute-fusion \
    --use-fused-weighted-squared-relu \
    --cross-entropy-loss-fusion \
    --cross-entropy-fusion-impl native \
    --moe-router-score-function sigmoid \
    --moe-grouped-gemm \
    --num-experts 128 \
    --moe-router-topk 6 \
    --moe-aux-loss-coeff 1e-4 \
    --moe-router-topk-scaling-factor 2.5 \
    --moe-router-enable-expert-bias \
    --moe-router-dtype fp32 \
    --moe-router-load-balancing-type none \
    --moe-shared-expert-intermediate-size 3712 \
    \
    --attention-backend flash \
    --disable-gloo-process-groups \
    --is-hybrid-model \
    --mamba-num-heads 64 \
    --mamba-head-dim 64 \
    --hybrid-override-pattern ${HYBRID_OVERRIDE_PATTERN} \
    --per-split-data-args-path ${BLEND_PATH} \
    --distributed-timeout-minutes 20 \
    --use-mcore-models \
    --data-cache-path ${DATACACHE_DIR} \
    --no-mmap-bin-files \
    --sequence-parallel \
    --untie-embeddings-and-output-weights \
    --disable-bias-linear \
    --init-method-std 0.0173 \
    --position-embedding-type none \
    --squared-relu \
    --num-layers 52 \
    --hidden-size 1920 \
    --num-attention-heads 32 \
    --group-query-attention \
    --num-query-groups 2 \
    --ffn-hidden-size 960 \
    --kv-channels 128 \
    --normalization RMSNorm \
    --attention-dropout 0.0 \
    --hidden-dropout 0.0 \
    --exit-duration-in-mins 230 \
    --tensor-model-parallel-size ${TP} \
    --expert-model-parallel-size ${EP} \
    --expert-tensor-parallel-size 1 \
    --pipeline-model-parallel-size ${PP} \
    --context-parallel-size ${CP} \
    --seq-length ${SEQLEN} \
    --max-position-embeddings ${SEQLEN} \
    --micro-batch-size 1 \
    --global-batch-size ${GBS} \
    --train-samples ${TRAIN_SAMPLES} \
    --lr-warmup-samples ${LR_WARMUP_SAMPLES} \
    --lr-decay-samples ${LR_DECAY_SAMPLES} \
    --lr-wsd-decay-samples ${LR_WSD_DECAY_SAMPLES} \
    --lr-wsd-decay-style minus_sqrt \
    --override-opt_param-scheduler \
    --lr ${LR} \
    --min-lr ${MIN_LR} \
    --weight-decay 0.0 \
    --clip-grad 1.0 \
    --lr-decay-style cosine \
    --log-interval 1 \
    --eval-iters 4 \
    --eval-interval 250 \
    --use-persistent-ckpt-worker \
    --ckpt-format torch_dist \
    --dist-ckpt-strictness ignore_all \
    --ckpt-fully-parallel-save \
    --ckpt-fully-parallel-load \
    --save-interval 100 \
    --load ${LOAD_CHECKPOINT_DIR} --finetune \
    --save ${CHECKPOINT_DIR} \
    --save-retain-interval 100 \
    --ckpt-format torch_dist \
    --log-progress  \
    --timing-log-option minmax \
    --log-params-norm \
    --log-num-zeros-in-grad \
    --log-throughput \
    --log-energy \
    --logging-level 30 \
    --log-memory-interval 100 \
    --bf16 \
    --adam-beta1 0.9 \
    --adam-beta2 0.98 \
    --use-distributed-optimizer \
    --ddp-num-buckets 8 \
    --ddp-pad-buckets-for-high-nccl-busbw \
    --no-create-attention-mask-in-dataloader \
    --num-workers 1 \
    --log-straggler \
    --disable-straggler-on-startup \
    --straggler-minmax-count 16 \
    --check-weight-hash-across-dp-replicas-interval 20000 \
    --tensorboard-dir ${TENSORBOARD_DIR} \
    --export-default-te-spec \
    "

# Tokenizer: SFT flow (default) vs. TikToken flow. BLEND_NAME selects between
# them when working with the nano-v3 SFT blend; other blends use TikToken.
if [ "$BLEND_NAME" == "does_mask_100r_0p" ]; then
    options="${options} \
    --sft \
    --sft-tokenizer-prompt-format identity \
    --tokenizer-type SFTTokenizer \
    --tokenizer-model ${TOKENIZER_MODEL}"
else
    options="${options} \
    --tokenizer-type TikTokenizer \
    --tiktoken-pattern v2 \
    --tokenizer-model ${TOKENIZER_MODEL}"
fi

KD_CONFIG=${KD_CONFIG:-"examples/flextron/kd_config.yaml"}

distill_options=" \
    --export-te-mcore-model \
    --export-kd-teacher-model-config=${REPO_DIR}/examples/flextron/teacher_model_config_12b.yaml \
    --export-kd-cfg ${REPO_DIR}/${KD_CONFIG} \
    --export-kd-teacher-load ${TEACHER} \
    --export-te-mcore-model \
    --export-model-type MambaModel"

flex_options=" \
    --flextron \
    --enable-router \
    --binary-mask \
    --soft-mask \
    --hard-sample-th 0.996 \
    --router-beta 1.0 \
    --original-model-sample-prob ${ORIGINAL_MODEL_SAMPLE_PROB} \
    --tau-init ${TAU_INIT} \
    --tau-decay ${TAU_DECAY} \
    --loss-alpha ${LOSS_ALPHA} \
    --lr-mult-router ${LR_MULT_ROUTER} \
    --router-gbs ${ROUTER_GBS} \
    --router-inter-dim 256 \
    --budget-list ${BUDGET} \
    --budget-probs ${SAMPLE_PROBS} \
    --budget-type param \
    --mlp-per-list 1.0 0.71429 0.51725 \
    --emb-per-list 1.0 0.71429 0.51725 \
    --moe-expert-per-list 1.0 0.75 0.5 \
    --mamba-per-list 1.0 0.75 0.5 \
    --head-per-list 1.0 0.75 0.5 \
    --linear-scaler-start ${LINEAR_SCALER_START} \
    --linear-scaler-end ${LINEAR_SCALER_END} \
    --slice \
    --router-std ${ROUTER_STD}"

# export WANDB_ENTITY="nvidia"
export WANDB_API_KEY="wandbapikey"
WANDB_PROJECT="Flextron_training"
WANDB_NAME="flextron_training_${DATETIME}"

# Enable W&B logging if API key is provided
if [ -n "${WANDB_API_KEY}" ]; then
    options="${options} \
    --wandb-project ${WANDB_PROJECT} \
    --wandb-exp-name ${WANDB_NAME} \
    --wandb-save-dir ${CHECKPOINT_DIR}/wandb"
fi


torchrun --nproc-per-node=8 ${REPO_DIR}/pretrain_mamba_flex.py ${options} ${distill_options} ${flex_options}


# for i in {1..1}; do sbatch -p batch --account=coreai_dlalgo_genai --job-name=nm6_flex_reasoning --time=1:00:00 --nodes=64 3B_hybrid_moe_flex_reasoning.sh ; done;











