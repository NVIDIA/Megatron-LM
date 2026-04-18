#!/bin/bash
#
# Transformer++ 350M ablation on 1 GH200 node (4 GPUs).
# 24L / 1024H / 2560F / 16h / 4kv (GQA).
# SwiGLU, RMSNorm, RoPE, AdamW, WSD schedule, bf16.
# 1B tokens on ClimbMix, ~1908 steps at 524K tokens/step (~30 min).
#
#SBATCH --account=infra01
#SBATCH --job-name=transformer-pp-350m-ablation
#SBATCH --time=01:00:00
#SBATCH --output=/iopsstor/scratch/cscs/%u/megatron-lm-research-baseline/_research/results/runs/%x-%j.log
#SBATCH --error=/iopsstor/scratch/cscs/%u/megatron-lm-research-baseline/_research/results/runs/%x-%j.log
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=72
#SBATCH --mem=460000
#SBATCH --no-requeue
#SBATCH --reservation=SD-69241-apertus-1-5

set -euo pipefail
echo "START TIME: $(date)"
export APERTUS_PHASE_SBATCH_START=$(date +%s.%N)

REPO_DIR=/iopsstor/scratch/cscs/$USER/megatron-lm-research-baseline
WORKDIR=${SLURM_SUBMIT_DIR:-$REPO_DIR}
cd $WORKDIR

PACKAGE_DIR=$REPO_DIR/_research/packages
DATASET_CACHE_DIR=$REPO_DIR/cache/dataset
TMP_CACHE_DIR=$REPO_DIR/cache/tmp

export PYTHONPATH=$WORKDIR:$PACKAGE_DIR:$WORKDIR${PYTHONPATH:+:$PYTHONPATH}
export CUDA_DEVICE_MAX_CONNECTIONS=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_CPP_LOG_LEVEL=WARNING
export TRITON_CACHE_DIR=$REPO_DIR/cache/triton
export TORCHINDUCTOR_CACHE_DIR=$REPO_DIR/cache/inductor
export TORCHINDUCTOR_USE_STATIC_CUDA_LAUNCHER=0
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export TMPDIR=$TMP_CACHE_DIR

mkdir -p $DATASET_CACHE_DIR $TRITON_CACHE_DIR $TORCHINDUCTOR_CACHE_DIR $TMP_CACHE_DIR $PACKAGE_DIR _research/results/runs _research/results/performance

MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n1)
MASTER_PORT=25678

NUM_LAYERS=24
HIDDEN=1024
FFN_HIDDEN=2560
NUM_HEADS=16
NUM_KV_HEADS=4

MBS=${SWEEP_MBS:-16}
GBS=${SWEEP_GBS:-128}
SEQ_LEN=4096
TRAIN_SAMPLES=244141    # 1B tokens / 4096 seq_len
LR=${SWEEP_LR:-3e-4}
MIN_LR=$(python3 -c "print(f'{float(\"$LR\")*0.1:.1e}')")

TP=1
PP=1
CP=1

OPTIMIZER=${SWEEP_OPTIMIZER:-adam}
MUON_SCALE_MODE=${SWEEP_MUON_SCALE_MODE:-spectral}

PROJECT=megatron-lm-research-baseline
OPT_TAG=$OPTIMIZER
if [ "$OPTIMIZER" = "adam" ]; then
    OPT_TAG=adamw   # Megatron's "adam" is decoupled WD = AdamW
fi
if [ "$OPTIMIZER" = "adaptive_muon" ] && [ -n "${SWEEP_ADAPTIVE_MUON_M2:-}" ]; then
    OPT_TAG=${SWEEP_ADAPTIVE_MUON_M2}
fi
FLAG_TAG=${SWEEP_FLAG_TAG:-}
EXP_NAME=transformer-pp-350m-abl-${OPT_TAG}-lr${LR}-gbs${GBS}${FLAG_TAG:+-$FLAG_TAG}-$(git rev-parse --short HEAD)

export APERTUS_LOG_DIR=$WORKDIR/_research/results/performance
export APERTUS_FEATURE=transformer-pp
export APERTUS_TRACK=transformer-pp-350m-ablation
export APERTUS_RUN_NAME=$EXP_NAME-$SLURM_JOB_ID
export APERTUS_LOG_ACT_STATS=0
export APERTUS_LOG_TOP1_ACC=0

TRANSFORMER_ENGINE_ARGS=(
    --transformer-impl transformer_engine
    --main-grads-dtype fp32
)

NETWORK_SIZE_ARGS=(
    --num-layers $NUM_LAYERS
    --hidden-size $HIDDEN
    --ffn-hidden-size $FFN_HIDDEN
    --num-attention-heads $NUM_HEADS
    --group-query-attention
    --num-query-groups $NUM_KV_HEADS
    --max-position-embeddings $SEQ_LEN
    --position-embedding-type rope
    --rotary-base 500000
    --make-vocab-size-divisible-by 128
    --normalization RMSNorm
    --swiglu
    --untie-embeddings-and-output-weights
    --attention-backend auto
    --seq-length $SEQ_LEN
)

TRAINING_ARGS=(
    --micro-batch-size $MBS
    --global-batch-size $GBS
    --train-samples ${SWEEP_TRAIN_SAMPLES:-$TRAIN_SAMPLES}
    --exit-duration-in-mins 55
    --log-interval 1
    --eval-interval $TRAIN_SAMPLES
    --eval-iters 0
    --optimizer $OPTIMIZER
    --dataloader-type single
    --manual-gc
    --manual-gc-interval 50
    --cross-entropy-loss-fusion
    --disable-bias-linear
    --no-check-for-nan-in-loss-and-grad
)

REGULARIZATION_ARGS=(
    --attention-dropout 0.0
    --hidden-dropout 0.0
    --weight-decay 0.1
    --clip-grad 1.0
    --adam-beta1 0.9
    --adam-beta2 0.95
)

# Muon-specific args (ignored by adam/sgd).
MUON_ARGS=()
if [[ "$OPTIMIZER" == *muon* ]]; then
    MUON_ARGS=(
        --muon-scale-mode $MUON_SCALE_MODE
        --muon-nesterov
        --muon-momentum 0.95
    )
    if [ "${SWEEP_MUON_NO_SPLIT_QKV:-0}" = "1" ]; then
        MUON_ARGS+=(--muon-no-split-qkv)
    fi
    if [ -n "${SWEEP_MUON_SCALAR_LR:-}" ]; then
        MUON_ARGS+=(--muon-scalar-lr $SWEEP_MUON_SCALAR_LR)
    fi
    if [ -n "${SWEEP_MUON_SCALAR_WD:-}" ]; then
        MUON_ARGS+=(--muon-scalar-weight-decay $SWEEP_MUON_SCALAR_WD)
    fi
    if [ "$OPTIMIZER" = "adaptive_muon" ] && [ -n "${SWEEP_ADAPTIVE_MUON_M2:-}" ]; then
        MUON_ARGS+=(--adaptive-muon-moment2-method $SWEEP_ADAPTIVE_MUON_M2)
    fi
fi

LEARNING_RATE_ARGS=(
    --lr $LR
    --min-lr $MIN_LR
    --lr-decay-style WSD
    --lr-warmup-samples 25600
    --lr-wsd-decay-style minus_sqrt
    --lr-wsd-decay-samples 48828     # last 20% of 244141
)

INITIALIZATION_ARGS=(
    --seed 42
    --init-method-std 0.02
)

MIXED_PRECISION_ARGS=(
    --bf16
)

DISTRIBUTED_ARGS=(
    --tensor-model-parallel-size $TP
    --pipeline-model-parallel-size $PP
    --context-parallel-size $CP
    --sequence-parallel
)
if [ "${SWEEP_DISTOPT:-1}" = "1" ]; then
    DISTRIBUTED_ARGS+=(--use-distributed-optimizer)
fi
if [ "${SWEEP_OVERLAP_GRAD:-1}" = "1" ]; then
    DISTRIBUTED_ARGS+=(--overlap-grad-reduce)
fi
# --overlap-param-gather breaks Muon's Newton-Schulz (per-bucket async
# gather feeds NS5 a cross-shard-inconsistent matrix, see README).
# Default off for muon/adaptive_muon, on otherwise.
if [ -z "${SWEEP_OVERLAP_PARAM:-}" ]; then
    if [[ "$OPTIMIZER" == *muon* ]]; then
        SWEEP_OVERLAP_PARAM=0
    else
        SWEEP_OVERLAP_PARAM=1
    fi
fi
if [ "$SWEEP_OVERLAP_PARAM" = "1" ]; then
    DISTRIBUTED_ARGS+=(--overlap-param-gather)
fi

LOGGING_ARGS=(
    --log-throughput
    --log-progress
    --log-memory-to-tensorboard
    --log-params-norm
    --tensorboard-dir $WORKDIR/_research/results/runs/tb-$SLURM_JOB_ID
    --tensorboard-log-interval 1
    --log-timers-to-tensorboard
)

TOKENIZER_ARGS=(
    --tokenizer-type GPT2BPETokenizer
    --vocab-file $WORKDIR/_research/data/gpt2-vocab.json
    --merge-file $WORKDIR/_research/data/gpt2-merges.txt
)

if [ "${SWEEP_MOCK:-0}" = "1" ]; then
  DATA_ARGS=(
      --mock-data
      --num-workers 2
  )
else
  DATA_ARGS=(
      --data-path /capstor/store/cscs/swissai/infra01/datasets/nvidia/Nemotron-ClimbMix/climbmix_small_megatron/climbmix_small
      --data-cache-path $DATASET_CACHE_DIR
      --split 99,1,0
      --num-workers 4
  )
fi

MEGATRON_ARGS="\
    ${TRANSFORMER_ENGINE_ARGS[@]} \
    ${NETWORK_SIZE_ARGS[@]} \
    ${TRAINING_ARGS[@]} \
    ${REGULARIZATION_ARGS[@]} \
    ${MUON_ARGS[@]} \
    ${LEARNING_RATE_ARGS[@]} \
    ${INITIALIZATION_ARGS[@]} \
    ${MIXED_PRECISION_ARGS[@]} \
    ${DISTRIBUTED_ARGS[@]} \
    ${LOGGING_ARGS[@]} \
    ${TOKENIZER_ARGS[@]} \
    ${DATA_ARGS[@]}"

if [ -n "${WANDB_API_KEY:-}" ]; then
    export WANDB_PROJECT=$PROJECT
    export WANDB_NAME=$EXP_NAME-$SLURM_JOB_ID
    MEGATRON_ARGS="$MEGATRON_ARGS --wandb-project $PROJECT --wandb-exp-name $EXP_NAME-$SLURM_JOB_ID --wandb-save-dir $WORKDIR/_research/results/runs"
else
    export WANDB_MODE=disabled
fi

ENV_SETUP="export MASTER_ADDR=$MASTER_ADDR; \
    export MASTER_PORT=$MASTER_PORT; \
    export RANK=\$SLURM_PROCID; \
    export WORLD_SIZE=\$SLURM_NTASKS; \
    export LOCAL_RANK=\$SLURM_LOCALID; \
    export LOCAL_WORLD_SIZE=$((SLURM_NTASKS/SLURM_NNODES)); \
    export PYTHONPATH=$WORKDIR:$PACKAGE_DIR:$WORKDIR\${PYTHONPATH:+:\$PYTHONPATH}; \
    cd $WORKDIR"

TRAINING_CMD="bash $WORKDIR/_research/launch/install_python_deps.sh $PACKAGE_DIR && python3 $WORKDIR/pretrain_gpt.py $MEGATRON_ARGS"

echo "CMD: $TRAINING_CMD"
export APERTUS_PHASE_PRE_SRUN=$(date +%s.%N)
srun -lu --mpi=pmix --network=disable_rdzv_get --environment=alps3 \
    --cpus-per-task $SLURM_CPUS_PER_TASK --wait 120 \
    bash -c "export APERTUS_PHASE_CONTAINER_STARTED=\$(date +%s.%N); $ENV_SETUP; $TRAINING_CMD"

echo "END TIME: $(date)"
