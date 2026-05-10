#!/bin/bash
#
# Transformer++ 350M ablation, Mamba 3 hybrid baseline, 1 GH200 node (4 GPUs).
# 24L / 1024H / 2560F. SwiGLU MLP. Pattern: M-M-M-*- x 3 (3:1 mamba:attn ratio
# in blocks, matching our DN/GDN/KDA/Mamba2 setup). NorMuon, WSD, bf16, 1B
# tokens on ClimbMix.
#
# Variant: Mamba 3 (Lahoti, Li, Chen, Wang, Bick, Kolter, Dao, Gu, ICLR 2026)
# via the upstream state-spaces/mamba mamba_ssm.modules.Mamba3 layer, wrapped
# in megatron/core/ssm/mamba3_mixer.py. Spec dispatch via
# megatron.core.models.hybrid.mamba3_layer_specs.mamba3_stack_spec
# (forks hybrid_stack_spec; only the mamba_layer slot is swapped).
# torch.compile is on by default in the wrapper to speed up the
# PyTorch-only fallback path on ARM64 GH200.
#
#SBATCH --job-name=transformer-pp-350m-ablation-muon-mamba3-lr1.5e-4-clip0.5
#SBATCH --time=01:00:00
#SBATCH --output=/iopsstor/scratch/cscs/%u/megatron-lm-research-baseline/_research/results/runs/%x-%j.log
#SBATCH --error=/iopsstor/scratch/cscs/%u/megatron-lm-research-baseline/_research/results/runs/%x-%j.log
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=72
#SBATCH --mem=460000
#SBATCH --no-requeue

set -euo pipefail
: "${MEGATRON_DATA_PATH:?please set MEGATRON_DATA_PATH to the Megatron-binary dataset prefix (without .bin/.idx), e.g. /path/to/climbmix_small}"
echo "START TIME: $(date)"
export APERTUS_PHASE_SBATCH_START=$(date +%s.%N)

WORKDIR=${SLURM_SUBMIT_DIR:-/iopsstor/scratch/cscs/$USER/megatron-lm-research-baseline}
REPO_DIR=$WORKDIR
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
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

mkdir -p $DATASET_CACHE_DIR $TRITON_CACHE_DIR $TORCHINDUCTOR_CACHE_DIR $TMP_CACHE_DIR $PACKAGE_DIR _research/results/runs _research/results/performance

MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n1)
MASTER_PORT=25678

NUM_LAYERS=24
HIDDEN=1024
FFN_HIDDEN=2560
NUM_HEADS=16
NUM_KV_HEADS=4

MBS=8   # reduced from 16 to fit GDN/DeltaNet state + chunkwise intermediates in 95GB GH200
GBS=128
SEQ_LEN=4096
TRAIN_SAMPLES=244141    # 1B tokens / 4096 seq_len
LR=1.5e-4
MIN_LR=1.5e-5

TP=1
PP=1
CP=1

PROJECT=${WANDB_PROJECT:-megatron-lm-research-baseline}
EXP_NAME=transformer-pp-350m-ablation-muon-gdn-$(git rev-parse --short HEAD)

export APERTUS_LOG_DIR=$WORKDIR/_research/results/performance
export APERTUS_FEATURE=deltanet-variants
export APERTUS_TRACK=transformer-pp-350m-ablation-deltanet
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

# Mamba 2 hybrid via NVIDIA's pretrain_hybrid.py + hybrid_stack_spec.
# Layer pattern (one char per layer): M=mamba, *=attention, -=MLP.
# 24 layers = "M-M-M-*-" x 3 = 12 "transformer blocks" with 9 mamba blocks
# and 3 attention blocks (3:1 mamba:attn ratio in blocks, matching our DN/GDN setup).
MAMBA_ARGS=(
    --hybrid-layer-pattern M-M-M-*-M-M-M-*-M-M-M-*-
    --spec megatron.core.models.hybrid.mamba3_layer_specs mamba3_stack_spec
    --use-mcore-models
    --no-create-attention-mask-in-dataloader
    --mamba-state-dim 128
    --mamba-head-dim 64
    --mamba-num-groups 8
)

TRAINING_ARGS=(
    --micro-batch-size $MBS
    --global-batch-size $GBS
    --train-samples $TRAIN_SAMPLES
    --exit-duration-in-mins 55
    --log-interval 1
    --eval-interval $TRAIN_SAMPLES
    --eval-iters 0
    --optimizer adaptive_muon
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
    --clip-grad 0.5
    --adam-beta1 0.9
    --adam-beta2 0.95
)

# NorMuon recipe (validated on 1B-token 350M ablation).
# --overlap-param-gather is omitted (breaks Muon, see README).
MUON_ARGS=(
    --muon-scale-mode spectral
    --muon-nesterov
    --muon-momentum 0.95
    --muon-scalar-lr 1.5e-3
    --muon-scalar-weight-decay 0.0
    --adaptive-muon-moment2-method normuon
)

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
    --use-distributed-optimizer
    --overlap-grad-reduce
)

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

DATA_ARGS=(
    --data-path $MEGATRON_DATA_PATH
    --data-cache-path $DATASET_CACHE_DIR
    --split 99,1,0
    --num-workers 4
)

MEGATRON_ARGS="\
    ${TRANSFORMER_ENGINE_ARGS[@]} \
    ${NETWORK_SIZE_ARGS[@]} \
    ${MAMBA_ARGS[@]} \
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

TRAINING_CMD="bash $WORKDIR/_research/launch/install_python_deps.sh $PACKAGE_DIR && python3 $WORKDIR/pretrain_hybrid.py $MEGATRON_ARGS"

echo "CMD: $TRAINING_CMD"
export APERTUS_PHASE_PRE_SRUN=$(date +%s.%N)
srun -lu --mpi=pmix --network=disable_rdzv_get --environment=$WORKDIR/_research/launch/alps3.toml \
    --cpus-per-task $SLURM_CPUS_PER_TASK --wait 120 \
    bash -c "export APERTUS_PHASE_CONTAINER_STARTED=\$(date +%s.%N); $ENV_SETUP; $TRAINING_CMD"

echo "END TIME: $(date)"
