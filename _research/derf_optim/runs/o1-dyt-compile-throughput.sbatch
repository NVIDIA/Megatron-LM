#!/bin/bash
#
# Option 1 throughput test: torch.compile fused (DyT + nn.Linear).
# Runs 200 iters of the Aurora @ 1e-2 recipe with --normalization DyT
# and APERTUS_DERF_OPTIM=compile so the spec swaps in our fused composite.
#
# Reference (unfused Derf): ~217 TFLOP/s/GPU (job 2075996, mid-run).
# Reference (RMSNorm baseline): ~310 TFLOP/s/GPU (Aurora rank 1).
#
#SBATCH --job-name=o1-dyt-compile
#SBATCH --time=00:20:00
#SBATCH --output=/iopsstor/scratch/cscs/%u/megatron-lm-research-baseline/_research/results/runs/%x-%j.log
#SBATCH --error=/iopsstor/scratch/cscs/%u/megatron-lm-research-baseline/_research/results/runs/%x-%j.log
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=72
#SBATCH --mem=460000
#SBATCH --no-requeue

set -euo pipefail
: "${MEGATRON_DATA_PATH:?please set MEGATRON_DATA_PATH}"
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

# Option 1 switch.
export APERTUS_DERF_OPTIM=compile
export APERTUS_DERF_COMPILE=1

mkdir -p $DATASET_CACHE_DIR $TRITON_CACHE_DIR $TORCHINDUCTOR_CACHE_DIR $TMP_CACHE_DIR $PACKAGE_DIR _research/results/runs _research/results/performance

MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n1)
MASTER_PORT=25678

PROJECT=${WANDB_PROJECT:-megatron-lm-research-baseline}
EXP_NAME=o1-dyt-compile-$(git rev-parse --short HEAD)

export APERTUS_LOG_DIR=$WORKDIR/_research/results/performance
export APERTUS_FEATURE=dyt-optim-option1-compile
export APERTUS_TRACK=derf-throughput-bench
export APERTUS_RUN_NAME=$EXP_NAME-$SLURM_JOB_ID
export APERTUS_LOG_ACT_STATS=0
export APERTUS_LOG_TOP1_ACC=0
export APERTUS_LOG_ROW_CV=0
export APERTUS_LOG_NEURON_STATS=0

MEGATRON_ARGS=(
    --transformer-impl transformer_engine
    --main-grads-dtype fp32
    --num-layers 24
    --hidden-size 1024
    --ffn-hidden-size 2560
    --num-attention-heads 16
    --group-query-attention
    --num-query-groups 4
    --max-position-embeddings 4096
    --position-embedding-type rope
    --rotary-base 500000
    --make-vocab-size-divisible-by 128
    --normalization DyT
    --swiglu
    --untie-embeddings-and-output-weights
    --attention-backend auto
    --seq-length 4096
    --micro-batch-size 16
    --global-batch-size 128
    --train-iters 200
    --log-interval 1
    --eval-interval 200
    --eval-iters 0
    --optimizer aurora
    --dataloader-type single
    --manual-gc
    --manual-gc-interval 50
    --cross-entropy-loss-fusion
    --disable-bias-linear
    --no-check-for-nan-in-loss-and-grad
    --attention-dropout 0.0
    --hidden-dropout 0.0
    --weight-decay 0.1
    --clip-grad 1.0
    --adam-beta1 0.9
    --adam-beta2 0.95
    --muon-scale-mode spectral
    --muon-nesterov
    --muon-momentum 0.95
    --muon-scalar-lr 1.5e-3
    --muon-scalar-weight-decay 0.0
    --polar-precision bf16
    --aurora-pp-iterations 2
    --aurora-pp-beta 0.5
    --aurora-num-ns-steps 12
    --lr 1e-2
    --min-lr 1e-3
    --lr-decay-style constant
    --lr-warmup-iters 10
    --seed 42
    --init-method-std 0.02
    --bf16
    --tensor-model-parallel-size 1
    --pipeline-model-parallel-size 1
    --context-parallel-size 1
    --sequence-parallel
    --use-distributed-optimizer
    --overlap-grad-reduce
    --log-throughput
    --log-progress
    --log-memory-to-tensorboard
    --log-params-norm
    --tensorboard-dir $WORKDIR/_research/results/runs/tb-$SLURM_JOB_ID
    --tensorboard-log-interval 1
    --log-timers-to-tensorboard
    --tokenizer-type GPT2BPETokenizer
    --vocab-file $WORKDIR/_research/data/gpt2-vocab.json
    --merge-file $WORKDIR/_research/data/gpt2-merges.txt
    --data-path $MEGATRON_DATA_PATH
    --data-cache-path $DATASET_CACHE_DIR
    --split 99,1,0
    --num-workers 4
)

MEGATRON_ARGS_STR="${MEGATRON_ARGS[@]}"
if [ -n "${WANDB_API_KEY:-}" ]; then
    export WANDB_PROJECT=$PROJECT
    export WANDB_NAME=$EXP_NAME-$SLURM_JOB_ID
    MEGATRON_ARGS_STR="$MEGATRON_ARGS_STR --wandb-project $PROJECT --wandb-exp-name $EXP_NAME-$SLURM_JOB_ID --wandb-save-dir $WORKDIR/_research/results/runs"
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
    export PYTORCH_CUDA_ALLOC_CONF=$PYTORCH_CUDA_ALLOC_CONF; \
    export APERTUS_DERF_OPTIM=$APERTUS_DERF_OPTIM; \
    export APERTUS_DERF_COMPILE=$APERTUS_DERF_COMPILE; \
    cd $WORKDIR"

TRAINING_CMD="bash $WORKDIR/_research/launch/install_python_deps.sh $PACKAGE_DIR && python3 $WORKDIR/pretrain_gpt.py $MEGATRON_ARGS_STR"

echo "CMD: $TRAINING_CMD"
export APERTUS_PHASE_PRE_SRUN=$(date +%s.%N)
srun -lu --mpi=pmix --network=disable_rdzv_get --environment=$WORKDIR/_research/launch/alps3.toml \
    --cpus-per-task $SLURM_CPUS_PER_TASK --wait 120 \
    bash -c "export APERTUS_PHASE_CONTAINER_STARTED=\$(date +%s.%N); $ENV_SETUP; $TRAINING_CMD"

echo "END TIME: $(date)"
