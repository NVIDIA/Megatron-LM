#!/bin/bash
#
# 350M ablation leaderboard - rank 1 (new).
# Optimizer: Aurora (Tilde Research, 2026), leverage-uniform polar; matrix LR 1e-2,
#            scalar LR 1.5e-3, spectral scale, Nesterov, momentum 0.95, polar bf16,
#            pp_iterations=2, pp_beta=0.5, 12-step simple-quintic Newton-Schulz.
# Architecture diff vs 09-aurora-qkn: adds --exclusive-self-attention (XSA,
#            arXiv 2603.09078). After core attention, subtracts each head's
#            output along its L2-normalized value direction. GQA-aware
#            implementation: v_hat is broadcast across query heads of each KV
#            group (no V materialization), wrapped in @torch.compile so peak
#            memory and bandwidth match the baseline. Zero parameters.
# Final loss: 2.167   Min loss: 2.009   (vs old leader 2.185 / 2.025)
# Throughput: ~234.8 TFLOP/s/GPU (in-session baseline ~227); the GQA-aware
#             rewrite + torch.compile reverses the 9% tax of the naive XSA path.
# W&B: https://wandb.ai/ischlag/megatron-lm-research-baseline/runs/mdmnb7mc
# Executed at git sha: 435b3cf2d
#
# Reference: https://blog.tilderesearch.com/blog/aurora ; XSA: arXiv 2603.09078
#
# To reproduce bitwise:
#   git checkout 435b3cf2d
#   sbatch _research/leaderboards/350m-ablation/runs/12-aurora-qkn-xsa.sbatch
#
#SBATCH --job-name=350m-abl-aurora-qkn-xsa
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

# REPO_DIR follows WORKDIR so a cluster-side sidecar clone (one per local
# worktree) keeps caches, packages, TB, and wandb out of the canonical path.
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
# XSA materializes [sq, b, np, hn] (4x V via repeat_interleave) per layer,
# adding ~1 GB activation memory * 24 layers. expandable_segments avoids the
# allocator fragmentation cliff that OOMs the cross-entropy logits buffer.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TRITON_CACHE_DIR=$REPO_DIR/cache/triton
export TORCHINDUCTOR_CACHE_DIR=$REPO_DIR/cache/inductor
export TORCHINDUCTOR_USE_STATIC_CUDA_LAUNCHER=0
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export TMPDIR=$TMP_CACHE_DIR

mkdir -p $DATASET_CACHE_DIR $TRITON_CACHE_DIR $TORCHINDUCTOR_CACHE_DIR $TMP_CACHE_DIR $PACKAGE_DIR _research/results/runs _research/results/performance

MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n1)
MASTER_PORT=25678

PROJECT=${WANDB_PROJECT:-megatron-lm-research-baseline}
EXP_NAME=350m-abl-aurora-qkn-xsa-$(git rev-parse --short HEAD)

export APERTUS_LOG_DIR=$WORKDIR/_research/results/performance
export APERTUS_FEATURE=transformer-pp-aurora
export APERTUS_TRACK=350m-ablation-leaderboard
export APERTUS_RUN_NAME=$EXP_NAME-$SLURM_JOB_ID
export APERTUS_LOG_ACT_STATS=1
export APERTUS_LOG_TOP1_ACC=1
export APERTUS_LOG_ROW_CV=1
export APERTUS_LOG_NEURON_STATS=1
export APERTUS_LOG_NEURON_INTERVAL=50
export APERTUS_LOG_PER_LAYER_GRADS=1
export APERTUS_LOG_LOSS_SPIKES=1

MEGATRON_ARGS=(
    # Architecture: 350M Transformer++ (24L, 1024H, 2560F, 16h, 4kv GQA)
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
    --normalization RMSNorm
    --swiglu
    --qk-layernorm
    --exclusive-self-attention
    --untie-embeddings-and-output-weights
    --attention-backend auto
    --seq-length 4096
    # Training: 1B tokens, GBS=128 (524K tokens/step), MBS=16 (DP=4)
    --micro-batch-size 16
    --global-batch-size 128
    --train-samples 244141
    --exit-duration-in-mins 55
    --log-interval 1
    --eval-interval 244141
    --eval-iters 0
    --optimizer aurora
    --dataloader-type single
    --manual-gc
    --manual-gc-interval 50
    --cross-entropy-loss-fusion
    --disable-bias-linear
    --no-check-for-nan-in-loss-and-grad
    # Regularization
    --attention-dropout 0.0
    --hidden-dropout 0.0
    --weight-decay 0.1
    --clip-grad 1.0
    --adam-beta1 0.9
    --adam-beta2 0.95
    # Aurora recipe
    --muon-scale-mode spectral
    --muon-nesterov
    --muon-momentum 0.95
    --muon-scalar-lr 1.5e-3
    --muon-scalar-weight-decay 0.0
    --polar-precision bf16
    --aurora-pp-iterations 2
    --aurora-pp-beta 0.5
    --aurora-num-ns-steps 12
    # WSD schedule, 200-step warmup, last 20% decay
    --lr 1e-2
    --min-lr 1e-3
    --lr-decay-style WSD
    --lr-warmup-samples 25600
    --lr-wsd-decay-style minus_sqrt
    --lr-wsd-decay-samples 48828
    # Init / precision
    --seed 42
    --init-method-std 0.02
    --bf16
    # Distributed: TP=1, PP=1, DP=4. --overlap-param-gather omitted (corrupts
    # Newton-Schulz; same caveat as Muon).
    --tensor-model-parallel-size 1
    --pipeline-model-parallel-size 1
    --context-parallel-size 1
    --sequence-parallel
    --use-distributed-optimizer
    --overlap-grad-reduce
    # Logging
    --log-throughput
    --log-progress
    --log-memory-to-tensorboard
    --log-params-norm
    --tensorboard-dir $WORKDIR/_research/results/runs/tb-$SLURM_JOB_ID
    --tensorboard-log-interval 1
    --log-timers-to-tensorboard
    # Tokenizer / data
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
    cd $WORKDIR"

TRAINING_CMD="bash $WORKDIR/_research/launch/install_python_deps.sh $PACKAGE_DIR && python3 $WORKDIR/pretrain_gpt.py $MEGATRON_ARGS_STR"

echo "CMD: $TRAINING_CMD"
export APERTUS_PHASE_PRE_SRUN=$(date +%s.%N)
srun -lu --mpi=pmix --network=disable_rdzv_get --environment=$WORKDIR/_research/launch/alps3.toml \
    --cpus-per-task $SLURM_CPUS_PER_TASK --wait 120 \
    bash -c "export APERTUS_PHASE_CONTAINER_STARTED=\$(date +%s.%N); $ENV_SETUP; $TRAINING_CMD"

echo "END TIME: $(date)"
