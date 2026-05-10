#!/bin/bash
#
# 350M-active MoE ablation - 64 experts, top-2 (8x total params vs 8E).
#
# Active params match dense aurora-qkn (~355M); total ~6.2B.
#   Per-layer FFN active = topk * 3 * hidden * moe_ffn = 2 * 3 * 1024 * 1280
#                        = 7.86M  (dense FFN: 3 * 1024 * 2560 = 7.86M)
#   Per-layer FFN total  = 64 * 3 * 1024 * 1280 = 252M  (32x dense)
# Optimizer: Aurora on 2D matrices (attention, embeds, output, router);
# expert weights are 3D under grouped-GEMM, so they fall through Aurora's
# `len(shape) != 2` predicate to AdamW. This makes the comparison vs
# dense aurora-qkn a clean test of "does sparsity buy you anything on this
# leaderboard" rather than an Aurora-on-experts test.
# All other arch knobs (RMSNorm, SwiGLU, RoPE, GQA 16h/4kv, qk-layernorm)
# stay identical to aurora-qkn.
#
# Reference: Mixtral 8x7B (Jiang et al., 2024); MoEUT (Csordas et al., 2024).
#
#SBATCH --job-name=350m-abl-aurora-qkn-moe-64e
#SBATCH --time=02:30:00
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

mkdir -p $DATASET_CACHE_DIR $TRITON_CACHE_DIR $TORCHINDUCTOR_CACHE_DIR $TMP_CACHE_DIR $PACKAGE_DIR _research/results/runs _research/results/performance

MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n1)
MASTER_PORT=25678

PROJECT=${WANDB_PROJECT:-megatron-lm-research-baseline}
EXP_NAME=350m-abl-aurora-qkn-moe-64e-$(git rev-parse --short HEAD)

export APERTUS_LOG_DIR=$WORKDIR/_research/results/performance
export APERTUS_FEATURE=transformer-pp-aurora-moe-64e
export APERTUS_TRACK=350m-ablation-aurora-qkn-moe-64e
export APERTUS_RUN_NAME=$EXP_NAME-$SLURM_JOB_ID
export APERTUS_LOG_ACT_STATS=0
export APERTUS_LOG_TOP1_ACC=0
export APERTUS_LOG_ROW_CV=1
export APERTUS_LOG_NEURON_STATS=0
export APERTUS_LOG_NEURON_INTERVAL=50

MEGATRON_ARGS=(
    # Architecture: 350M-active Transformer++ MoE (24L, 1024H, 16h, 4kv GQA)
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
    --untie-embeddings-and-output-weights
    --attention-backend auto
    --seq-length 4096
    # MoE: 64 experts, top-2, expert FFN 1280 (active matches dense, 8x total vs 8E)
    --num-experts 64
    --moe-router-topk 2
    --moe-ffn-hidden-size 1280
    --moe-router-load-balancing-type aux_loss
    --moe-aux-loss-coeff 1e-2
    --moe-z-loss-coeff 1e-3
    --moe-grouped-gemm
    --moe-permute-fusion
    --moe-token-dispatcher-type alltoall
    # Training: 1B tokens, GBS=128 (524K tokens/step), MBS=4 (DP=4, 8 grad-accum
    # steps). 64 experts replicate ~12 GB bf16 weights + ~12 GB grads on each
    # GPU on top of the MoE permute/dispatch activations; MBS=8 OOMs at the
    # fused CE step (96GB GH200). Halving MBS again here.
    --micro-batch-size 4
    --global-batch-size 128
    --train-samples 244141
    --exit-duration-in-mins 140
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
    # Aurora recipe (applies to 2D matrices; 3D expert weights fall through to AdamW)
    --muon-scale-mode spectral
    --muon-nesterov
    --muon-momentum 0.95
    --muon-scalar-lr 1.5e-3
    --muon-scalar-weight-decay 0.0
    --polar-precision bf16
    --aurora-pp-iterations 2
    --aurora-pp-beta 0.5
    --aurora-num-ns-steps 12
    # WSD schedule, 200-step warmup, last 20% decay (same as aurora-qkn)
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
    # Distributed: TP=1, PP=1, EP=1, DP=4. EP=1 fits tightly (64 experts
    # * 31.5M = 2GB/layer, 24L = ~48GB on a 96GB GH200 for bf16 weights;
    # bf16 grads add another 48 GB on the .grad fp16/bf16 reduce path —
    # may need EP>1 if OOM). --overlap-param-gather
    # omitted (corrupts Newton-Schulz; same caveat as Muon).
    --tensor-model-parallel-size 1
    --pipeline-model-parallel-size 1
    --expert-model-parallel-size 1
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
