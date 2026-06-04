#!/bin/bash
#
# Nano-3.5 SFT — Ultra-final blend at 288K seq, full 1-epoch run.
#
# Hyperparams mirror Wen Liang's ultra-final script
#   /lustre/fsw/portfolios/llmservice/users/wdai/megatron-lm-ultra/ultra-v3-sft-jan30blend-mar13-256k.sh
# (same data, same TRAIN_SAMPLES/LR/save/eval cadence) — only the model arch
# is swapped to nano-3.5.
#
# Architecture: nano-3.5 (52L / 2688h / 1856 ffn / 32 heads / 2 GQA / kv=128,
#               128 experts topk=6 scaling=2.5 / shared ffn=3712 /
#               mamba 64h x 64d / MTP=1 pattern *E / no latent-MoE)
# Parallelism:  TP=8, EP=8, CP=4, PP=1, ETP=1 — 64 GB200 nodes, DP=8.
# Seq:          294912 (288K, matches packed data; pre-LC, extrapolates past
#               nano-3.5's 256K pretraining max — same setup soumyes used in
#               ultra-v3-sft-jan30blend-mar21-256k-dcp.sh).
# Dispatch:     alltoall (EP=8 fits one NVLink domain) — no HybridEP/MNNVL.
# Base ckpt:    post-reinit iter_0000001 from nano-3.5-init-embeddings.sh.

#SBATCH -p batch
#SBATCH -q normal
#SBATCH --account=nemotron_n4_post
#SBATCH --ntasks-per-node=4
#SBATCH --nodes=64
#SBATCH --time=4:00:00
#SBATCH --exclusive
#SBATCH --gpus-per-node=4
#SBATCH --mem=0
#SBATCH --comment='{"APS": {"auto_resume_mode": "singleton_dependency"}}'
#SBATCH --dependency=singleton
#SBATCH --job-name=nano-3.5-sft-alex-512k-hermes-rerun-jun2-guyueh-mlm-vlm2-rebase

################################################################
### TransformerEngine
################################################################
export NVTE_FWD_LAYERNORM_SM_MARGIN=16
export NVTE_BWD_LAYERNORM_SM_MARGIN=16
export NVTE_CPU_OFFLOAD_V1=1
export TORCHINDUCTOR_WORKER_START=fork

################################################################
### UCX (prevents memory hook conflicts in multi-node)
################################################################
export UCX_MEM_MMAP_HOOK_MODE=none
export UCX_MEM_CUDA_HOOK_MODE=none
export UCX_MEM_MALLOC_HOOKS=none
export UCX_ERROR_SIGNALS=none

################################################################
### General
################################################################
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=16
export CUDA_DEVICE_MAX_CONNECTIONS=1
export SHELL=/bin/bash

#export HF_HOME="/lustre/fsw/portfolios/llmservice/users/soumyes/.cache/huggingface/"

NAME=${SLURM_JOB_NAME}

OUTPUT_ROOT="/lustre/fsw/portfolios/llmservice/users/guyueh/sft-runs"
MEGATRON_LM_DIR="/lustre/fsw/portfolios/llmservice/users/guyueh/mac_mirror/megatron-lm-vlm2-rebase"
IMAGE="/lustre/fsw/portfolios/llmservice/users/guyueh/container_images/megatron_lm_26.04_a6d61fb.sqsh"

BINDPCIE_SCRIPT="/lustre/fsw/portfolios/llmservice/users/soumyes/sft-runs/code/bindpcie.sh"

WANDB_PROJECT=${WANDB_PROJECT:-"nano-3.5-sft-guyueh"}
WANDB_ARGS=""
if [ -z "${WANDB_API_KEY:-}" ] && [ -f "${HOME}/.bashrc" ]; then
    source <(grep -E '^[[:space:]]*export[[:space:]]+WANDB_API_KEY=' "${HOME}/.bashrc" | tail -n 1)
fi
if [ -z "${WANDB_API_KEY:-}" ]; then
    echo "WARNING: WANDB_API_KEY is not set. Disabling W&B logging for this run."
else
    export WANDB_API_KEY
    WANDB_ARGS="--wandb-project ${WANDB_PROJECT} --wandb-exp-name ${NAME}"
fi

RUN_DIR="${OUTPUT_ROOT}"
LOGS_DIR="${RUN_DIR}/logs/${NAME}/"
CHECKPOINT_DIR="${RUN_DIR}/checkpoints/${NAME}"
DATACACHE_DIR="${RUN_DIR}/data_cache/${NAME}"
TENSORBOARD_DIR="${RUN_DIR}/tensorboard/${NAME}"
HF_CACHE_DIR="${DATACACHE_DIR}/hf_cache"
export HF_HOME="${HF_CACHE_DIR}/home"
export HF_DATASETS_CACHE="${HF_CACHE_DIR}/datasets"

mkdir -p ${LOGS_DIR}
mkdir -p ${CHECKPOINT_DIR}
mkdir -p ${DATACACHE_DIR}
mkdir -p ${TENSORBOARD_DIR}
mkdir -p ${HF_HOME}
mkdir -p ${HF_DATASETS_CACHE}

export TRITON_CACHE_DIR="/tmp/triton-cache"

################################################################
### Base checkpoint (Megatron format, pre-LC, with MTP, post-reinit)
### Output of nano-3.5-init-embeddings.sh — iter_0000001 has the
### reinit'd low-norm embeddings (latest_checkpointed_iteration.txt = 1).
################################################################
# BASE_MODEL_PATH="/lustre/fsw/portfolios/llmservice/users/soumyes/sft-runs/reinit-embeddings-ckpts/nano-3.5-init-embeddings-iter37k-pre-lc/checkpoints"
BASE_MODEL_PATH="/lustre/fsw/portfolios/llmservice/projects/llmservice_nlp_fm/nemotron6/3b_hybrid_moe_nano3p5/checkpoints/base_lc_reinit_emb/"

DATETIME=`date +'date_%y-%m-%d_time_%H-%M-%S'`
if [ -n "${SLURM_JOB_ID:-}" ] ; then
    SCRIPT_PATH=$(scontrol show job "$SLURM_JOB_ID" | awk -F= '/Command=/{print $2}')
    ENV_LOG_FILENAME=${NAME}_${SLURM_JOB_ID}_${DATETIME}.env.log
else
    SCRIPT_PATH=$(realpath "$0")
    ENV_LOG_FILENAME=${NAME}_${DATETIME}.env.log
fi

SCRIPT_DIR=$(dirname ${SCRIPT_PATH})

################################################################
### Log environment
################################################################
echo "<< START PATHS >>" |& tee -a ${LOGS_DIR}/${ENV_LOG_FILENAME}
echo "IMAGE=${IMAGE}" |& tee -a ${LOGS_DIR}/${ENV_LOG_FILENAME}
echo "BINDPCIE_SCRIPT=${BINDPCIE_SCRIPT}" |& tee -a ${LOGS_DIR}/${ENV_LOG_FILENAME}
echo "MEGATRON_LM_DIR=${MEGATRON_LM_DIR}" |& tee -a ${LOGS_DIR}/${ENV_LOG_FILENAME}
echo "BASE_MODEL_PATH=${BASE_MODEL_PATH}" |& tee -a ${LOGS_DIR}/${ENV_LOG_FILENAME}
echo "RUN_DIR=${RUN_DIR}" |& tee -a ${LOGS_DIR}/${ENV_LOG_FILENAME}
echo "LOGS_DIR=${LOGS_DIR}" |& tee -a ${LOGS_DIR}/${ENV_LOG_FILENAME}
echo "CHECKPOINT_DIR=${CHECKPOINT_DIR}" |& tee -a ${LOGS_DIR}/${ENV_LOG_FILENAME}
echo "DATACACHE_DIR=${DATACACHE_DIR}" |& tee -a ${LOGS_DIR}/${ENV_LOG_FILENAME}
echo "TENSORBOARD_DIR=${TENSORBOARD_DIR}" |& tee -a ${LOGS_DIR}/${ENV_LOG_FILENAME}
echo "HF_HOME=${HF_HOME}" |& tee -a ${LOGS_DIR}/${ENV_LOG_FILENAME}
echo "HF_DATASETS_CACHE=${HF_DATASETS_CACHE}" |& tee -a ${LOGS_DIR}/${ENV_LOG_FILENAME}
echo "SCRIPT_DIR=${SCRIPT_DIR}" |& tee -a ${LOGS_DIR}/${ENV_LOG_FILENAME}
echo "<< END PATHS >>" |& tee -a ${LOGS_DIR}/${ENV_LOG_FILENAME}
echo -e "\n\n" |& tee -a ${LOGS_DIR}/${ENV_LOG_FILENAME}

echo "<< START GIT >>" |& tee -a ${LOGS_DIR}/${ENV_LOG_FILENAME}
echo "GIT LOG" |& tee -a ${LOGS_DIR}/${ENV_LOG_FILENAME}
git -C ${MEGATRON_LM_DIR} log --oneline -1 |& tee -a ${LOGS_DIR}/${ENV_LOG_FILENAME}
echo -e "\n\n" |& tee -a ${LOGS_DIR}/${ENV_LOG_FILENAME}
echo "GIT STATUS" |& tee -a ${LOGS_DIR}/${ENV_LOG_FILENAME}
git -C ${MEGATRON_LM_DIR} status --porcelain --branch |& tee -a ${LOGS_DIR}/${ENV_LOG_FILENAME}
echo -e "\n\n" |& tee -a ${LOGS_DIR}/${ENV_LOG_FILENAME}
echo "GIT DIFF" |& tee -a ${LOGS_DIR}/${ENV_LOG_FILENAME}
git -C ${MEGATRON_LM_DIR} diff |& tee -a ${LOGS_DIR}/${ENV_LOG_FILENAME}
echo "<< END GIT >>" |& tee -a ${LOGS_DIR}/${ENV_LOG_FILENAME}
echo -e "\n\n" |& tee -a ${LOGS_DIR}/${ENV_LOG_FILENAME}

echo "<< START ENV >>" |& tee -a ${LOGS_DIR}/${ENV_LOG_FILENAME}
env |& tee -a ${LOGS_DIR}/${ENV_LOG_FILENAME}
echo "<< END ENV >>" |& tee -a ${LOGS_DIR}/${ENV_LOG_FILENAME}

################################################################
### Hyperparameters
################################################################
SEQ_LEN=524288           # 288K — matches WDai's packed data (>nano-3.5 256K max)
TRAIN_SAMPLES=877915    # 3 shards exact: 592000 (consumed by end of part1) + 285915 (part2 wc -l). Training stops at iter ~13717 = 1 clean epoch on part2.
LR_WARMUP_SAMPLES=4800
LR_DECAY_SAMPLES=612508  # UNCHANGED: cosine still ends at iter ~9645. LR=MIN_LR for part2 (constant 5e-6 across iter 9645->13717).
LOG_INTERVAL=10
SAVE_INTERVAL=250
SAVE_RETAIN_INTERVAL=500
GBS=64
LR=2e-5
MIN_LR=5e-6

################################################################
### Tokenizer + data
###  - Tokenizer: ultra-v3 SFTTokenizer dir (HF format, has chat_template).
###  - Blend: same Ultra Mar-12 packed blend (Wenliang confirmed
###    Ultra SFT blend reused for Nano-3.5; no re-ablation per Oleksii).
################################################################
TOKENIZER_MODEL_PATH="/lustre/fs1/portfolios/llmservice/projects/llmservice_modelalignment_ppo/users/adithyare/nemotron_super/tokenizer"
BLEND_PATH="/lustre/fsw/portfolios/llmservice/users/soumyes/sft-runs/blends/Ultra-SFTb2-512K-hermes20k.part0.json"

        # --calculate-per-token-loss \
OPTIONS=" \
    --async-save \
    --sft \
    --sft-tokenizer-prompt-format identity \
    --distributed-timeout-minutes 240 \
    --num-dataset-builder-threads 32 \
    --tokenizer-type SFTTokenizer \
    --tokenizer-model ${TOKENIZER_MODEL_PATH} \
        --recompute-granularity selective \
        --recompute-modules moe \
        \
        --fine-grained-activation-offloading \
        --offload-modules moe_act \
        \
        --mtp-use-repeated-layer \
        \
        --context-parallel-size 8 \
        --tensor-model-parallel-size 8 \
        --expert-model-parallel-size 8 \
        --expert-tensor-parallel-size 1 \
        --pipeline-model-parallel-size 1 \
        --hybrid-override-pattern MEMEM*EMEMEM*EMEMEM*EMEMEM*EMEMEM*EMEMEMEM*EMEMEMEME \
        --mtp-hybrid-override-pattern \"*E\" \
        \
        --pretrained-checkpoint ${BASE_MODEL_PATH} \
        --save-interval ${SAVE_INTERVAL} \
        --save-retain-interval ${SAVE_RETAIN_INTERVAL} \
        --lr $LR \
        --min-lr $MIN_LR \
        --lr-decay-style cosine \
        --train-samples ${TRAIN_SAMPLES} \
        --lr-warmup-samples ${LR_WARMUP_SAMPLES} \
        --lr-decay-samples ${LR_DECAY_SAMPLES} \
        --seq-length ${SEQ_LEN} \
        --max-position-embeddings ${SEQ_LEN} \
        --log-interval ${LOG_INTERVAL} \
        --micro-batch-size 1 \
        --global-batch-size ${GBS} \
        --overlap-grad-reduce \
        --overlap-param-gather \
        \
        --mtp-num-layers 2 \
        --mtp-loss-scaling-factor 0.1 \
        \
        --high-priority-stream-groups ep \
        --manual-gc-interval 10 \
        --ddp-num-buckets 8 \
        --manual-gc \
        \
        --moe-permute-fusion \
        --cross-entropy-loss-fusion \
        --cross-entropy-fusion-impl native \
        --use-fused-weighted-squared-relu \
        \
        --moe-token-dispatcher-type alltoall \
        --moe-shared-expert-overlap \
        --moe-router-score-function sigmoid \
        --moe-grouped-gemm \
        --num-experts 128 \
        --moe-router-topk 6 \
        --moe-aux-loss-coeff 1e-4 \
        --moe-router-topk-scaling-factor 2.5 \
        --moe-router-enable-expert-bias \
        --moe-router-dtype fp32 \
        --moe-router-load-balancing-type seq_aux_loss \
        --moe-shared-expert-intermediate-size 3712 \
        \
        --attention-backend flash \
        --num-workers 1 \
        --disable-gloo-process-groups \
        --ckpt-format torch_dist \
        --ckpt-fully-parallel-save \
        --ckpt-fully-parallel-load \
        --ckpt-assume-constant-structure \
        --dist-ckpt-save-pre-mcore-014 \
        --use-persistent-ckpt-worker \
        \
        --squared-relu \
        --no-mmap-bin-files \
        --exit-duration-in-mins 230 \
        --rerun-mode validate_results \
        --no-create-attention-mask-in-dataloader \
        \
        --sequence-parallel \
        --use-distributed-optimizer \
        --override-opt-param-scheduler \
        \
        --mamba-num-heads 64 \
        --mamba-head-dim 64 \
        --is-hybrid-model \
        --untie-embeddings-and-output-weights \
        --init-method-std 0.0173 \
        --position-embedding-type none \
        --num-layers 52 \
        --hidden-size 2688 \
        --num-attention-heads 32 \
        --group-query-attention \
        --num-query-groups 2 \
        --ffn-hidden-size 1856 \
        --kv-channels 128 \
        --save ${CHECKPOINT_DIR} \
        --load ${CHECKPOINT_DIR} \
        --per-split-data-args-path ${BLEND_PATH} \
        --data-cache-path ${DATACACHE_DIR} \
        --weight-decay 0.1 \
        --clip-grad 1.0 \
        --attention-dropout 0.0 \
        --hidden-dropout 0.0 \
        --disable-bias-linear \
        --normalization RMSNorm \
        --adam-beta1 0.9 \
        --adam-beta2 0.95 \
        --log-params-norm \
        --log-num-zeros-in-grad \
        --log-throughput \
        --log-timers-to-tensorboard \
        --log-progress \
        --log-energy \
        --log-memory-interval 200 \
        --logging-level 20 \
        --log-straggler \
        --disable-straggler-on-startup \
        --straggler-minmax-count 16 \
        --check-weight-hash-across-dp-replicas-interval 20000 \
        --ddp-pad-buckets-for-high-nccl-busbw \
        --timing-log-option minmax \
        --eval-interval 1000 \
        --eval-iters 14 \
        --bf16 \
        --use-mcore-models \
        --spec megatron.core.models.mamba.mamba_layer_specs mamba_stack_spec \
        ${WANDB_ARGS} \
        --dist-ckpt-strictness log_unexpected \
        --tensorboard-dir ${TENSORBOARD_DIR}"

RUN_CMD="python -u ${MEGATRON_LM_DIR}/pretrain_mamba.py ${OPTIONS}"

LAUNCH_CMD="${BINDPCIE_SCRIPT} --cpu=node --mem=node -- ${RUN_CMD}"

srun -l \
     --mpi=none \
     --no-container-mount-home \
     --container-image=${IMAGE} \
     --container-mounts="/lustre:/lustre" \
     --container-env=UCX_MEM_MMAP_HOOK_MODE,UCX_MEM_CUDA_HOOK_MODE,UCX_MEM_MALLOC_HOOKS,UCX_ERROR_SIGNALS,NVTE_CPU_OFFLOAD_V1,NVTE_FWD_LAYERNORM_SM_MARGIN,NVTE_BWD_LAYERNORM_SM_MARGIN,TORCHINDUCTOR_WORKER_START,PYTORCH_CUDA_ALLOC_CONF,OMP_NUM_THREADS,CUDA_DEVICE_MAX_CONNECTIONS,SLURM_LOCALID,WANDB_API_KEY,HF_HOME,HF_DATASETS_CACHE \
     --output="${LOGS_DIR}/%x_%j_${DATETIME}.log" \
     sh -c "${LAUNCH_CMD}"
