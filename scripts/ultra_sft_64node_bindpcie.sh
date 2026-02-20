#!/bin/bash
#
# ultra_sft_64node_bindpcie.sh
# ============================
# 64-node Ultra SFT run with segment=16, HybridEP/UCX env, and bindpcie (NUMA binding).
# Based on ultra-v3-sft-hsg-mainfeb5merge-mxfp4_newbase; paths and model options unchanged.
#
# --- Summary of changes (for teammates) ---
#
# 1) SBATCH
#    - Replaced --dependency=singleton with --segment=16
#      so all 64 nodes sit in the same NVLink domain.
#
# 2) Environment variables
#    - NVTE_CPU_OFFLOAD_V1=1
#    - NUM_OF_HYBRID_EP_RANKS_PER_NVLINK_DOMAIN=64, USE_MNNVL=1
#      (segment=16, 4 GPUs/node => 16*4=64)
#    - UCX (4 vars): UCX_MEM_MMAP_HOOK_MODE, UCX_MEM_CUDA_HOOK_MODE,
#                    UCX_MEM_MALLOC_HOOKS, UCX_ERROR_SIGNALS
#      to avoid memory hook conflicts in multi-node.
#
# 3) bindpcie (CPU/memory NUMA binding)
#    - BINDPCIE_SCRIPT="${MEGATRON_LM_DIR}/scripts/bindpcie.sh"
#    - Launch: bindpcie --cpu=node --mem=node -- python ... (LAUNCH_CMD)
#    - SLURM_LOCALID is passed into the container so bindpcie can use it as local rank.
#    - Requires: scripts/bindpcie.sh in megatron-lm-ultra repo, and numactl in the container.
#
# 4) srun
#    - Added --mpi=none
#    - --container-env lists the above env vars + SLURM_LOCALID
#      so they are visible inside the container.
#
# --- End of summary ---

#SBATCH -p batch
#SBATCH -q normal
#SBATCH --account=llmservice_nemotron_ultra
#SBATCH --ntasks-per-node=4
#SBATCH --nodes=64
#SBATCH --time=3:45:00
#SBATCH --exclusive
#SBATCH --gpus-per-node=4
#SBATCH --mem=0
#SBATCH --segment=16
#SBATCH --job-name=ultra-v3-sft-hsg-mainfeb5merge-mxfp4_newbase

################################################################
### TransformerEngine
################################################################
export NVTE_FWD_LAYERNORM_SM_MARGIN=16
export NVTE_BWD_LAYERNORM_SM_MARGIN=16
export NVTE_CPU_OFFLOAD_V1=1
export TORCHINDUCTOR_WORKER_START=fork

################################################################
### HybridEP / MNNVL (segment=16 => 16*4=64 ranks per NVLink domain)
### EP % NUM_OF_HYBRID_EP_RANKS_PER_NVLINK_DOMAIN == 0 for hybridep optimization.
################################################################
export NUM_OF_HYBRID_EP_RANKS_PER_NVLINK_DOMAIN=64
export USE_MNNVL=1

################################################################
### UCX (prevents memory hook conflicts in multi-node)
################################################################
export UCX_MEM_MMAP_HOOK_MODE=none
export UCX_MEM_CUDA_HOOK_MODE=none
export UCX_MEM_MALLOC_HOOKS=n
export UCX_ERROR_SIGNALS=

################################################################
### General
################################################################
export QUANTIZATION_TYPE_DEBUG=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=16

# Debug: See NCCL operations during checkpoint load
# export NCCL_DEBUG=INFO
# export TORCH_DISTRIBUTED_DEBUG=DETAIL

export HF_HOME="/lustre/fsw/portfolios/llmservice/users/adithyare/.cache/huggingface/"

NAME=${SLURM_JOB_NAME}

OUTPUT_ROOT="/lustre/fs1/portfolios/llmservice/projects/llmservice_modelalignment_ppo/users/adithyare/nemotron_ultra/sft-runs"
MEGATRON_LM_DIR="/lustre/fs1/portfolios/llmservice/projects/llmservice_modelalignment_ppo/users/adithyare/code/megatron-lm-ultra"
IMAGE="/lustre/fs1/portfolios/llmservice/projects/llmservice_modelalignment_ppo/users/adithyare/containers/pt_ultra_mamba_ssmv230_23jan28.sqsh"

# bindpcie: CPU/mem binding (--cpu=node, --mem=node). Requires numactl in container.
BINDPCIE_SCRIPT="${MEGATRON_LM_DIR}/scripts/bindpcie.sh"

# WANDB_API_KEY: loaded from ~/.bashrc (do NOT hardcode here)
WANDB_PROJECT="ultra-v3-sft-hsg"

RUN_DIR="${OUTPUT_ROOT}"
LOGS_DIR="${RUN_DIR}/${NAME}/logs/"
CHECKPOINT_DIR="${RUN_DIR}/${NAME}/checkpoints/"
DATACACHE_DIR="${RUN_DIR}/${NAME}/data_cache/"
TENSORBOARD_DIR="${RUN_DIR}/${NAME}/tensorboard/"

mkdir -p ${LOGS_DIR}
mkdir -p ${CHECKPOINT_DIR}
mkdir -p ${DATACACHE_DIR}
mkdir -p ${TENSORBOARD_DIR}

export TRITON_CACHE_DIR="/tmp/triton-cache"


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
echo "RUN_DIR=${RUN_DIR}" |& tee -a ${LOGS_DIR}/${ENV_LOG_FILENAME}
echo "LOGS_DIR=${LOGS_DIR}" |& tee -a ${LOGS_DIR}/${ENV_LOG_FILENAME}
echo "CHECKPOINT_DIR=${CHECKPOINT_DIR}" |& tee -a ${LOGS_DIR}/${ENV_LOG_FILENAME}
echo "DATACACHE_DIR=${DATACACHE_DIR}" |& tee -a ${LOGS_DIR}/${ENV_LOG_FILENAME}
echo "TENSORBOARD_DIR=${TENSORBOARD_DIR}" |& tee -a ${LOGS_DIR}/${ENV_LOG_FILENAME}
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


#--result-rejected-tracker-filename ${RESULT_REJECTED_TRACKER_FILENAME} \
#--iterations-to-skip ${ITERATIONS_TO_SKIP} \
#--rerun-mode validate_results \
#
#--enable-experimental \
#--moe-shared-expert-overlap \

        # Can not use with FP4

        # MXFP8
        #--moe-router-padding-for-fp8 \
        #--fp8-format e4m3 \
        #--fp8-recipe mxfp8 \
        #--fp8-param-gather \
        #--reuse-grad-buf-for-mxfp8-param-ag \

        # Additional options
        #--recompute-modules layernorm moe_act \
        #
        #--recompute-granularity selective \
        #--recompute-modules moe \
        #
        #--tp-comm-overlap \

        # Short context, use
        # --enable-cuda-graph \
        # Long context, use
        # --recompute-granularity selective \
        # --recompute-modules moe \

        # NVFP4 args
        # --keep-mtp-spec-in-bf16 \
        # --keep-mamba-stack-attention-linear-in-bf16 \
        # --keep-mamba-out-proj-in-mxfp8 \
        # --keep-moe-latent-projections-in-bf16 \
        # --first-last-layers-bf16 \
        # --num-layers-at-start-in-bf16 0 \
        # --num-layers-at-end-in-bf16 14 \
        # --fp4-format e2m1 \
        # --fp4-recipe nvfp4 \

        # checkpoint load fix
        # --cuda-graph-scope mamba attn moe_router \
                # --ckpt-fully-parallel-load \
        # --async-save \
                        # --use-persistent-ckpt-worker \


SEQ_LEN=262144
TRAIN_SAMPLES=10000
LR_WARMUP_SAMPLES=100
LR_DECAY_SAMPLES=$((TRAIN_SAMPLES-LR_WARMUP_SAMPLES))
LOG_INTERVAL=1
SAVE_INTERVAL=20
SAVE_RETAIN_INTERVAL=100
GBS=64
LR=1e-5
MIN_LR=2e-6

TOKENIZER_MODEL_PATH="/lustre/fs1/portfolios/llmservice/projects/llmservice_modelalignment_ppo/users/adithyare/nemotron_super/tokenizer"
BASE_MODEL_PATH="/lustre/fs1/portfolios/llmservice/projects/llmservice_modelalignment_ppo/users/adithyare/nemotron_ultra/reinit-embeddings-ckpts/phase1_fp32rs_continued_iter_0600000/checkpoints"
BLEND_PATH="/lustre/fsw/portfolios/llmservice/users/adithyare/nemotron_ultra/blend_jan21.json"


OPTIONS=" \
    --sft \
    --sft-tokenizer-prompt-format identity \
    --distributed-timeout-minutes 5 \
    --num-dataset-builder-threads 32 \
    --tokenizer-type SFTTokenizer \
    --tokenizer-model ${TOKENIZER_MODEL_PATH} \
        \
        --recompute-granularity selective \
        --recompute-modules moe \
        --mtp-use-repeated-layer \
        \
        --context-parallel-size 16 \
        --tensor-model-parallel-size 8 \
        --expert-model-parallel-size 128 \
        --expert-tensor-parallel-size 1 \
        --pipeline-model-parallel-size 1 \
        --hybrid-override-pattern MEMEMEM*EMEMEM*EMEMEMEM*EMEMEMEM*EMEMEM*EMEMEMEM*EMEMEMEM*EMEMEM*EMEMEMEM*EMEMEMEM*EMEMEM*EMEMEMEM*EMEMEMEME \
        --mtp-hybrid-override-pattern \"*E\" \
        \
        --pretrained-checkpoint ${BASE_MODEL_PATH} \
        --save-interval ${SAVE_INTERVAL} \
        --save-retain-interval ${SAVE_RETAIN_INTERVAL} \
        --lr $LR \
        --min-lr $MIN_LR \
        --lr-decay-style constant \
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
        --calculate-per-token-loss \
        --mtp-loss-scaling-factor 0.3 \
        \
        --cuda-graph-scope mamba attn moe_router \
        --te-rng-tracker \
        --high-priority-stream-groups ep \
        --manual-gc-interval 10 \
        --ddp-num-buckets 10 \
        --manual-gc \
        \
        --moe-latent-size 2048 \
        --moe-permute-fusion \
        --cross-entropy-loss-fusion \
        --cross-entropy-fusion-impl native \
        --use-fused-weighted-squared-relu \
        \
        --moe-token-dispatcher-type alltoall \
        --moe-router-score-function sigmoid \
        --moe-grouped-gemm \
        --num-experts 512 \
        --moe-router-topk 22 \
        --moe-aux-loss-coeff 1e-4 \
        --moe-router-topk-scaling-factor 5.0 \
        --moe-router-enable-expert-bias \
        --moe-router-dtype fp32 \
        --moe-router-load-balancing-type seq_aux_loss \
        --moe-shared-expert-intermediate-size 10240 \
        \
        --attention-backend flash \
        --num-workers 1 \
        --disable-gloo-process-groups \
        --ckpt-format torch_dist \
        --ckpt-fully-parallel-save \
        --ckpt-fully-parallel-load \
        --ckpt-assume-constant-structure \
        --use-persistent-ckpt-worker \
        \
        --squared-relu \
        --no-mmap-bin-files \
        --exit-duration-in-mins 5750 \
        --no-create-attention-mask-in-dataloader \
        \
        --sequence-parallel \
        --use-distributed-optimizer \
        --override-opt-param-scheduler \
        \
        --mamba-num-heads 256 \
        --is-hybrid-model \
        --untie-embeddings-and-output-weights \
        --init-method-std 0.014 \
        --position-embedding-type none \
        --num-layers 108 \
        --hidden-size 8192 \
        --num-attention-heads 64 \
        --group-query-attention \
        --num-query-groups 2 \
        --ffn-hidden-size 5120 \
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
        --no-load-optim \
        --adam-beta1 0.9 \
        --adam-beta2 0.95 \
        --log-params-norm \
        --log-num-zeros-in-grad \
        --log-throughput \
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
        --te-precision-config-file /lustre/fs1/portfolios/llmservice/projects/llmservice_nlp_fm/nemotron6/code_ultra/te_quant.cfg \
        --bf16 \
        --use-mcore-models \
        --spec megatron.core.models.mamba.mamba_layer_specs mamba_stack_spec \
        --wandb-project ${WANDB_PROJECT} \
        --wandb-exp-name ${NAME} \
        --dist-ckpt-strictness log_unexpected \
        --tensorboard-dir ${TENSORBOARD_DIR}"

RUN_CMD="python -u ${MEGATRON_LM_DIR}/pretrain_mamba.py ${OPTIONS}"

# Launch via bindpcie: CPU=node, mem=node (NUMA binding per rank). Script uses LOCAL_RANK or SLURM_LOCALID.
LAUNCH_CMD="${BINDPCIE_SCRIPT} --cpu=node --mem=node -- ${RUN_CMD}"

srun -l \
     --mpi=none \
     --no-container-mount-home \
     --container-image=${IMAGE} \
     --container-mounts="/lustre:/lustre" \
     --container-env=NUM_OF_HYBRID_EP_RANKS_PER_NVLINK_DOMAIN,USE_MNNVL,UCX_MEM_MMAP_HOOK_MODE,UCX_MEM_CUDA_HOOK_MODE,UCX_MEM_MALLOC_HOOKS,UCX_ERROR_SIGNALS,NVTE_CPU_OFFLOAD_V1,SLURM_LOCALID \
     --output="${LOGS_DIR}/%x_%j_${DATETIME}.log" \
     sh -c "${LAUNCH_CMD}"
