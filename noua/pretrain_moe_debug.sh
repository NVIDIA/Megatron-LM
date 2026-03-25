#!/bin/bash
# MoE model training script
# Usage: SLURM_JOB_ID=22004084 bash noua/pretrain_moe_debug.sh

set -euo pipefail
export WANDB_API_KEY=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
export DEBUG_PORT=5678

# ── Profiling ─────────────────────────────────────────────────────────────────
PROFILE=${PROFILE:-0}
PROFILE_RANKS=${PROFILE_RANKS:-"0,1,2,3"}  # Global GPU ranks for profiling

# ── Paths ────────────────────────────────────────────────────────────────────
MEGATRON_LM=/fsx/nouamane/projects/Pai-Megatron-Patch/Megatron-LM-260325
TRANSFORMER_ENGINE_SRC=/fsx/nouamane/projects/TransformerEngine-260127
EMERGING_OPTIMIZERS=/fsx/nouamane/projects/Emerging-Optimizers
# CONTAINER_IMAGE="/fsx/nouamane/docker_images/nemo-25.11-nemotron.sqsh"
CONTAINER_IMAGE="/fsx/nouamane/docker_images/library+nouamane-26-02+latest.sqsh"
CONTAINER_MOUNTS="/scratch:/scratch,/fsx:/fsx"


SCRIPT_DIR="$(dirname "$(readlink -f "$0")")"

# ── Cluster ──────────────────────────────────────────────────────────────────
NNODES=${NNODES:-1}
GPUS_PER_NODE=4
TOTAL_GPUS=$((NNODES * GPUS_PER_NODE))

# ── Training ─────────────────────────────────────────────────────────────────
MBS=${MBS:-2}
GBS=$((2 * TOTAL_GPUS * MBS))

# ── Logging ──────────────────────────────────────────────────────────────────
RUN_NAME="${NNODES}n_gbs${GBS}_localattn"
TIMESTAMP=$(date +'%y%m%d_%H%M%S')
CKPT_DIR="${SCRIPT_DIR}/checkpoints/${RUN_NAME}"
TB_DIR="${CKPT_DIR}/tensorboard"
LOG_DIR="${SCRIPT_DIR}/slurm_logs/${RUN_NAME}"
mkdir -p "${LOG_DIR}"

cp -f "${BASH_SOURCE[0]}" "${LOG_DIR}/$(basename "${BASH_SOURCE[0]}")-${TIMESTAMP}.sh" 2>/dev/null || true

# ── Profile command setup ─────────────────────────────────────────────────────
if [[ ${PROFILE} -eq 1 ]]; then
    NSYS_PATH="${LOG_DIR}"
    mkdir -p "${NSYS_PATH}"

    # Template with node rank in filename - evaluated inside srun
    NSYS_CMD_BASE="nsys profile --sample=none --cpuctxsw=none -t cuda,nvtx \
        --capture-range=cudaProfilerApi \
        --capture-range-end=stop \
        --cuda-graph-trace=node \
        -f true -x true \
        -o ${NSYS_PATH}/${RUN_NAME}-${TIMESTAMP}-node\${SLURM_PROCID}"

    # Compute which nodes contain the profiled GPU ranks
    PROFILE_NODE_RANKS=""
    for gpu_rank in ${PROFILE_RANKS//,/ }; do
        node_rank=$((gpu_rank / GPUS_PER_NODE))
        if [[ ! ",${PROFILE_NODE_RANKS}," == *",${node_rank},"* ]]; then
            PROFILE_NODE_RANKS="${PROFILE_NODE_RANKS:+${PROFILE_NODE_RANKS},}${node_rank}"
        fi
    done

    PROFILE_CMD_CHECK="if [[ \",${PROFILE_NODE_RANKS},\" == *\",\${SLURM_PROCID},\"* ]]; then PROFILE_CMD=\"${NSYS_CMD_BASE}\"; else PROFILE_CMD=\"\"; fi"
    PROFILE_PARAMS="profiling.use_nsys_profiler=true profiling.profile_step_start=5 profiling.profile_step_end=7 'profiling.profile_ranks=[${PROFILE_RANKS}]'"
else
    PROFILE_CMD_CHECK=""
    NSYS_CMD_BASE=""
    PROFILE_PARAMS=""
fi

# Capture extra CLI args to pass to torchrun
EXTRA_ARGS="$*"

# ── Arg groups ───────────────────────────────────────────────────────────────
MODEL_ARGS=(
    --num-layers 4   # TODO:
    --hidden-size 2048
    --ffn-hidden-size 6144
    --num-attention-heads 20
    --kv-channels 192
    --normalization RMSNorm
    --norm-epsilon 1e-6
    --apply-layernorm-1p
    --swiglu
    --position-embedding-type rope
    --rotary-base 10000
    --disable-bias-linear
    --untie-embeddings-and-output-weights
    --hidden-dropout 0.0
    --attention-dropout 0.0
    --seq-length 4096
    --max-position-embeddings 4096
    --make-vocab-size-divisible-by 128
    --init-method-std 0.008
)

MLA_ARGS=(
    --multi-latent-attention                      # = glm4.7's (attn)
    --q-lora-rank 768                             # = glm4.7's 768 (attn)
    --kv-lora-rank 512                             # = glm4.7's 512 (attn)
    --qk-head-dim 192                             # = glm4.7's 192 (attn)
    --qk-pos-emb-head-dim 64                      # = glm4.7's 64 (attn)
    --v-head-dim 256                             # = glm4.7's 256 (attn)
    --qk-clip                                    # = kimi2 (attn+muon)
)

MOE_ARGS=(
    --num-experts 128
    --moe-ffn-hidden-size 768
    --moe-router-topk 7
    --moe-shared-expert-intermediate-size 768
    --moe-layer-freq "'([0]+[1]*3)'"
    --moe-router-score-function sigmoid
    --moe-router-pre-softmax
    --moe-router-load-balancing-type seq_aux_loss
    --moe-router-bias-update-rate 0.001
    --moe-aux-loss-coeff 0.0001
    --moe-router-enable-expert-bias
    --moe-router-topk-scaling-factor 2.5
    --moe-router-dtype fp32
)


OPTIMIZER_ARGS=(
    --optimizer adam
    --use-distributed-optimizer
    # --optimizer dist_muon
    --lr 1e-3                          
    --min-lr 1e-4        
    --adam-beta1 0.9                         
    --adam-beta2 0.999                         
    --adam-eps 1e-8                             
    --muon-momentum 0.95                        
    --muon-use-nesterov                        
    --muon-scale-mode shape_scaling              
    --muon-extra-scale-factor 1.0                
    --muon-num-ns-steps 5                        
    --weight-decay 0.1                 # TODO:                                      
    --lr-decay-style WSD                      
    --lr-warmup-samples $((20 * GBS))   # TODO:                 
    --lr-wsd-decay-style cosine                     
    --lr-wsd-decay-samples 0
    --clip-grad 1.0                             
)
    # --rampup-batch-size 8 8 10000         # TODO:                 



# Note: Muon/dist_muon does not support --use-distributed-optimizer
MODEL_PARALLEL_ARGS=(
    --tensor-model-parallel-size 1
    --pipeline-model-parallel-size 2
    --num-virtual-stages-per-pipeline-rank 2
    --expert-model-parallel-size 2
    --overlap-grad-reduce
    --overlap-param-gather
    --moe-token-dispatcher-type alltoall
    --overlap-moe-expert-parallel-comm              
    # --moe-shared-expert-overlap  # disable moe_shared_expert_overlap when enabling overlap_moe_expert_parallel_comm    
)


FUSING_ARGS=(
    --no-rope-fusion   
    --mla-down-proj-fusion
    --moe-grouped-gemm
    --moe-router-fusion
    --moe-permute-fusion
    --cross-entropy-loss-fusion
    --cross-entropy-fusion-impl te
    --attention-backend fused # TODO: for return_max_logit
    --transformer-impl transformer_engine
    --no-check-for-nan-in-loss-and-grad
    --disable-symmetric-registration
    #TODO: fusion
    #TODO: recomputation
    #TODO: cuda graph
    #TODO: miscellaneous
)

TRAINING_ARGS=(
    --micro-batch-size ${MBS}
    --global-batch-size ${GBS}
    --train-samples $((100 * GBS))
    --eval-interval 999999999
    --eval-iters 0
    --bf16
    --ckpt-format torch_dist
    --async-save
    --ckpt-fully-parallel-load
    --seed 41
    --exit-signal-handler
    # --exit-signal
)


DATA_ARGS=(
    --tokenizer-type HuggingFaceTokenizer
    --tokenizer-model alehc/swissai-tokenizer
    # --data-cache-path ${DATASET_CACHE}
    # --per-split-data-args-path ${DATA_CONFIG}
    --mock-data
    --dataloader-type single
    --num-workers 0
    --num-dataset-builder-threads 8
    --reset-position-ids
    --reset-attention-mask
    --eod-mask-loss
)

LOGGING_ARGS=(
    --log-interval 1
    --log-throughput
    --log-timers-to-tensorboard
    --tensorboard-queue-size 1
    --tensorboard-dir ${TB_DIR}
    --save-interval 1000
    --save ${CKPT_DIR}
    --wandb-project qwen3-moekk
    --wandb-exp-name ${RUN_NAME}
    --distributed-timeout-minutes 600
    --manual-gc
    --manual-gc-interval 500
    --moe-per-layer-logging
    --log-params-norm
    --log-progress
    --timing-log-level 2  # TODO:
    --log-memory-interval 100
    --log-device-memory-used
    --log-max-attention-logit
    --logging-level 10
)

# ── Validate ─────────────────────────────────────────────────────────────────
if [[ -z "${SLURM_JOB_ID:-}" ]]; then
    echo "Error: SLURM_JOB_ID not set. Run with: SLURM_JOB_ID=<jobid> bash $(basename "$0")"
    exit 1
fi

# ── SLURM Setup ──────────────────────────────────────────────────────────────
unset MASTER_ADDR SLURM_JOB_NODELIST

export SLURM_JOB_NODELIST=$(scontrol show job -o "${SLURM_JOB_ID}" \
    | tr ' ' '\n' \
    | awk -F= '$1=="NodeList"{print $2}' \
    | awk '$0 != "(null)" {print; exit}')

MASTER_ADDR="$(scontrol show hostnames ${SLURM_JOB_NODELIST} | head -n 1)"
MASTER_PORT=$((29500 + RANDOM % 1000))

echo "Job ${SLURM_JOB_ID} | Nodes: ${SLURM_JOB_NODELIST} | Master: ${MASTER_ADDR}:${MASTER_PORT}"

LOG_FILE="${LOG_DIR}/${RUN_NAME}-${TIMESTAMP}-${SLURM_JOB_ID}.out"
echo "Logging to: ${LOG_FILE}"

set +e

srun \
    --jobid=${SLURM_JOB_ID} \
    --overlap \
    --nodes=${NNODES} \
    --ntasks=${NNODES} \
    --ntasks-per-node=1 \
    --container-image=${CONTAINER_IMAGE} \
    --container-mounts=${CONTAINER_MOUNTS} \
    --no-container-mount-home \
    bash -lc "
set -x

# ── Distributed Setup ──
export SLURM_GPUS_ON_NODE=${GPUS_PER_NODE}
export SLURM_PROCID=\${SLURM_PROCID}
export MASTER_ADDR=${MASTER_ADDR}
export MASTER_PORT=${MASTER_PORT}

# ── Code Paths ──
export PYTHONPATH=${EMERGING_OPTIMIZERS}:${MEGATRON_LM}:\${PYTHONPATH:-}
export PYTHONPATH=${TRANSFORMER_ENGINE_SRC}:\${PYTHONPATH}


# ── NCCL / Communication ──
export NCCL_DEBUG=WARN
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=36000
export NCCL_TIMEOUT=36000000
export NCCL_NVLS_ENABLE=0
export NCCL_P2P_NET_CHUNKSIZE=2097152
export NCCL_GRAPH_REGISTER=0
export TORCH_NCCL_AVOID_RECORD_STREAMS=0
export TORCH_NCCL_HIGH_PRIORITY=1
export NVLINK_DOMAIN_SIZE=4
export UB_SKIPMC=1


# ── TransformerEngine ──
export NVTE_DEBUG=1
export NVTE_DEBUG_LEVEL=2
export NVTE_NORM_FWD_USE_CUDNN=1
export NVTE_NORM_BWD_USE_CUDNN=1
export NVTE_FWD_LAYERNORM_SM_MARGIN=0
export NVTE_BWD_LAYERNORM_SM_MARGIN=0
export NVTE_ALLOW_NONDETERMINISTIC_ALGO=1
export NVTE_USE_CUTLASS_GROUPED_GEMM=0
export NVTE_CUTLASS_GROUPED_GEMM_WARN_FALLBACK=1

# ── PyTorch / CUDA ──
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_DEVICE_MAX_CONNECTIONS=1

# ── Libfabric ──
# export FI_MR_CACHE_MONITOR=disabled

# ── HuggingFace ──
# export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=False
# export HF_HOME=/iopsstor/scratch/cscs/ntazi/.cache/huggingface
# export HF_HUB_CACHE=/iopsstor/scratch/cscs/ntazi/.cache/huggingface/hub

# ── Triton ──
# export TRITON_CACHE_DIR=/iopsstor/scratch/cscs/ntazi/.cache/triton/triton_cache_\${SLURM_PROCID}
# mkdir -p \${TRITON_CACHE_DIR}

# ── Wandb ──
export WANDB_MODE=offline
export WANDB_API_KEY=\${WANDB_API_KEY:-}

cd ${MEGATRON_LM}

# # ── Log package versions ──
# echo '=== Package Versions ==='
# python -c '
# import subprocess, pathlib
# def git_info(pkg_path):
#     try:
#         d = str(pathlib.Path(pkg_path).resolve().parent)
#         fmt = \"%h %s (%ci)\"
#         return subprocess.check_output([\"git\", \"log\", \"-1\", \"--format=\" + fmt], cwd=d, stderr=subprocess.DEVNULL).decode().strip()
#     except Exception:
#         return \"unknown\"
# import megatron.core as mc; print(f\"megatron-core: {mc.__version__} | {git_info(mc.__file__)}\")
# import transformer_engine as te; print(f\"transformer-engine: {te.__version__} | {git_info(te.__file__)}\")
# try:
#     import transformer_engine_torch as te_torch
#     print(f\"  TE .so: {te_torch.__file__}\")
# except Exception as e:
#     print(f\"  TE .so: failed to load ({e})\")
# ' 2>&1 || echo 'WARNING: version check failed'
# echo '========================'

PROFILE_CMD=\"\"
${PROFILE_CMD_CHECK}
echo \"Node \${SLURM_PROCID}: PROFILE_CMD=\${PROFILE_CMD:-none}\"

\${PROFILE_CMD} python -m torch.distributed.run \
    --nproc_per_node=\${SLURM_GPUS_ON_NODE} \
    --nnodes=${NNODES} \
    --node_rank=\${SLURM_PROCID} \
    --master_addr=\${MASTER_ADDR} \
    --master_port=\${MASTER_PORT} \
    pretrain_gpt.py \
    ${MODEL_ARGS[*]} \
    ${MLA_ARGS[*]} \
    ${MOE_ARGS[*]} \
    ${MODEL_PARALLEL_ARGS[*]} \
    ${FUSING_ARGS[*]} \
    ${TRAINING_ARGS[*]} \
    ${OPTIMIZER_ARGS[*]} \
    ${DATA_ARGS[*]} \
    ${LOGGING_ARGS[*]} \
    ${PROFILE_PARAMS} \
    ${EXTRA_ARGS}
" 2>&1 | tee "${LOG_FILE}"
