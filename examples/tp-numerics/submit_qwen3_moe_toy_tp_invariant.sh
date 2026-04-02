#!/bin/bash
#
# Toy MoE TP-invariant numerics smoke test. Raw Megatron-LM.
#
# Toy MoE: Qwen3-30B-A3B base architecture + overrides:
#   - 4 layers (vs 48 in full A3B)
#   - 8 experts (vs 128)
#   - top-2 (vs top-8)
#   - moe_grouped_gemm=False (CRITICAL: bypasses BIK if True)
#   - moe_token_dispatcher_type=alltoall
#   - EP=1, GBS=1, MBS=1
#   ⇒ fits on 1 GPU at TP=1
#
# Runs both ways:
#   - Interactive (active salloc):   bash submit_qwen3_moe_toy_tp_invariant.sh
#   - SLURM batch:                   sbatch submit_qwen3_moe_toy_tp_invariant.sh
#
# Tunable env (override at invocation):
#   TP_SIZE       (default 1)        — tensor-parallel degree
#   TRAIN_ITERS   (default 10)
#   OUTPUT_DIR    (default ./output/qwen3_moe_toy_tp${TP_SIZE}_iter${TRAIN_ITERS})

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --time=00:30:00
#SBATCH --job-name=qwen3_moe_toy_tp_invariant
#SBATCH --output=slurm_%j.log
#SBATCH --exclusive

set -euo pipefail

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
WORKSPACE="${WORKSPACE:-/home/scratch.jinzex_sw/jinzex-ai-workspace/engineering/reflection-engineering}"
MEGATRON_PATH="${MEGATRON_PATH:-${WORKSPACE}/third-party/Megatron-LM}"
CONTAINER_IMAGE="${CONTAINER_IMAGE:-${WORKSPACE}/containers/nemo-25.11.01.sqsh}"
CONTAINER_MOUNTS="${CONTAINER_MOUNTS:-/home/scratch.jinzex_sw:/home/scratch.jinzex_sw}"

TP_SIZE=${TP_SIZE:-1}
TRAIN_ITERS=${TRAIN_ITERS:-10}
OUTPUT_DIR="${OUTPUT_DIR:-${WORKSPACE}/projects/Numerics/tp-numerics/results-b300/qwen3_moe_toy_tp${TP_SIZE}_iter${TRAIN_ITERS}}"
LOG_FILE="${OUTPUT_DIR}/training.log"

mkdir -p "${OUTPUT_DIR}"

# ---------------------------------------------------------------------------
# Environment (TP-invariant numerics + determinism)
# ---------------------------------------------------------------------------
export NVTE_TP_INVARIANT_MODE=1
export NVTE_ALLOW_NONDETERMINISTIC_ALGO=0
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export NCCL_ALGO=^NVLS
export NCCL_NVLS_ENABLE=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# Force unfused attention.
export NVTE_FLASH_ATTN=0
export NVTE_FUSED_ATTN=0
export CUDA_DEVICE_MAX_CONNECTIONS=1

# Sequence parallel only when TP>1.
SP_FLAG=""
if [[ "${TP_SIZE}" -gt 1 ]]; then
    SP_FLAG="--sequence-parallel"
fi

# TE TP-invariant patches: applied inside the same srun as training because
# squashfs containers are read-only per-srun. Detects TE version and picks
# the matching subdir under patches/; fails fast if unsupported.
SETUP_CMD="\
P=${MEGATRON_PATH}/examples/tp-numerics/patches && \
TE_VERSION=\$(python3 -c 'import transformer_engine as te; print(te.__version__.split(\"+\")[0])') && \
PATCH_SRC=\$P/v\${TE_VERSION}/transformer_engine/pytorch && \
if [ ! -d \$PATCH_SRC ]; then echo \"ERROR: no TP-invariant patches for TE \$TE_VERSION (have: \$(ls \$P | grep ^v | tr '\n' ' '))\"; exit 1; fi && \
cp -r \$PATCH_SRC/* /opt/venv/lib/python3.12/site-packages/transformer_engine/pytorch/"

# ---------------------------------------------------------------------------
# Training command — toy MoE config
# ---------------------------------------------------------------------------
TRAINING_CMD="python ${MEGATRON_PATH}/pretrain_gpt.py \
    --distributed-timeout-minutes 60 \
    --tensor-model-parallel-size ${TP_SIZE} \
    --pipeline-model-parallel-size 1 \
    --expert-model-parallel-size 1 \
    --expert-tensor-parallel-size 1 \
    --context-parallel-size 1 \
    --use-distributed-optimizer \
    ${SP_FLAG} \
    --transformer-impl transformer_engine \
    --attention-backend unfused \
    --micro-batch-size 1 \
    --global-batch-size 1 \
    --train-iters ${TRAIN_ITERS} \
    --eval-iters 0 \
    --eval-interval $((TRAIN_ITERS + 1)) \
    --num-layers 4 \
    --hidden-size 2048 \
    --ffn-hidden-size 6144 \
    --num-attention-heads 32 \
    --group-query-attention \
    --num-query-groups 4 \
    --kv-channels 128 \
    --normalization RMSNorm \
    --norm-epsilon 1e-06 \
    --swiglu \
    --qk-layernorm \
    --disable-bias-linear \
    --untie-embeddings-and-output-weights \
    --hidden-dropout 0.0 \
    --attention-dropout 0.0 \
    --no-bias-dropout-fusion \
    --no-bias-swiglu-fusion \
    --no-persist-layer-norm \
    --no-rope-fusion \
    --position-embedding-type rope \
    --rotary-base 1000000 \
    --rotary-percent 1.0 \
    --max-position-embeddings 40960 \
    --seq-length 2048 \
    --vocab-size 32000 \
    --null-tokenizer-vocab-includes-eod \
    --padded-vocab-size 151936 \
    --make-vocab-size-divisible-by 128 \
    --init-method-std 0.02 \
    --lr 3e-4 \
    --min-lr 3e-5 \
    --lr-decay-style cosine \
    --lr-warmup-iters 0 \
    --lr-decay-iters ${TRAIN_ITERS} \
    --weight-decay 0.033 \
    --adam-beta1 0.9 \
    --adam-beta2 0.95 \
    --adam-eps 1e-5 \
    --clip-grad 1.0 \
    --optimizer adam \
    --bf16 \
    --use-cpu-initialization \
    --deterministic-mode \
    --batch-invariant-mode \
    --mock-data \
    --tokenizer-type NullTokenizer \
    --seed 1234 \
    --num-experts 8 \
    --moe-ffn-hidden-size 768 \
    --moe-router-load-balancing-type aux_loss \
    --moe-router-topk 2 \
    --moe-aux-loss-coeff 1e-3 \
    --moe-permute-fusion \
    --moe-token-dispatcher-type alltoall \
    --log-interval 1 \
    --tensorboard-dir ${OUTPUT_DIR}/tensorboard"

# ---------------------------------------------------------------------------
# Launch (1 GPU at TP=1, EP=1)
# ---------------------------------------------------------------------------
NUM_GPUS=${TP_SIZE}
srun ${SRUN_EXTRA_ARGS:-} --mpi=pmix --overlap --cpu-bind=none -l \
     --ntasks=${NUM_GPUS} --ntasks-per-node=${NUM_GPUS} \
     --gpus-per-node=${NUM_GPUS} \
     --no-container-mount-home \
     --container-image="${CONTAINER_IMAGE}" \
     --container-mounts="${CONTAINER_MOUNTS}" \
     --container-workdir="${MEGATRON_PATH}" \
     bash -c "${SETUP_CMD} && ${TRAINING_CMD}" 2>&1 | tee "${LOG_FILE}"

EXPECTED_ITER1="lm loss: 1.232552E+01 grad norm: 6.592"
EXPECTED_LAST="lm loss: 1.449088E+01 grad norm: 1.412"

extract_iter() {
    grep "iteration *$1/" "${LOG_FILE}" | head -1 \
        | grep -oP '(lm loss|grad norm): \S+' | tr '\n' ' ' | sed 's/ $//'
}
ITER1=$(extract_iter 1)
ITER_LAST=$(extract_iter "${TRAIN_ITERS}")

echo
echo "Done. Loss curve in: ${LOG_FILE}"
echo "Iter 1  (this run):  ${ITER1}"
echo "Iter ${TRAIN_ITERS} (this run):  ${ITER_LAST}"

if [[ "${TP_SIZE}" == "1" && "${TRAIN_ITERS}" == "10" ]]; then
    echo "Expected iter 1:     ${EXPECTED_ITER1}"
    echo "Expected iter 10:    ${EXPECTED_LAST}"
    if [[ "${ITER1}" != "${EXPECTED_ITER1}" || "${ITER_LAST}" != "${EXPECTED_LAST}" ]]; then
        echo "FAIL: numerics mismatch (regression?)"
        exit 1
    fi
    echo "PASS"
else
    echo "(No assertion: non-default TP_SIZE/TRAIN_ITERS — expected values only valid for default config.)"
fi
