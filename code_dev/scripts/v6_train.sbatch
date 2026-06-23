#!/bin/bash
# V6 train-parity: load the SHARED init dist-ckpt (v6_save_init.sbatch) into a
# TP=${TP} (+--sequence-parallel when TP>1) Gemma 4 E4B smoke model and train
# ${NUM_STEPS} steps, logging per-iter "lm loss" and "grad norm". torch_dist
# auto-reshards the TP=1 init into the TP=D model, and --finetune resets the
# iteration to 0 + uses a FRESH optimizer/RNG, so all degrees start from
# BIT-IDENTICAL weights with identical data (--mock-data, fixed --seed) and
# identical GBS/MBS. The per-step loss + total grad-norm must therefore MATCH
# TP=1 to a bf16 tolerance that does NOT grow with TP degree. This exercises the
# backward / gradient-reduction path of the SP fixes (PLE grad + norm grad-tag).
#
# Parameterized by env: TP (1/2/4/8), SP (set to 1 to add --sequence-parallel).
# GPUS-per-node MUST equal TP. Submit with, e.g.:
#   sbatch --gpus-per-node=2 --mem=400G --job-name=IMPL-gemma4-tpsp_v6_tp2 \
#          --export=ALL,TP=2,SP=1 code_dev/scripts/v6_train.sbatch
#SBATCH --account=coreai_dlalgo_genai
#SBATCH -p batch
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --time=00:40:00
#SBATCH --output=/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_genai/users/ataghibakhsh/Gemma4_mlm/Megatron-LM-tpsp/code_dev/shared-state/logs/%j_%x.out

set -eu
unset SLURM_CPUS_PER_TASK SLURM_TRES_PER_TASK SLURM_CPU_BIND SLURM_DISTRIBUTION

TP=${TP:-1}
SP=${SP:-0}

IMAGE=/lustre/fsw/portfolios/coreai/projects/coreai_nvfm_llm/containers/nemo-25.11.nemotron_3_nano-mamba_ssm_2.3.0.sqsh
MLM=/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_genai/users/ataghibakhsh/Gemma4_mlm/Megatron-LM-tpsp
SHARED=${MLM}/code_dev/shared-state
INITCKPT=${SHARED}/implementations/IMPL-gemma4-tpsp/v6_init_ckpt
TRITON=${SHARED}/v6_triton_cache
DATACACHE=${SHARED}/v6_data_cache
mkdir -p ${TRITON} ${DATACACHE}

# ---- SMOKE dims (identical to v6_save_init.sbatch).
NUM_LAYERS=42
HIDDEN=2560
FFN=2048
HEADS=8
KV_GROUPS=2
VOCAB=32768
SEQLEN=1024
MBS=1
GBS=4
SEED=1234
NUM_STEPS=${NUM_STEPS:-10}
TRAIN_SAMPLES=$((GBS * NUM_STEPS))

# --sequence-parallel only when TP>1 (rejected at TP=1).
SP_FLAG=""
if [ "${SP}" = "1" ] && [ "${TP}" -gt 1 ]; then
    SP_FLAG="--sequence-parallel"
fi

options=" \
    --use-mcore-models \
    --transformer-impl local \
    --num-layers ${NUM_LAYERS} \
    --hidden-size ${HIDDEN} \
    --ffn-hidden-size ${FFN} \
    --num-attention-heads ${HEADS} \
    --group-query-attention \
    --num-query-groups ${KV_GROUPS} \
    --normalization RMSNorm \
    --norm-epsilon 1e-6 \
    --qk-layernorm \
    --disable-bias-linear \
    --position-embedding-type none \
    --untie-embeddings-and-output-weights \
    --attention-dropout 0.0 \
    --hidden-dropout 0.0 \
    --init-method-std 0.02 \
    --tensor-model-parallel-size ${TP} \
    ${SP_FLAG} \
    --pipeline-model-parallel-size 1 \
    --context-parallel-size 1 \
    --seq-length ${SEQLEN} \
    --max-position-embeddings ${SEQLEN} \
    --micro-batch-size ${MBS} \
    --global-batch-size ${GBS} \
    --train-samples ${TRAIN_SAMPLES} \
    --seed ${SEED} \
    --lr 1e-4 \
    --min-lr 1e-5 \
    --lr-decay-style cosine \
    --lr-warmup-samples 0 \
    --weight-decay 0.0 \
    --clip-grad 1.0 \
    --adam-beta1 0.9 \
    --adam-beta2 0.95 \
    --bf16 \
    --mock-data \
    --tokenizer-type NullTokenizer \
    --vocab-size ${VOCAB} \
    --no-create-attention-mask-in-dataloader \
    --distributed-timeout-minutes 20 \
    --disable-gloo-process-groups \
    --num-workers 1 \
    --log-interval 1 \
    --eval-iters 0 \
    --eval-interval 100000 \
    --save-interval 100000 \
    --ckpt-format torch_dist \
    --load ${INITCKPT} \
    --finetune \
    --no-load-optim \
    --no-load-rng \
    --logging-level 20 \
    --data-cache-path ${DATACACHE} \
    "

srun --container-image=$IMAGE \
     --container-mounts=/lustre:/lustre,/home/ataghibakhsh:/home/ataghibakhsh \
     --no-container-mount-home \
     bash -c "
set -x
export CUDA_DEVICE_MAX_CONNECTIONS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1
export TRITON_CACHE_DIR=${TRITON}
export TRITON_HOME=${TRITON}
export TORCH_COMPILE_DISABLE=1
export TORCHDYNAMO_DISABLE=1
export PYTHONPATH=${MLM}:\${PYTHONPATH:-}
cd ${MLM}
echo '########## V6 train-parity TP='${TP}' SP='${SP}' steps='${NUM_STEPS}' ##########'
# Port depends on TP so two degrees co-scheduled on one node never collide (EADDRINUSE).
MASTER_PORT=\$((12430 + ${TP}))
torchrun --nproc-per-node=${TP} --master-port=\${MASTER_PORT} ${MLM}/pretrain_gemma4.py ${options}
echo EXIT_TRAIN=\$?
echo '########## V6 train-parity TP='${TP}' DONE ##########'
"
