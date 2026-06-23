#!/bin/bash
# V6 init-save: build the SMOKE-dimension Gemma 4 E4B model at TP=1 from a FIXED
# seed (random init) and write a Megatron torch_dist (distributed) checkpoint.
# This is the SHARED INIT that every TP degree in v6_train.sbatch loads via
# --finetune (iteration reset to 0, fresh optimizer/RNG), so the loss/grad-norm
# comparison across TP=1/2/4/8 starts from BIT-IDENTICAL weights. torch_dist
# auto-reshards the TP=1 save into the TP=D model at load time.
#
# We run 1 training iter with --save-interval 1 so the checkpoint is written;
# the comparison runs all load it with --finetune so the post-step-1 weights
# simply serve as the common starting point (symmetric across degrees). 1 GPU.
#SBATCH --account=coreai_dlalgo_genai
#SBATCH -p batch
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=1
#SBATCH --mem=200G
#SBATCH --time=00:40:00
#SBATCH --job-name=IMPL-gemma4-tpsp_v6_init
#SBATCH --output=/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_genai/users/ataghibakhsh/Gemma4_mlm/Megatron-LM-tpsp/code_dev/shared-state/logs/%j_%x.out

set -eu
unset SLURM_CPUS_PER_TASK SLURM_TRES_PER_TASK SLURM_CPU_BIND SLURM_DISTRIBUTION

IMAGE=/lustre/fsw/portfolios/coreai/projects/coreai_nvfm_llm/containers/nemo-25.11.nemotron_3_nano-mamba_ssm_2.3.0.sqsh
MLM=/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_genai/users/ataghibakhsh/Gemma4_mlm/Megatron-LM-tpsp
SHARED=${MLM}/code_dev/shared-state
INITCKPT=${SHARED}/implementations/IMPL-gemma4-tpsp/v6_init_ckpt
TRITON=${SHARED}/v6_triton_cache
DATACACHE=${SHARED}/v6_data_cache
mkdir -p ${INITCKPT} ${TRITON} ${DATACACHE}

# ---- SMOKE dims (fit 1 H100 WITH optimizer; full architecture intact).
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

# Exactly 1 iteration -> --save-interval 1 writes the checkpoint, then we exit.
TRAIN_SAMPLES=${GBS}

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
    --tensor-model-parallel-size 1 \
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
    --ckpt-format torch_dist \
    --save ${INITCKPT} \
    --save-interval 1 \
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
echo '########## V6 init-save (TP=1, smoke dims, fixed seed) ##########'
torchrun --nproc-per-node=1 --master-port=12431 ${MLM}/pretrain_gemma4.py ${options}
echo EXIT_INIT=\$?
echo '########## V6 init-save DONE ##########'
ls -R ${INITCKPT} | head -40
"
