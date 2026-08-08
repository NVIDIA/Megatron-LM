#!/bin/bash
#SBATCH --account=coreai_dlalgo_genai
#SBATCH -p interactive
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=1
#SBATCH --mem=200G
#SBATCH --time=00:30:00
#SBATCH --job-name=HYBRID-gemma4-mlm_V7
#SBATCH --output=/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_genai/users/ataghibakhsh/Gemma4_mlm/code_dev/shared-state/logs/%j_%x.out

# ============================================================================
# V7: Gemma 4 E4B interactive training smoke test (Megatron-LM).
#
# STRIPPED: no distillation / flextron / elastic / MoE / teacher args. Plain
# single-GPU pretrain from RANDOM init with MOCK data + NullTokenizer (vocab
# 262144). The bar is: starts, finite/stable loss for a few steps, no crash.
# Final-logit softcap (30*tanh) and sqrt(H) embed scaling live inside
# Gemma4Model.forward, so they are exercised on the training loss path.
#
# DDP=1, TP=PP=CP=1, no SP, bf16 full precision (no FP8/NVFP4).
# Launch:  sbatch examples/gemma4/gemma4_e4b_interactive.sh
# ============================================================================

set -eu

# Proven container launch idiom.
unset SLURM_CPUS_PER_TASK SLURM_TRES_PER_TASK SLURM_CPU_BIND SLURM_DISTRIBUTION

IMAGE=/lustre/fsw/portfolios/coreai/projects/coreai_nvfm_llm/containers/nemo-25.11.nemotron_3_nano-mamba_ssm_2.3.0.sqsh
ROOT=/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_genai/users/ataghibakhsh/Gemma4_mlm
MLM=${ROOT}/Megatron-LM
RUN_DIR=${ROOT}/code_dev/shared-state/v7_run
mkdir -p ${RUN_DIR}/triton_cache ${RUN_DIR}/checkpoints ${RUN_DIR}/data_cache

# ---- model dims (Gemma 4 E4B). head_dim (256/512), softcap, PLE, sqrt(H) embed
# ---- scaling, KV bus and rope/mask selection all come from Gemma4TransformerConfig.
NUM_LAYERS=42
HIDDEN=2560
# Real E4B ffn is 10240. The full model is ~4.55B params; its fp32 optimizer master
# weights + Adam moments overflow one 80 GB H100 (V7 is random init, not sharded).
# Shrink the MLP for the V7 smoke ONLY -- this keeps all 42 layers and therefore the
# heterogeneous head_dim (256/512), full_attention_layers, KV-bus producer/borrower
# band, softcap, sqrt(H) embed, PLE and sandwich norms fully intact and on the loss
# path. Override with FFN=10240 for the real-size run (needs more GPUs or sharding).
FFN=${FFN:-2048}
HEADS=8
KV_GROUPS=2
# Real E4B vocab is 262144. For the V7 smoke (random init, fp32 master weights +
# Adam moments) the full 262144 embedding + PLE token table blows past one H100's
# 80 GB in the optimizer's fp32 clone. Shrink the vocab for the smoke ONLY; the
# pretrain_gemma4 builder keeps vocab_size_per_layer_input == vocab so the PLE
# table shrinks with it. The architecture (42 layers, heads, head_dim 256/512,
# softcap, sqrt(H) embed, PLE, KV bus) is unchanged. Override with VOCAB=262144.
VOCAB=${VOCAB:-32768}

# ---- smoke-run knobs (small so it fits 1 H100 from random init).
SEQLEN=${SEQLEN:-1024}
MBS=${MBS:-1}
GBS=${GBS:-4}
NUM_STEPS=${NUM_STEPS:-20}
TRAIN_SAMPLES=$((GBS * NUM_STEPS))

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
    --logging-level 20 \
    --log-throughput \
    --data-cache-path ${RUN_DIR}/data_cache \
    "

# NOTE: --untie-embeddings-and-output-weights is set for the SMOKE run only so
# the builder does not try to tie weights through a path that random-init does
# not need. Real (converted) E4B ties embeddings; drop this flag and the builder
# shares the embedding (the bridge owns the tied weight). The softcap + sqrt(H)
# embed scaling are independent of tying and remain on the loss path either way.

srun --container-image=$IMAGE \
     --container-mounts=/lustre:/lustre,/home/ataghibakhsh:/home/ataghibakhsh \
     --no-container-mount-home \
     bash -c "
set -x
export CUDA_DEVICE_MAX_CONNECTIONS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1
export TRITON_CACHE_DIR=${RUN_DIR}/triton_cache
export TRITON_HOME=${RUN_DIR}/triton_cache
export TORCH_COMPILE_DISABLE=1
export TORCHDYNAMO_DISABLE=1
# pretrain_gemma4.py applies the nvrx __version__ shim itself before importing
# megatron, so no preamble is needed. Put the gemma4-e4b MLM checkout first.
export PYTHONPATH=${MLM}:\${PYTHONPATH:-}
cd ${MLM}
echo '########## V7 gemma4 E4B training smoke ##########'
torchrun --nproc-per-node=1 --master-port=12399 ${MLM}/pretrain_gemma4.py ${options}
echo EXIT_TRAIN=\$?
echo '########## V7 DONE ##########'
"
