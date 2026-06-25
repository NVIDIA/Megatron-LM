#!/bin/bash
# ============================================================================
# Gemma 4 E4B INTERACTIVE load-and-train smoke (Megatron-LM).
#
# Run this *inside* an already-allocated interactive container shell, e.g.:
#
#   srun -A coreai_dlalgo_genai -p interactive --nodes=1 --gpus-per-node=8 \
#        --container-image=/lustre/fsw/portfolios/coreai/projects/coreai_nvfm_llm/containers/nemo-25.11.nemotron_3_nano-mamba_ssm_2.3.0.sqsh \
#        --container-mounts=/lustre:/lustre,/home/ataghibakhsh:/home/ataghibakhsh \
#        --no-container-mount-home --pty bash
#   # then, from the prompt inside the container:
#   bash examples/gemma4/gemma4_e4b_load_interactive.sh
#
# Unlike gemma4_e4b_interactive.sh (which is an sbatch + random-init smoke with a
# shrunk FFN/vocab), this:
#   * is a plain interactive script -- no SBATCH header, no srun wrapper, just
#     env + a direct torchrun, so it runs in the shell you already `srun --pty`'d into;
#   * LOADS the HF->MLM converted checkpoint (mlm_ckpt) with --finetune;
#   * uses the REAL E4B dims (FFN 10240, vocab 262144) so param shapes match the ckpt;
#   * has NO flextron / distillation / MoE / teacher args and no flex/KD logging.
# Data is still MOCK + NullTokenizer -- this is a "does it load + step without
# crashing, finite loss" smoke, not a real fine-tune.
# ============================================================================
set -eu

# ---- env (copied from the proven gemma4 V7 inner bash + reference interactive) --
unset SLURM_CPUS_PER_TASK SLURM_TRES_PER_TASK SLURM_CPU_BIND SLURM_DISTRIBUTION || true
export CUDA_DEVICE_MAX_CONNECTIONS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1
export TORCH_COMPILE_DISABLE=1
export TORCHDYNAMO_DISABLE=1
export TORCH_INDUCTOR_DISABLE=1

ROOT=/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_genai/users/ataghibakhsh/Gemma4_mlm
MLM=${ROOT}/Megatron-LM
IMPL=${ROOT}/code_dev/shared-state/implementations/HYBRID-gemma4-mlm
RUN_DIR=${ROOT}/code_dev/shared-state/load_run
mkdir -p ${RUN_DIR}/triton_cache ${RUN_DIR}/checkpoints ${RUN_DIR}/data_cache
export TRITON_CACHE_DIR=${RUN_DIR}/triton_cache
export TRITON_HOME=${RUN_DIR}/triton_cache

# ---- converted HF->MLM checkpoint. The converter saved a FLAT dist-checkpoint
# ---- (for its own parity check), which Megatron's --load cannot find. We point at
# ---- mlm_ckpt_mg/, a Megatron-layout view (latest_checkpointed_iteration.txt ->
# ---- "release", release/ -> symlinks to the distcp shards). Weights only, so
# ---- --finetune + --no-load-optim/rng. torch_dist reshards across TP on load.
LOAD_CKPT=${LOAD_CKPT:-${IMPL}/mlm_ckpt_mg}

# ---- parallelism. *** TP MUST BE 1 OR 2. *** Gemma4SelfAttention is an explicit
# ---- parallelism=1 eager port (see gemma4_attention.py docstring): its custom QKV
# ---- reshape does NOT implement the GQA KV-replication path the base attention uses
# ---- when num_query_groups (2) < TP. So TP>=4 produces wrong head sizes -> async
# ---- CUDA illegal-memory-access (surfaces as an NCCL error at the grad-norm
# ---- all_reduce). TP=2 is the max safe value (2 query groups split 1-per-rank).
# ---- PP is unsupported too (the cross-layer kv_bus assumes all layers on one stage).
# ---- To add GPUs for memory, scale DP (NPROC = TP * DP): the distributed optimizer
# ---- shards Adam state across DP. e.g. NPROC=8 TP=2 -> DP=4.
NPROC=${NPROC:-8}
TP=${TP:-2}
PP=${PP:-1}
CP=${CP:-1}
if [ "${TP}" -gt 2 ]; then
  echo "ERROR: TP=${TP} unsupported -- Gemma4SelfAttention is a parallelism<=2 port (num_query_groups=2). Use TP=1 or TP=2." >&2
  exit 1
fi

# ---- REAL Gemma 4 E4B dims -- must match the checkpoint param shapes. ----------
NUM_LAYERS=${NUM_LAYERS:-42}
HIDDEN=${HIDDEN:-2560}
FFN=${FFN:-10240}
HEADS=${HEADS:-8}
KV_GROUPS=${KV_GROUPS:-2}
VOCAB=${VOCAB:-262144}

# ---- smoke knobs ----------------------------------------------------------------
SEQLEN=${SEQLEN:-1024}
MBS=${MBS:-1}
GBS=${GBS:-4}
NUM_STEPS=${NUM_STEPS:-20}
TRAIN_SAMPLES=$((GBS * NUM_STEPS))

# NOTE: tied embeddings (real E4B) -> do NOT pass --untie-embeddings-and-output-weights.
# head_dim is heterogeneous (256/512) and set inside Gemma4TransformerConfig, so we
# intentionally do NOT pass --kv-channels (matches the convert-time config). Softcap +
# sqrt(H) embed scaling live in Gemma4Model.forward and are on the loss path.
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
    --attention-dropout 0.0 \
    --hidden-dropout 0.0 \
    --init-method-std 0.02 \
    --tensor-model-parallel-size ${TP} \
    --pipeline-model-parallel-size ${PP} \
    --context-parallel-size ${CP} \
    --seq-length ${SEQLEN} \
    --max-position-embeddings ${SEQLEN} \
    --micro-batch-size ${MBS} \
    --global-batch-size ${GBS} \
    --train-samples ${TRAIN_SAMPLES} \
    --lr 1e-5 \
    --min-lr 1e-6 \
    --lr-decay-style cosine \
    --lr-warmup-samples 0 \
    --weight-decay 0.0 \
    --clip-grad 1.0 \
    --adam-beta1 0.9 \
    --adam-beta2 0.95 \
    --bf16 \
    --use-distributed-optimizer \
    --load ${LOAD_CKPT} \
    --finetune \
    --ckpt-format torch_dist \
    --ckpt-fully-parallel-load \
    --no-load-optim \
    --no-load-rng \
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

export PYTHONPATH=${MLM}:${PYTHONPATH:-}
cd ${MLM}
echo "########## gemma4 E4B LOAD smoke (ckpt=${LOAD_CKPT}, NPROC=${NPROC} TP=${TP} PP=${PP}) ##########"
torchrun --nproc-per-node=${NPROC} --master-port=12399 ${MLM}/pretrain_gemma4.py ${options}
echo "EXIT_TRAIN=$?"
echo "########## DONE ##########"
