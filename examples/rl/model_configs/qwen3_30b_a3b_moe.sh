#!/bin/bash 

TP=${TP:-4}
PP=${PP:-1}
NODES_REQUIRED=${NODES_REQUIRED:-1}

echo "Using Qwen3-30B-A3B model checkpoint"
SCRIPT_PATH="${BASH_SOURCE[0]}"
source $(dirname $SCRIPT_PATH)/common.sh

# Default values
GRPO_CLAMP_EPS_LOWER=${GRPO_CLAMP_EPS_LOWER:-0.2}
GRPO_CLAMP_EPS_UPPER=${GRPO_CLAMP_EPS_UPPER:-0.2}
MAX_INFERENCE_BS=${MAX_INFERENCE_BS:-32}
GRPO_GROUP_SIZE=${GRPO_GROUP_SIZE:-16}
GRPO_PROMPTS_PER_STEP=${GRPO_PROMPTS_PER_STEP:-64}
GRPO_ITERATIONS=${GRPO_ITERATIONS:-1}
GRPO_KL_BETA=${GRPO_KL_BETA:-"0.0"}
TRAINING_BATCH_SIZE=${TRAINING_BATCH_SIZE:-256}
MICRO_BATCH_SIZE=${MICRO_BATCH_SIZE:-1}
MAX_SEQ_LENGTH=${MAX_SEQ_LENGTH:-8192}
EXIT_INTERVAL=${EXIT_INTERVAL:-20}
CHKPT_SAVE_INTERVAL=${CHKPT_SAVE_INTERVAL:-20}

ENV_DEPENDENT="\
  --micro-batch-size $MICRO_BATCH_SIZE \
  --global-batch-size $TRAINING_BATCH_SIZE \
  --grpo-group-size $GRPO_GROUP_SIZE \
  --grpo-prompts-per-step $GRPO_PROMPTS_PER_STEP \
  --grpo-iterations $GRPO_ITERATIONS \
  --grpo-clamp-eps-lower $GRPO_CLAMP_EPS_LOWER \
  --grpo-clamp-eps-upper $GRPO_CLAMP_EPS_UPPER \
  --grpo-kl-beta $GRPO_KL_BETA \
  --langrl-env-config $ENV_CONFIG "


MODEL_OPTIONS="
--seq-length $MAX_SEQ_LENGTH \
--inference-max-seq-length $MAX_SEQ_LENGTH \
--inference-max-requests $MAX_INFERENCE_BS \
--pretrained-checkpoint $CHECKPOINT \
--no-use-tokenizer-model-from-checkpoint-args \
--seq-length 8192 \
--inference-max-seq-length 8192 \
--bf16 \
--tensor-model-parallel-size $TP  \
--pipeline-model-parallel-size $PP  \
--expert-model-parallel-size $EP \
--attention-backend flash \
--transformer-impl transformer_engine \
--te-rng-tracker \
--tokenizer-type HuggingFaceTokenizer \
--tokenizer-model Qwen/Qwen3-30B-A3B \
--tokenizer-hf-include-special-tokens \
--untie-embeddings-and-output-weights \
--num-layers 48 \
--hidden-size 2048 \
--ffn-hidden-size 6144 \
--num-attention-heads 32 \
--kv-channels 128 \
--max-position-embeddings 8192 \
--group-query-attention \
--num-query-groups 4 \
--normalization RMSNorm \
--norm-epsilon 1e-6 \
--position-embedding-type rope \
--rotary-percent 1.0 \
--rotary-base 1000000 \
--use-rotary-position-embeddings \
--swiglu \
--disable-bias-linear \
--num-experts 128 \
--moe-router-topk 8 \
--moe-ffn-hidden-size 768 \
--moe-aux-loss-coeff 0.001 \
--moe-router-load-balancing-type aux_loss \
--attention-dropout 0.0 \
--hidden-dropout 0.0 \
--no-masked-softmax-fusion \
--attention-softmax-in-fp32 \
--vocab-size 151936 \
--make-vocab-size-divisible-by 128 \
--dist-ckpt-strictness log_unexpected \
--qk-layernorm \
--moe-token-dispatcher-type alltoall \
--moe-layer-freq 1 \
--optimizer adam \
--adam-beta1 0.9 \
--adam-beta2 0.999 \
--adam-eps 1e-8 \
--lr 1e-6 \
--min-lr 1e-7 \
--lr-warmup-samples 0 \
--clip-grad 1.0 \
--weight-decay 0.01 \
--no-load-optim \
--ckpt-format torch_dist
"
