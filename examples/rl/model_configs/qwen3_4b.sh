#!/bin/bash

TP=${TP:-1}
PP=${PP:-1}
NODES_REQUIRED=${NODES_REQUIRED:-1}

echo "Using Qwen3 4B model checkpoint"
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
MAX_SEQ_LENGTH=${MAX_SEQ_LENGTH:-32768}
EXIT_INTERVAL=${EXIT_INTERVAL:-16}
CHKPT_SAVE_INTERVAL=${CHKPT_SAVE_INTERVAL:-16}

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

# Model configuration based on MegatronBridge run_config.yaml
MODEL_OPTIONS="\
  --ckpt-format torch_dist \
  --seq-length $MAX_SEQ_LENGTH \
  --inference-max-seq-length $MAX_SEQ_LENGTH \
  --inference-max-batch-size $MAX_INFERENCE_BS \
  --pretrained-checkpoint $CHECKPOINT \
  --num-layers 36 \
  --hidden-size 2560 \
  --ffn-hidden-size 9728 \
  --num-attention-heads 32 \
  --kv-channels 128 \
  --max-position-embeddings 40960 \
  --group-query-attention \
  --num-query-groups 8 \
  --normalization RMSNorm \
  --norm-epsilon 1e-6 \
  --qk-layernorm \
  --position-embedding-type rope \
  --rotary-percent 1.0 \
  --rotary-base 1000000 \
  --use-rotary-position-embeddings \
  --swiglu \
  --disable-bias-linear \
  --attention-dropout 0.0 \
  --hidden-dropout 0.0 \
  --no-masked-softmax-fusion \
  --attention-softmax-in-fp32 \
  --tokenizer-type HuggingFaceTokenizer \
  --tokenizer-model Qwen/Qwen3-4B \
  --langrl-inference-server-type "inplace_megatron_chat" \
  --langrl-inference-server-conversation-template "Qwen/Qwen3-4B" \
  --vocab-size 151936 \
  --make-vocab-size-divisible-by 128 \
  --optimizer adam \
  --adam-beta1 0.9 \
  --adam-beta2 0.999 \
  --adam-eps 1e-8 \
  --lr 1e-6 \
  --min-lr 1e-7 \
  --lr-warmup-samples 0 \
  --clip-grad 1.0 \
  --weight-decay 0.01 \
  --recompute-granularity selective \
  --recompute-activations \
  --recompute-modules core_attn \
  "

