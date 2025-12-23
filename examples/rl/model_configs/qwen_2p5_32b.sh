#!/bin/bash
TP=${TP:-8}
PP=${PP:-1}
NODES_REQUIRED=${NODES_REQUIRED:-8}
LLM="qwen2p5_32b"

echo "Using Qwen 2.5 32B model checkpoint"
SCRIPT_PATH="${BASH_SOURCE[0]}"
source $(dirname $SCRIPT_PATH)/common.sh

# In all cases, one can override those values.
# However, running without envs will give you some
# good perf out of the box for established envs.
if [ "$(basename "$ENV_CONFIG")" = "dapo.yaml" ]; then
  echo "Using DAPO environment config"
  GRPO_CLAMP_EPS_LOWER=${GRPO_CLAMP_EPS_LOWER:-0.2}
  GRPO_CLAMP_EPS_UPPER=${GRPO_CLAMP_EPS_UPPER:-0.28}
  MAX_INFERENCE_BS=${MAX_INFERENCE_BS:-32}
  GRPO_GROUP_SIZE=${GRPO_GROUP_SIZE:-16}
  GRPO_PROMPTS_PER_STEP=${GRPO_PROMPTS_PER_STEP:-64}
  GRPO_ITERATIONS=${GRPO_ITERATIONS:-1}
  GRPO_KL_BETA=${GRPO_KL_BETA:-"0.0"}
  TRAINING_BATCH_SIZE=${TRAINING_BATCH_SIZE:-1024}
  MICRO_BATCH_SIZE=${MICRO_BATCH_SIZE:-1}
  MAX_SEQ_LENGTH=${MAX_SEQ_LENGTH:-12000}
  EXIT_INTERVAL=${EXIT_INTERVAL:-16}
  CHKPT_SAVE_INTERVAL=${CHKPT_SAVE_INTERVAL:-16}
else
  # Some default values if config is unsupported.
  echo "Undected environment config, using default values"
  GRPO_CLAMP_EPS_LOWER=${GRPO_CLAMP_EPS_LOWER:-0.2}
  GRPO_CLAMP_EPS_UPPER=${GRPO_CLAMP_EPS_UPPER:-0.2}
  MAX_INFERENCE_BS=${MAX_INFERENCE_BS:-64}
  GRPO_GROUP_SIZE=${GRPO_GROUP_SIZE:-16}
  GRPO_PROMPTS_PER_STEP=${GRPO_PROMPTS_PER_STEP:-32}
  GRPO_ITERATIONS=${GRPO_ITERATIONS:-1}
  GRPO_KL_BETA=${GRPO_KL_BETA:-"0.0"}
  TRAINING_BATCH_SIZE=${TRAINING_BATCH_SIZE:-512}
  MICRO_BATCH_SIZE=${MICRO_BATCH_SIZE:-1}
  MAX_SEQ_LENGTH=${MAX_SEQ_LENGTH:-12000}
  EXIT_INTERVAL=${EXIT_INTERVAL:-16}
  CHKPT_SAVE_INTERVAL=${CHKPT_SAVE_INTERVAL:-16}
fi

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

# Original Qwen model uses a wrong padding_id token. unsloth tokenizer fixes it.
MODEL_OPTIONS="\
  --calculate-per-token-loss \
  --ckpt-format torch_dist \
  --seq-length $MAX_SEQ_LENGTH \
  --inference-max-seq-length $MAX_SEQ_LENGTH \
  --inference-max-batch-size $MAX_INFERENCE_BS \
  --pretrained-checkpoint $CHECKPOINT \
  --untie-embeddings-and-output-weights \
  --disable-bias-linear \
  --add-qkv-bias \
  --normalization RMSNorm \
  --norm-epsilon 1e-5 \
  --group-query-attention \
  --num-query-groups 8 \
  --no-masked-softmax-fusion \
  --attention-softmax-in-fp32 \
  --attention-dropout 0.0 \
  --hidden-dropout 0.0 \
  --weight-decay 0.0 \
  --position-embedding-type rope \
  --rotary-percent 1.0 \
  --rotary-base 1000000 \
  --use-rotary-position-embeddings \
  --swiglu \
  --num-layers 64  \
  --hidden-size 5120  \
  --ffn-hidden-size 27648 \
  --num-attention-heads 40  \
  --max-position-embeddings 131072 \
  --tokenizer-type HuggingFaceTokenizer \
  --tokenizer-model unsloth/Qwen2.5-32B \
  --lr 1e-6 \
  --lr-warmup-samples 0 \
  --make-vocab-size-divisible-by 128 \
  --clip-grad 1.0 \
  --recompute-granularity selective \
  --recompute-activations "
