#!/bin/bash
TP=${TP:-8}
PP=${PP:-1}
NODES_REQUIRED=${NODES_REQUIRED:-4}
LLM="llama3p1_8b_instruct"
EXTRAS=""

echo "Using Llama 3.1 8B Instruct model checkpoint"
SCRIPT_PATH="${BASH_SOURCE[0]}"
source $(dirname $SCRIPT_PATH)/common.sh

# In all cases, one can override those values.
# However, running without envs will give you some
# good perf out of the box for established envs.
if [ "$(basename "$ENV_CONFIG")" = "dapo.yaml" ]; then
  echo "Using DAPO environment config"
  GRPO_CLAMP_EPS_LOWER=${GRPO_CLAMP_EPS_LOWER:-0.2}
  GRPO_CLAMP_EPS_UPPER=${GRPO_CLAMP_EPS_UPPER:-0.2}
  MAX_INFERENCE_BS=${MAX_INFERENCE_BS:-64}
  GRPO_GROUP_SIZE=${GRPO_GROUP_SIZE:-16}
  GRPO_PROMPTS_PER_STEP=${GRPO_PROMPTS_PER_STEP:-64}
  GRPO_ITERATIONS=${GRPO_ITERATIONS:-1}
  GRPO_KL_BETA=${GRPO_KL_BETA:-"0.0"}
  ENTROPY_WEIGHT=${ENTROPY_WEIGHT:-"0.0"}
  TRAINING_BATCH_SIZE=${TRAINING_BATCH_SIZE:-1024}
  MICRO_BATCH_SIZE=${MICRO_BATCH_SIZE:-2}
  MAX_SEQ_LENGTH=${MAX_SEQ_LENGTH:-8192}
  EXIT_INTERVAL=${EXIT_INTERVAL:-16}
  CHKPT_SAVE_INTERVAL=${CHKPT_SAVE_INTERVAL:-16}
elif [ "$(basename "$ENV_CONFIG")" = "openmathinstructv2.yaml" ]; then
  GRPO_CLAMP_EPS_LOWER=${GRPO_CLAMP_EPS_LOWER:-0.2}
  GRPO_CLAMP_EPS_UPPER=${GRPO_CLAMP_EPS_UPPER:-0.2}
  MAX_INFERENCE_BS=${MAX_INFERENCE_BS:-64}
  GRPO_GROUP_SIZE=${GRPO_GROUP_SIZE:-32}
  GRPO_PROMPTS_PER_STEP=${GRPO_PROMPTS_PER_STEP:-64}
  GRPO_ITERATIONS=${GRPO_ITERATIONS:-1}
  GRPO_KL_BETA=${GRPO_KL_BETA:-"0.01"}
  ENTROPY_WEIGHT=${ENTROPY_WEIGHT:-"0.0"}
  TRAINING_BATCH_SIZE=${TRAINING_BATCH_SIZE:-512}
  MICRO_BATCH_SIZE=${MICRO_BATCH_SIZE:-1}
  MAX_SEQ_LENGTH=${MAX_SEQ_LENGTH:-4096}
  EXIT_INTERVAL=${EXIT_INTERVAL:-16}
  CHKPT_SAVE_INTERVAL=${CHKPT_SAVE_INTERVAL:-16}
  EXTRAS="--lr-warmup-samples 5120"
else
  # Some default values if config is missing.
  echo "Undected environment config, using default values"
  GRPO_CLAMP_EPS_LOWER=${GRPO_CLAMP_EPS_LOWER:-0.2}
  GRPO_CLAMP_EPS_UPPER=${GRPO_CLAMP_EPS_UPPER:-0.2}
  MAX_INFERENCE_BS=${MAX_INFERENCE_BS:-32}
  MAX_SEQ_LENGTH=${MAX_SEQ_LENGTH:-4096}
  GRPO_GROUP_SIZE=${GRPO_GROUP_SIZE:-32}
  GRPO_PROMPTS_PER_STEP=${GRPO_PROMPTS_PER_STEP:-64}
  GRPO_KL_BETA=${GRPO_KL_BETA:-"0.01"}
  ENTROPY_WEIGHT=${ENTROPY_WEIGHT:-"0.0"}
  GRPO_ITERATIONS=${GRPO_ITERATIONS:-1}
  TRAINING_BATCH_SIZE=${TRAINING_BATCH_SIZE:-512}
  MICRO_BATCH_SIZE=${MICRO_BATCH_SIZE:-1}
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
  --grpo-entropy-term-weight $ENTROPY_WEIGHT \
  --langrl-env-config $ENV_CONFIG "

MODEL_OPTIONS="\
  --disable-bias-linear \
  --ckpt-format torch_dist \
  --seq-length $MAX_SEQ_LENGTH \
  --inference-max-seq-length $MAX_SEQ_LENGTH \
  --inference-max-batch-size $MAX_INFERENCE_BS \
  --pretrained-checkpoint $CHECKPOINT \
  --add-qkv-bias \
  --normalization RMSNorm \
  --group-query-attention \
  --num-query-groups 8 \
  --no-masked-softmax-fusion \
  --attention-softmax-in-fp32 \
  --attention-dropout 0.0 \
  --hidden-dropout 0.0 \
  --untie-embeddings-and-output-weights \
  --weight-decay 0.1 \
  --position-embedding-type rope \
  --rotary-percent 1.0 \
  --rotary-base 500000 \
  --use-rotary-position-embeddings \
  --swiglu \
  --num-layers 32  \
  --hidden-size 4096  \
  --ffn-hidden-size 14336 \
  --num-attention-heads 32  \
  --max-position-embeddings 131072  \
  --tokenizer-type HuggingFaceTokenizer \
  --tokenizer-model unsloth/Meta-Llama-3.1-8B-Instruct \
  --langrl-inference-server-type "inplace_megatron_chat" \
  --langrl-inference-server-conversation-template "unsloth/Meta-Llama-3.1-8B-Instruct" \
  --lr 3e-7 \
  --make-vocab-size-divisible-by 128 \
  --clip-grad 1.0 \
  --rl-use-sequence-packing \
  --rl-sequence-packing-algo fifo \
  $EXTRAS"

