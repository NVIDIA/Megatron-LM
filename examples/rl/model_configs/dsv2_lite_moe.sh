#!/bin/bash
TP=${TP:-2}
PP=${PP:-1}
EP=${EP:-1}
NODES_REQUIRED=${NODES_REQUIRED:-1}
LLM="dsv2_lite"

echo "Using Deepseek-v2-lite  model checkpoint (not the exact model weights..)"
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
  MAX_SEQ_LENGTH=${MAX_SEQ_LENGTH:-11999}
  EXIT_INTERVAL=${EXIT_INTERVAL:-16}
  CHKPT_SAVE_INTERVAL=${CHKPT_SAVE_INTERVAL:-16}
else
  # Some default values if config is unsupported.
  echo "Undected environment config, using default values"
  GRPO_CLAMP_EPS_LOWER=${GRPO_CLAMP_EPS_LOWER:-0.2}
  GRPO_CLAMP_EPS_UPPER=${GRPO_CLAMP_EPS_UPPER:-0.4}
  MAX_INFERENCE_BS=${MAX_INFERENCE_BS:-64}
  GRPO_GROUP_SIZE=${GRPO_GROUP_SIZE:-8}
  GRPO_PROMPTS_PER_STEP=${GRPO_PROMPTS_PER_STEP:-8}
  GRPO_ITERATIONS=${GRPO_ITERATIONS:-1}
  GRPO_KL_BETA=${GRPO_KL_BETA:-"0.0"}
  TRAINING_BATCH_SIZE=${TRAINING_BATCH_SIZE:-64}
  MICRO_BATCH_SIZE=${MICRO_BATCH_SIZE:-1}
  MAX_SEQ_LENGTH=${MAX_SEQ_LENGTH:-8192}
  EXIT_INTERVAL=${EXIT_INTERVAL:-20}
  CHKPT_SAVE_INTERVAL=${CHKPT_SAVE_INTERVAL:-20}
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


MODEL_OPTIONS="\
  --use-checkpoint-args \
  --enable-experimental \
  --cross-entropy-loss-fusion \
  --cross-entropy-fusion-impl native \
  --moe-aux-loss-coeff 0.0 \
  --moe-router-dtype fp64 \
  --moe-router-load-balancing-type none \
  --moe-token-dispatcher-type alltoall \
  --attention-backend flash \
  --disable-gloo-process-groups \
  --grpo-default-temperature 1.2 \
  --grpo-default-top-p 0.95 \
  --disable-chunked-prefill \
  --calculate-per-token-loss \
  --seq-length $MAX_SEQ_LENGTH \
  --inference-max-seq-length $MAX_SEQ_LENGTH \
  --inference-max-batch-size $MAX_INFERENCE_BS \
  --pretrained-checkpoint $CHECKPOINT \
  --distributed-timeout-minutes 60 \
  --use-mcore-models \
  --no-mmap-bin-files \
  --disable-bias-linear \
  --norm-epsilon 1e-5 \
  --init-method-std 0.014 \
  --exit-duration-in-mins 5750 \
  --max-position-embeddings 8192 \
  --tensor-model-parallel-size $TP  \
  --pipeline-model-parallel-size $PP  \
  --expert-model-parallel-size $EP \
  --no-masked-softmax-fusion \
  --attention-softmax-in-fp32 \
  --weight-decay 0.01 \
  --clip-grad 0.1 \
  --tiktoken-pattern v2 \
  --tokenizer-type TikTokenizer \
  --tokenizer-model ${TOKENIZER_MODEL} \
  --no-use-tokenizer-model-from-checkpoint-args \
  --dist-ckpt-strictness log_unexpected
  --ckpt-format torch_dist \
  --ckpt-fully-parallel-save \
  --ckpt-fully-parallel-load \
  --use-distributed-optimizer \
  --overlap-grad-reduce \
  --overlap-param-gather \
  --no-create-attention-mask-in-dataloader \
  --lr 1e-7 \
  --lr-warmup-samples 0 \
  --no-load-optim \
  --decode-only-cuda-graphs \
  --rl-inference-logprobs-is-correction \
  --rl-importance-sampling-truncation-coef 5.0 \
  "

# 1. remove importance sampling


# 2. removed any form of load balancing loss