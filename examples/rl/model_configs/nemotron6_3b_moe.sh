#!/bin/bash
TP=${TP:-2}
PP=${PP:-1}
EP=${EP:-32}
NODES_REQUIRED=${NODES_REQUIRED:-4}
LLM="nemotron6_3b_moe"

echo "Using Nemotron6 3B MOE model checkpoint"
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
  EXIT_INTERVAL=${EXIT_INTERVAL:-20}
  CHKPT_SAVE_INTERVAL=${CHKPT_SAVE_INTERVAL:-20}
else
  # Some default values if config is unsupported.
  echo "Undected environment config, using default values"
  GRPO_CLAMP_EPS_LOWER=${GRPO_CLAMP_EPS_LOWER:-0.2}
  GRPO_CLAMP_EPS_UPPER=${GRPO_CLAMP_EPS_UPPER:-0.28}
  MAX_INFERENCE_BS=${MAX_INFERENCE_BS:-64}
  GRPO_GROUP_SIZE=${GRPO_GROUP_SIZE:-2}
  GRPO_PROMPTS_PER_STEP=${GRPO_PROMPTS_PER_STEP:-16}
  GRPO_ITERATIONS=${GRPO_ITERATIONS:-1}
  GRPO_KL_BETA=${GRPO_KL_BETA:-"0.0"}
  TRAINING_BATCH_SIZE=${TRAINING_BATCH_SIZE:-32}
  MICRO_BATCH_SIZE=${MICRO_BATCH_SIZE:-1}
  MAX_SEQ_LENGTH=${MAX_SEQ_LENGTH:-1024}
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
  --rl-skip-bos-token \
  --no-rl-use-sequence-packing \
  --rl-partial-rollouts \
  --moe-pad-experts-for-cuda-graph-inference \
  --inference-dynamic-batching-max-tokens 8192 \
  --inference-dynamic-batching-max-requests 128 \
  --inference-dynamic-batching-num-cuda-graphs 2 \
  --decode-only-cuda-graphs \
  --cuda-graph-impl local \
  --cuda-graph-scope full \
  --use-checkpoint-args \
  --enable-experimental \
  --cross-entropy-loss-fusion \
  --cross-entropy-fusion-impl native \
  --moe-aux-loss-coeff 0.0 \
  --moe-router-dtype fp64 \
  --moe-router-load-balancing-type aux_loss \
  --moe-router-score-function sigmoid \
  --moe-token-dispatcher-type alltoall \
  --moe-router-enable-expert-bias \
  --moe-router-topk-scaling-factor 2.5 \
  --disable-gloo-process-groups \
  --rl-default-top-k -1 \
  --rl-default-temperature 1.0 \
  --rl-default-top-p 1.0 \
  --rl-inference-logprobs-is-correction \
  --rl-importance-sampling-truncation-coef 10.0 \
  --seq-length $MAX_SEQ_LENGTH \
  --inference-max-seq-length $MAX_SEQ_LENGTH \
  --inference-max-requests $MAX_INFERENCE_BS \
  --pretrained-checkpoint $CHECKPOINT \
  --distributed-timeout-minutes 60 \
  --use-mcore-models \
  --no-mmap-bin-files \
  --disable-bias-linear \
  --norm-epsilon 1e-5 \
  --init-method-std 0.014 \
  --exit-duration-in-mins 5750 \
  --max-position-embeddings $MAX_SEQ_LENGTH \
  --tensor-model-parallel-size $TP  \
  --pipeline-model-parallel-size $PP  \
  --expert-model-parallel-size $EP \
  --expert-tensor-parallel-size 1 \
  --weight-decay 0.01 \
  --clip-grad 1.0 \
  --tiktoken-pattern v2 \
  --tokenizer-type TikTokenizer \
  --tokenizer-model ${TOKENIZER_MODEL} \
  --dist-ckpt-strictness log_unexpected
  --ckpt-format torch_dist \
  --ckpt-fully-parallel-save \
  --ckpt-fully-parallel-load \
  --use-distributed-optimizer \
  --overlap-grad-reduce \
  --overlap-param-gather \
  --no-create-attention-mask-in-dataloader \
  --lr 3e-6 \
  --min-lr 3e-6 \
  --lr-decay-style constant \
  --lr-warmup-samples 640 \
  --lr-warmup-init 0.3e-7 \
  --no-load-optim \
  --no-load-rng \
  "
