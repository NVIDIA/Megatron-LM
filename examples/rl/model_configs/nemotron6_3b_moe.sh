#!/bin/bash
TP=${TP:-2}
PP=${PP:-1}
EP=${EP:-32}
NODES_REQUIRED=${NODES_REQUIRED:-4}
LLM="nemotron6_3b_moe"

echo "Using Nemotron6 3B MOE model checkpoint"
SCRIPT_PATH="${BASH_SOURCE[0]}"
source $(dirname $SCRIPT_PATH)/common.sh


echo "Undected environment config, using default values"
GRPO_CLAMP_EPS_LOWER=${GRPO_CLAMP_EPS_LOWER:-0.2}
GRPO_CLAMP_EPS_UPPER=${GRPO_CLAMP_EPS_UPPER:-0.28}
MAX_INFERENCE_BS=${MAX_INFERENCE_BS:-64}
GRPO_GROUP_SIZE=${GRPO_GROUP_SIZE:-16}
GRPO_PROMPTS_PER_STEP=${GRPO_PROMPTS_PER_STEP:-64}
GRPO_ITERATIONS=${GRPO_ITERATIONS:-1}
GRPO_KL_BETA=${GRPO_KL_BETA:-"0.0"}
TRAINING_BATCH_SIZE=${TRAINING_BATCH_SIZE:-1024}
MICRO_BATCH_SIZE=${MICRO_BATCH_SIZE:-1}
MAX_SEQ_LENGTH=${MAX_SEQ_LENGTH:-8192}
EXIT_INTERVAL=${EXIT_INTERVAL:-15}
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

MODEL_OPTIONS="\
  --no-use-tokenizer-model-from-checkpoint-args \
  --rl-skip-bos-token \
  --rl-use-sequence-packing \
  --rl-partial-rollouts \
  --moe-pad-experts-for-cuda-graph-inference \
  --inference-dynamic-batching-num-cuda-graphs 4 \
  --inference-dynamic-batching-max-requests 128 \
  --inference-dynamic-batching-paused-buffer-size-gb 5 \
  --inference-dynamic-batching-buffer-size-gb 5 \
  --inference-dynamic-batching-unified-memory-level 1 \
  --rl-training-cuda-graphs \
  --empty-unused-memory-level 0 \
  --rl-parallel-generation-tasks 128 \
  --inference-dynamic-batching-cuda-graph-mixed-prefill-count 0 \
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
  --tokenizer-type HuggingFaceTokenizer \
  --tokenizer-model ${TOKENIZER_MODEL} \
  --tokenizer-hf-include-special-tokens \
  --dist-ckpt-strictness log_unexpected \
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
  --moe-permute-fusion \
  --eval-interval 1000 \
  --timing-log-level 2 \
  "
