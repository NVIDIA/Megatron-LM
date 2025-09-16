# Reinforcement Learning in megatron

This is an example of GRPO implementation within megatron-lm.
For implementation details check out `train_rl.py` and `megatron/rl/rl_utils.py`.
For the environment details, check the `megatron.rl` module.

The following experiment will train the Qwen 2.5 32B model on the [DAPO17k](https://huggingface.co/datasets/BytedTsinghua-SIA/DAPO-Math-17k) dataset and will run evaluation on [AIME2024](https://huggingface.co/datasets/Maxwell-Jia/AIME_2024).
After 300 steps, you should get about 0.7 pass@32 on AIME with the average training reward of 0.6. 

## Setup

You should be able to run qwen2p5_32b_grpo.sh using the `nvcr.io/nvidia/pytorch:25.06-py3` container with these additional dependencies:

```bash
pip install flask-restful uvloop datasets evaluate
```

Specify these environment variables and create the required directories:

```bash
export CUDA_DEVICE_MAX_CONNECTIONS=1

CHECKPOINT="" # <Specify path to the base model checkpoint>
RUN_DIR="" # <Specify path for bookkeeping>
WANDB_PROJECT="" # <Specify>
WANDB_EXP_NAME="" # <Specify>

LOG_DIR=$RUN_DIR/logs
DATA_CACHE_DIR=$RUN_DIR/data_cache
CHECKPOINT_DIR=$RUN_DIR/checkpoints
TB_DIR=$RUN_DIR/tensorboard
```

## Convert the checkpoint

You can convert a [Huggingface Qwen checkpoint](https://huggingface.co/Qwen/Qwen2.5-32B) to megatron-lm format using the `megatron-lm/tools/checkpoint/convert.py` script:

```bash
TP=8
HF_FORMAT_DIR=<PATH_TO_HF_FORMAT_DIR>
MEGATRON_FORMAT_DIR=<PATH_TO_MEGATRON_FORMAT_DIR>
TOKENIZER_MODEL=HF_FORMAT_DIR

python ./tools/checkpoint/convert.py \
    --bf16 \
    --model-type GPT \
    --loader llama_mistral \
    --saver core \
    --target-tensor-parallel-size ${TP} \
    --checkpoint-type hf \
    --load-dir ${HF_FORMAT_DIR} \
    --save-dir ${MEGATRON_FORMAT_DIR} \
    --tokenizer-model ${TOKENIZER_MODEL} \
    --model-size qwen2.5 \
    --loader-transformer-impl transformer_engine \
    --make-vocab-size-divisible-by 128 \
```

## Experiment command

NOTE: Depending on the environment you are running it the provided script might require minor changes.

```bash

COMMON_OPTIONS="\
    --tensor-model-parallel-size $TP  \
    --pipeline-model-parallel-size $PP  \
    --use-mcore-models \
    --transformer-impl transformer_engine \
    --bf16 \
    --te-rng-tracker \
    --enable-cuda-graph \
    --inference-dynamic-batching-num-cuda-graphs 1 \
    --inference-dynamic-batching-buffer-size-gb 20 \
    --data-parallel-random-init \
    --attention-backend flash \
    --timing-log-level 1 \
    --log-timers-to-tensorboard \
    --initialize-socket-comms \
    "

GRPO_CLAMP_EPS_LOWER=0.2
GRPO_CLAMP_EPS_UPPER=0.28
MAX_INFERENCE_BS=32
GRPO_GROUP_SIZE=16
GRPO_PROMPTS_PER_STEP=64
GRPO_ITERATIONS=1
GRPO_KL_BETA="0.0"
TRAINING_BATCH_SIZE=1024
MICRO_BATCH_SIZE=1
MAX_SEQ_LENGTH=11999

MODEL_OPTIONS="\
  --ckpt-format torch \
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

ENV_DEPENDENT="\
  --langrl-env-config "examples/rl/environment_configs/dapo.yaml" \
  --micro-batch-size $MICRO_BATCH_SIZE \
  --global-batch-size $TRAINING_BATCH_SIZE \
  --grpo-group-size $GRPO_GROUP_SIZE \
  --grpo-prompts-per-step $GRPO_PROMPTS_PER_STEP \
  --grpo-iterations $GRPO_ITERATIONS \
  --grpo-clamp-eps-lower $GRPO_CLAMP_EPS_LOWER \
  --grpo-clamp-eps-upper $GRPO_CLAMP_EPS_UPPER \
  --grpo-kl-beta $GRPO_KL_BETA \
  --env-config $ENV_CONFIG "


torchrun \
    --nproc-per-node=8 \
    --nnodes=8 \
    train_rl.py \
    --mock-data \
    --distributed-timeout-minutes 60 \
    --train-samples 48828125 \
    --log-interval 10 \
    --log-progress  \
    --timing-log-option minmax \
    --log-params-norm \
    --log-num-zeros-in-grad \
    --log-throughput \
    --adam-beta1 0.9 \
    --adam-beta2 0.95 \
    --adam-eps 1e-8 \
    --no-create-attention-mask-in-dataloader \
    --accumulate-allreduce-grads-in-fp32 \
    --calculate-per-token-loss \
    --log-straggler \
    --disable-straggler-on-startup \
    --perform-rl-step \
    --use-distributed-optimizer \
    --straggler-minmax-count 16 \
    --eval-interval 20 \
    --rl-prompts-per-eval 32 \
    --tensorboard-log-interval 1 \
    --empty-unused-memory-level 2 \
    --data-cache-path ${DATA_CACHE_DIR} \
    --save $CHECKPOINT_DIR \
    --load $CHECKPOINT_DIR \
    --tensorboard-dir $TB_DIR \
    --langrl-inference-server-type inplace_megatron \
    --seed $SEED \
    --sequence-parallel \
    --finetune \
    --save-interval 20 \
    --wandb-project $WANDB_PROJECT \
    --wandb-exp-name $WANDB_EXP_NAME \
    ${MODEL_OPTIONS} \
    ${COMMON_OPTIONS} \
    ${ENV_DEPENDENT} $@
```
