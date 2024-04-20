#!/bin/bash

if [ $# -ne 1 ]; then
    echo "1 arguments are required: wandb_run_name."
    echo "Usage: $0 wandb_run_name"
    exit 1
fi

# 必要な環境変数がちゃんとあるかチェックする
echo "> Check env vars..."
required_env_vars=(\
 "WANDB_PROJECT" "WANDB_API_KEY" "HF_TOKEN"\
 "MASTER_ADDR" "MASTER_PORT" "NNODES" "NODE_RANK"\
 "CHECKPOINT_LOCAL_PATH" "CHECKPOINT_PATH" "LOAD_CHECKPOINT_PATH"\
 "TOKENIZER_MODEL" "DATA_PATH"\
 "TMP_SIZE" "PMP_SIZE"\
)

for var in "${required_env_vars[@]}"; do
  if [ -z "${!var}" ]; then
    echo "ERROR: env $var is not set." >&2
    exit 1
  fi
done

RUN_NAME=$1
echo "START RUN: $RUN_NAME"

export CUDA_DEVICE_MAX_CONNECTIONS=1
export GPUS_PER_NODE=8
WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES))
ITERATION=10000
VOCAB_SIZE_DIVISIBLE_BY=$((128/$TMP_SIZE*2))


GPT_ARGS="
    --tensor-model-parallel-size $TMP_SIZE \
    --pipeline-model-parallel-size $PMP_SIZE \
    --num-layers 2 \
    --hidden-size 256 \
    --ffn-hidden-size 14336 \
    --num-attention-heads 2 \
    --group-query-attention \
    --num-query-groups 2 \
    --seq-length 4096 \
    --max-position-embeddings 32768 \
    --micro-batch-size 1 \
    --global-batch-size 8 \
    --lr 0.00015 \
    --train-iters $ITERATION \
    --lr-decay-iters 320000 \
    --lr-decay-style cosine \
    --min-lr 1.0e-5 \
    --weight-decay 1e-2 \
    --lr-warmup-fraction .01 \
    --clip-grad 1.0 \
    --fp16 \
    --swiglu \
    --disable-bias-linear \
    --normalization RMSNorm \
    --use-rotary-position-embeddings \
    --untie-embeddings-and-output-weights \
    --num-experts 8 \
    --moe-type mixtral \
    --tokenizer-type HFTokenizer \
    --tokenizer-model $TOKENIZER_MODEL \
    --no-load-optim \
    --no-load-rng \
    --no-masked-softmax-fusion \
    --use-flash-attn \
    --make-vocab-size-divisible-by $VOCAB_SIZE_DIVISIBLE_BY
"


DATA_ARGS="
    --data-path $DATA_PATH \
    --split 949,50,1
"

OUTPUT_ARGS="
    --log-interval 100 \
    --save-interval 1000 \
    --eval-interval 200 \
    --eval-iters 10
"

LOG_ARGS="
    --wandb-project $WANDB_PROJECT \
    --wandb-exp-name node_${NODE_RANK} \
    --wandb-group-name $RUN_NAME \
    --wandb-entity abeja-geniac \
    --tensorboard-dir ./training_results \
    --wandb-save-dir ./training_results
"

torchrun $DISTRIBUTED_ARGS pretrain_gpt.py \
    $GPT_ARGS \
    $DATA_ARGS \
    $OUTPUT_ARGS \
    $LOG_ARGS \
    --save $CHECKPOINT_LOCAL_PATH \
    # --load $LOAD_CHECKPOINT_PATH

./script/sync_models.sh $ITERATION
