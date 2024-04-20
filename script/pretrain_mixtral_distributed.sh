#!/bin/bash

# arguments のチェック
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
 "CHECKPOINT_LOCAL_PATH" "CHECKPOINT_PATH" "LOAD_CHECKPOINT_PATH" "TOKENIZER_MODEL"\
 "TMP_SIZE" "PMP_SIZE"\
)

for var in "${required_env_vars[@]}"; do
  if [ -z "${!var}" ]; then
    echo "ERROR: env $var is not set." >&2
    exit 1
  fi
done

# checkpoint_local_path が wandb のディレクトリになっていないかチェックする 
if [[ $CHECKPOINT_LOCAL_PATH == ./training_results* ]]; then
    echo "CHECKPOINT_LOCAL_PATH を wandb のディレクトリにすると wandb dir が消えるため失敗します"
    exit 1
fi

RUN_NAME=$1
echo "START RUN: $RUN_NAME"

# 主要な変数の定義
export CUDA_DEVICE_MAX_CONNECTIONS=1
export GPUS_PER_NODE=8
WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES))
ITERATION=107250
VOCAB_SIZE_DIVISIBLE_BY=$TMP_SIZE

# Mixtral からの継続学習か、学習途中からの再開かで引数を変える
if [[ $LOAD_CHECKPOINT_PATH == /mnt/nfs/models* ]]; then
  LOAD_CHECKPOINT_ARGS="
    --no-load-optim \
    --no-load-rng"
else
  LOAD_CHECKPOINT_ARGS=""
fi

DISTRIBUTED_ARGS="
    --nproc_per_node $GPUS_PER_NODE \
    --nnodes $NNODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT
"

GPT_ARGS="
    --tensor-model-parallel-size $TMP_SIZE \
    --pipeline-model-parallel-size $PMP_SIZE \
    --num-layers 32 \
    --hidden-size 4096 \
    --ffn-hidden-size 14336 \
    --num-attention-heads 32 \
    --group-query-attention \
    --num-query-groups 8 \
    --seq-length 4096 \
    --max-position-embeddings 32768 \
    --micro-batch-size 2 \
    --global-batch-size 1020 \
    --lr 4e-5 \
    --train-iters $ITERATION \
    --lr-decay-iters $ITERATION \
    --lr-decay-style cosine \
    --min-lr 2.0e-6 \
    --weight-decay 0.1 \
    --lr-warmup-fraction .01 \
    --clip-grad 1.0 \
    --bf16 \
    --use-flash-attn \
    --swiglu \
    --disable-bias-linear \
    --normalization RMSNorm \
    --use-rotary-position-embeddings \
    --untie-embeddings-and-output-weights \
    --num-experts 8 \
    --moe-type mixtral \
    --tokenizer-type HFTokenizer \
    --tokenizer-model $TOKENIZER_MODEL \
    --no-masked-softmax-fusion \
    --make-vocab-size-divisible-by $VOCAB_SIZE_DIVISIBLE_BY \
    --no-check-for-nan-in-loss-and-grad \
    --router-aux-loss-coef 0.02 \
    --use-distributed-optimizer
"

# weight path の形式でデータを指定
DATA_PATHS="
1.0 /mnt/nfs/data/xxx \
"

DATA_ARGS="
    --data-path $DATA_PATHS \
    --split 995,5,0
"

OUTPUT_ARGS="
    --log-interval 100 \
    --save-interval 1000 \
    --eval-interval 1000 \
    --eval-iters 100
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
    $LOAD_CHECKPOINT_ARGS \
    --distributed-backend nccl \
    --save $CHECKPOINT_LOCAL_PATH \
    --load $LOAD_CHECKPOINT_PATH
