#!/bin/bash

print_help_and_exit() { echo -e "Usage:\n\t$0 MODEL_PATH TOKENIZER_PATH {7B|13B} [PROMPT_TXT]"; exit 1; }
[[ "$#" -lt 3 || "$#" -gt 4 ]] && print_help_and_exit

export CUDA_DEVICE_MAX_CONNECTIONS=1

MODEL_SIZE=$1  # Either 7B or 13B
MODEL_PATH=$2  # Path to a model in Megatron format
TOKENIZER_MODEL_PATH=$3  # Path to tokenizer.model
PROMPT_FILE=$4  # A textfile with a single prompt

: ${DATA_BLEND:="1.0 datasets/dmc_demo/llama2_tokenized/hf_wiki_20231101_en_train_text_document"}
: ${DATA_CACHE:="datasets/dmc_demo/cache_llama2"}

ARGS=" \
    --use-flash-attn \
    --seed 1234 \
    --tokenizer-model $TOKENIZER_MODEL_PATH \
    --load $MODEL_PATH \
    --global-batch-size 32 \
    --micro-batch-size 2 \
    --untie-embeddings-and-output-weights \
    --disable-bias-linear \
    --no-position-embedding \
    --use-rotary-position-embeddings \
    --rotary-percent 0.5 \
    --attention-dropout 0.0 \
    --hidden-dropout 0.0 \
    --exit-duration-in-mins 225 \
    --tensor-model-parallel-size 1 \
    --pipeline-model-parallel-size 1 \
    --seq-length 4096 \
    --max-position-embeddings 4096 \
    --train-samples 60000000 \
    --lr-decay-samples 13722 \
    --lr-warmup-samples 0 \
    --lr 3.0e-5 \
    --min-lr 3.0e-5 \
    --lr-decay-style cosine \
    --log-interval 100 \
    --eval-iters 10 \
    --eval-interval 2000 \
    --tokenizer-type Llama2Tokenizer \
    --save-interval 1000 \
    --save ../ \
    --finetune \
    --split 99,1,0 \
    --clip-grad 1.0 \
    --weight-decay 0.1 \
    --adam-beta1 0.9 \
    --adam-beta2 0.95 \
    --log-params-norm \
    --log-num-zeros-in-grad \
    --use-checkpoint-args \
    --no-load-optim \
    --no-load-rng \
    --normalization RMSNorm \
    --no-masked-softmax-fusion \
    --bf16 \
    --skip-train \
    --tensorboard-dir ../tensorboard
"

if [ "$MODEL_SIZE" == "7B" ]; then
    ARGS+=" --num-layers 32 --num-attention-heads 32 --hidden-size 4096 "
elif [ "$MODEL_SIZE" == "13B" ]; then
    ARGS+=" --num-layers 40 --num-attention-heads 40 --hidden-size 5120 "
else
    print_help_and_exit
fi

GENERATE_ARGS="--generate-len 2048 --generate-print --generate-dmc --generate-prompt-phase"

if [ "$PROMPT_FILE" != "" ]; then
    GENERATE_ARGS+=" --generate-prompt-file $PROMPT_FILE"
else
    ARGS+=" --data-path $DATA_BLEND"
    ARGS+=" --data-cache-path $DATA_CACHE"
    GENERATE_ARGS+=" --generate-context-len 512"
fi

python generate_with_dmc.py $ARGS $GENERATE_ARGS
