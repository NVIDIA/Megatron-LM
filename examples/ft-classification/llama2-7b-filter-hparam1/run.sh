PRETRAINED_MODEL_PATH=$1
TOKENIZER_MODEL=$2
DATA_PATH=$3
CHECKPOINT_PATH=$4
TENSORBOARD_LOGS_PATH=$5

GPT_MODEL_ARGS=(
    --seq-length 4096 
    --max-position-embeddings 4096 
    --num-layers 32
    --hidden-size 4096
    --ffn-hidden-size 11008
    --num-attention-heads 32
    --hidden-dropout 0.0
    --attention-dropout 0.0
    --make-vocab-size-divisible-by 128
    --norm-epsilon 1.0e-05
    --disable-bias-linear
    --swiglu
    --untie-embeddings-and-output-weights
    --use-rotary-position-embeddings
    --normalization RMSNorm
    --no-position-embedding
    --no-query-key-layer-scaling
    --use-flash-attn
    --bf16
)

LOGISTICS_ARGS=(
    --tokenizer-type Llama2Tokenizer
    --tokenizer-model $TOKENIZER_MODEL
    --pretrained-checkpoint $PRETRAINED_MODEL_PATH
    --load $CHECKPOINT_PATH

    --save-interval 5000
    --save $CHECKPOINT_PATH
    --log-interval 100
    --eval-interval 2500
    --eval-iters 100 

    --tensorboard-dir $TENSORBOARD_LOGS_PATH 
    --tensorboard-log-interval 100

    --seed 1234
)

TRAINING_ARGS=(
    --micro-batch-size 2
    --global-batch-size 64
    --epochs 5

    --lr 5.0e-6
    --lr-decay-style cosine
    --lr-warmup-fraction 0.065
    --min-lr 5.0e-7

    --weight-decay 0.1 
    --adam-beta1 0.9 
    --adam-beta2 0.95 
    --init-method-std 0.01
    --clip-grad 1.0
)

MODEL_PARALLEL_ARGS=(
	--tensor-model-parallel-size 2
    --pipeline-model-parallel-size 2
    --no-async-tensor-model-parallel-allreduce
)

DATA_ARGS=(
    --task FILTER
    --train-data $DATA_PATH/alpaca_data_train.json
    --valid-data $DATA_PATH/alpaca_data_valid.json
)


python tasks/main.py \
    ${GPT_MODEL_ARGS[@]} \
    ${LOGISTICS_ARGS[@]} \
    ${TRAINING_ARGS[@]} \
    ${MODEL_PARALLEL_ARGS[@]} \
    ${DATA_ARGS[@]}
