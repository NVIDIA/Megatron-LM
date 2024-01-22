PRETRAINED_MODEL_PATH=$1
TOKENIZER_MODEL=$2
DATA_PATH=$3
CHECKPOINT_DIR=$5
TENSORBOARD_LOGS_PATH=$6

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
    --no-masked-softmax-fusion
    --no-query-key-layer-scaling
    --use-flash-attn
    --bf16
)

LOGISTICS_ARGS=(
    --tokenizer-type Llama2Tokenizer
    --tokenizer-model $TOKENIZER_MODEL
    --pretrained-checkpoint $PRETRAINED_CHECKPOINT

    --save-interval 5000
    --save $CHECKPOINT_PATH
    --log-interval 100
    --eval-interval 1000
    --eval-iters 50 

    --tensorboard-dir $TENSORBOARD_LOGS_PATH 
    --tensorboard-log-interval 100

    --seed 1234
)

TRAINING_ARGS=(
    --micro-batch-size 1
    --global-batch-size 1024
    --epochs 5

    --lr 5.0e-5
    --lr-decay-style linear
    --lr-warmup-fraction 0.065
    --min-lr 5.0e-6

    --weight-decay 0.1 
    --adam-beta1 0.9 
    --adam-beta2 0.95 
    --init-method-std 0.01
    --clip-grad 1.0 s
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
