set -e

PRETRAINED_LLAMA_MODEL_PATH=$1
TOKENIZER_MODEL=$2
export BIN_IDX_PATH=$3
DATA_CACHE=$4
CHECKPOINT_DIR=$5
TENSORBOARD_LOGS_PATH=$6
ENG_TOK=$7
AR_TOK=$((20 - $ENG_TOK))

LR_RATE=3e-5
LR_WARMUP_ITERS=100
TRAIN_ITER=5_000
GLOBAL_BATCH_SIZE=1024

SAVE_INTERVAL=100
LOG_INTERVAL=10
EVAL_INTERVAL=100
EVAL_ITER=100
SPLIT_INFO='1,0,0'


# DISTRIBUTED_ARGS=(
#     --nproc_per_node $GPUS_PER_NODE 
#     --nnodes $NUM_NODES 
#     --master_addr $MASTER_ADDR 
#     --master_port $MASTER_PORT
# )

EXP_NAME='en_'$ENG_TOK-'ar_'$AR_TOK
CKPT_DIR=$CHECKPOINT_DIR/$EXP_NAME/ckpts/
CACHE_DIR=$DATA_CACHE/$EXP_NAME/cache
TBOARD_DIR=$TENSORBOARD_LOGS_PATH/$EXP_NAME/tensorboard
mkdir -p $TBOARD_DIR
mkdir -p $CACHE_DIR
mkdir -p $CKPT_DIR

GPT_MODEL_ARGS=(
    --seq-length 4096 
    --max-position-embeddings 4096 
    --tokenizer-type Llama2Tokenizer
    --exit-on-missing-checkpoint
    --use-checkpoint-args
    --untie-embeddings-and-output-weights
    --use-rotary-position-embeddings
    --normalization RMSNorm
    --no-position-embedding
    --no-masked-softmax-fusion
    --no-query-key-layer-scaling
)

LOGISTICS_ARGS=(
    --save $CKPT_DIR
    --load $PRETRAINED_LLAMA_MODEL_PATH 
    --tokenizer-model $TOKENIZER_MODEL
    --split $SPLIT_INFO
    --log-interval $LOG_INTERVAL
    --save-interval $SAVE_INTERVAL 
    --eval-interval $EVAL_INTERVAL
    --eval-iters $EVAL_ITER
    --tensorboard-dir $TBOARD_DIR
    --tensorboard-log-interval $LOG_INTERVAL
    --data-cache-path $CACHE_DIR
    --log-validation-ppl-to-tensorboard 
)

TRAINING_ARGS=(
    --no-initialization
    --no-load-optim
    --no-load-rng
    --micro-batch-size 1 
    --global-batch-size $GLOBAL_BATCH_SIZE
    --train-iters $TRAIN_ITER
    --lr-decay-style cosine 
    --weight-decay 0.1 
    --adam-beta1 0.9 
    --adam-beta2 0.95 
    --init-method-std 0.006 
    --clip-grad 1.0 
    --lr $LR_RATE
    --min-lr $LR_RATE
    --lr-warmup-iters $LR_WARMUP_ITERS
    --use-flash-attn
    --bf16
)

MODEL_PARALLEL_ARGS=(
    --tensor-model-parallel-size 2
    --pipeline-model-parallel-size 2
    --no-async-tensor-model-parallel-allreduce
)

source examples/pretrain-llama/training/llama_ve/llama_ve_init_emb_en_reasoning_ar_data_mix_hyp_tune/iter_prob.sh

# $BIN_IDX_PATH/$BIN_IDX_PATH/torchrun ${DISTRIBUTED_ARGS[@]} pretrain_gpt.py \
python pretrain_gpt.py \
    ${GPT_MODEL_ARGS[@]} \
    ${LOGISTICS_ARGS[@]} \
    ${TRAINING_ARGS[@]} \
    ${MODEL_PARALLEL_ARGS[@]} \
    ${DATA_PATH[@]}
