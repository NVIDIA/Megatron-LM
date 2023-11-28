
PRETRAINED_LLAMA_MODEL_PATH=$1
TOKENIZER_MODEL=$2
BIN_IDX_PATH=$3
DATA_CACHE=$4
CHECKPOINT_DIR=$5
TENSORBOARD_LOGS_PATH=$6

# DISTRIBUTED_ARGS=(
#     --nproc_per_node $GPUS_PER_NODE 
#     --nnodes $NUM_NODES 
#     --master_addr $MASTER_ADDR 
#     --master_port $MASTER_PORT
# )


for ENG_LANG_PROB in 10 9 7 6 5 4 3 2 1 0; do

AR_LANG_PROB=$((10-ENG_LANG_PROB))
LR_RATE=1e-5

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

mkdir -p $TENSORBOARD_LOGS_PATH/'en-'$ENG_LANG_PROB/tensorboard
mkdir -p $DATA_CACHE/'en-'$ENG_LANG_PROB/cache
mkdir -p $CHECKPOINT_DIR/'en-'$ENG_LANG_PROB

LOGISTICS_ARGS=(
    --save $CHECKPOINT_DIR/$LR_RATE/
    --load $PRETRAINED_LLAMA_MODEL_PATH 
    --tokenizer-model $TOKENIZER_MODEL
    --split 9998,1,1 
    --log-interval 10
    --save-interval 150 
    --eval-interval 150
    --eval-iters 150
    --tensorboard-dir $TENSORBOARD_LOGS_PATH/$LR_RATE/tensorboard
    --tensorboard-log-interval 10
    --data-cache-path $DATA_CACHE/$LR_RATE/cache
    --log-validation-ppl-to-tensorboard 
)

TRAINING_ARGS=(
    --no-initialization
    --no-load-optim
    --no-load-rng
    --micro-batch-size 1 
    --global-batch-size 1024
    --train-iters 2250
    --lr-decay-style cosine 
    --weight-decay 0.1 
    --adam-beta1 0.9 
    --adam-beta2 0.95 
    --init-method-std 0.006 
    --clip-grad 1.0 
    --lr $LR_RATE
    --min-lr $LR_RATE
    --lr-warmup-iters 10
    --use-flash-attn
    --bf16
)
# --use-mcore-models

MODEL_PARALLEL_ARGS=(
	--tensor-model-parallel-size 2
    --pipeline-model-parallel-size 2
    --no-async-tensor-model-parallel-allreduce
)



# $BIN_IDX_PATH/$BIN_IDX_PATH/torchrun ${\DISTRIBUTED_ARGS[@]}\ pretrain_gpt.py\ \
python pretrain_gpt.py \
    ${GPT_MODEL_ARGS[@]} \
    ${LOGISTICS_ARGS[@]} \
    ${TRAINING_ARGS[@]} \
    ${MODEL_PARALLEL_ARGS[@]} \
    ${DATA_PATH[@]}
  

done