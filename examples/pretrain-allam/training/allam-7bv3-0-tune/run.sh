PRETRAINED_MODEL_PATH=$1
TOKENIZER_MODEL=$2
BIN_IDX_PATH=$3
DATA_CACHE=$4
CHECKPOINT_DIR=$5
TENSORBOARD_LOGS_PATH=$6
ENG_TOK=$7

# DISTRIBUTED_ARGS=(
#     --nproc_per_node $GPUS_PER_NODE 
#     --nnodes $NUM_NODES 
#     --master_addr $MASTER_ADDR 
#     --master_port $MASTER_PORT
# )

AR_TOK=$((10 - $ENG_TOK))
echo "Training with ENG_TOK: $ENG_TOK"
echo "Training with AR_TOK: $AR_TOK"

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
    --tokenizer-type Llama2Tokenizer
    --untie-embeddings-and-output-weights
    --use-rotary-position-embeddings
    --normalization RMSNorm
    --no-position-embedding
    --no-masked-softmax-fusion
    --no-query-key-layer-scaling
)

LOGISTICS_ARGS=(
    --save $CHECKPOINT_DIR/en$ENG_TOK-ar$AR_TOK
    --load $PRETRAINED_MODEL_PATH
    --load-iteration 120000
    --dataloader-type reset-single
    --tokenizer-model $TOKENIZER_MODEL
    --split 99996,2,2 
    --log-interval 10
    --save-interval 250 
    --eval-interval 250
    --eval-iters 10
    --tensorboard-dir $TENSORBOARD_LOGS_PATH/en$ENG_TOK-ar$AR_TOK 
    --tensorboard-log-interval 10
    --data-cache-path $DATA_CACHE
    --log-validation-ppl-to-tensorboard 
    --seed 1234
)

TRAINING_ARGS=(
    --micro-batch-size 1
    --global-batch-size 1024
    --train-iters 122500
    --override-opt_param-scheduler
    --lr 3.0e-05
    --lr-decay-style cosine 
    --weight-decay 0.1 
    --adam-beta1 0.9 
    --adam-beta2 0.95 
    --init-method-std 0.01
    --clip-grad 1.0 
    --min-lr 3.0e-05
    --lr-warmup-iters 0
    --use-flash-attn
    --bf16
)


MODEL_PARALLEL_ARGS=(
	--tensor-model-parallel-size 2
    --pipeline-model-parallel-size 2
    --no-async-tensor-model-parallel-allreduce
)

source examples/pretrain-allam/training/allam-7bv3-0-tune/iterator_prob.sh

# torchrun ${DISTRIBUTED_ARGS[@]} pretrain_gpt.py \
python pretrain_gpt.py \
    ${GPT_MODEL_ARGS[@]} \
    ${LOGISTICS_ARGS[@]} \
    ${TRAINING_ARGS[@]} \
    ${MODEL_PARALLEL_ARGS[@]} \
    ${DATA_PATH[@]}
  
  
  
  
  
