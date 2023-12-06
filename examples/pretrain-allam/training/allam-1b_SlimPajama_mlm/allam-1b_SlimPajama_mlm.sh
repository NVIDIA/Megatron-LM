
TOKENIZER_MODEL=$1
BIN_IDX_PATH=$2
DATA_CACHE=$3
CHECKPOINT_DIR=$4
TENSORBOARD_LOGS_PATH=$5

# DISTRIBUTED_ARGS=(
#     --nproc_per_node $GPUS_PER_NODE 
#     --nnodes $NUM_NODES 
#     --master_addr $MASTER_ADDR 
#     --master_port $MASTER_PORT
# )

GPT_MODEL_ARGS=(
    --seq-length 2048 
    --max-position-embeddings 2048 
    --num-layers 24
    --hidden-size 2048
    --ffn-hidden-size 3072
    --num-attention-heads 16
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
    --save $CHECKPOINT_DIR 
    --tokenizer-model $TOKENIZER_MODEL
    --split 99990,8,2 
    --log-interval 100
    --save-interval 5000 
    --eval-interval 1000 
    --eval-iters 50
    --tensorboard-dir $TENSORBOARD_LOGS_PATH 
    --tensorboard-log-interval 100
    --data-cache-path $DATA_CACHE
    --log-validation-ppl-to-tensorboard 
    --seed 1234
)

TRAINING_ARGS=(
    --micro-batch-size 8 
    --global-batch-size 512
    --train-iters 286000
    --lr 0.0002 
    --lr-decay-style cosine 
    --weight-decay 0.1 
    --adam-beta1 0.9 
    --adam-beta2 0.95 
    --init-method-std 0.01
    --clip-grad 1.0 
    --min-lr 2.0e-05
    --lr-warmup-iters 400
    --use-flash-attn
    --bf16
)


MODEL_PARALLEL_ARGS=(
	--tensor-model-parallel-size 1
    --pipeline-model-parallel-size 1
    --no-async-tensor-model-parallel-allreduce
)

source examples/pretrain-allam/training/allam-1b_SlimPajama_mlm/iter_prob.sh

# torchrun ${\DISTRIBUTED_ARGS[@]}\ pretrain_gpt.py\ \
python pretrain_gpt.py \
    ${GPT_MODEL_ARGS[@]} \
    ${LOGISTICS_ARGS[@]} \
    ${TRAINING_ARGS[@]} \
    ${MODEL_PARALLEL_ARGS[@]} \
    ${DATA_PATH[@]}
  
  
  
  
  
