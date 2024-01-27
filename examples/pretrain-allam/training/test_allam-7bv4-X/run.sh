PRETRAINED_MODEL_PATH=$1
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

GPT_MODEL_ARGS=(
    --use-mcore-models
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
    --no-query-key-layer-scaling
    --use-distributed-optimizer
    --overlap-grad-reduce
)

LOGISTICS_ARGS=(
    --save $CHECKPOINT_DIR 
    --dataloader-type single 
    --tokenizer-model $TOKENIZER_MODEL
    --split 1,0,0 
    --log-interval 1
    --log-params-norm
    --log-throughput
    --save-interval 5 
    --eval-interval 1000 
    --eval-iters 0
    --tensorboard-dir $TENSORBOARD_LOGS_PATH 
    --tensorboard-log-interval 1
    --data-cache-path $DATA_CACHE
    --log-validation-ppl-to-tensorboard 
    --log-memory-to-tensorboard
    --log-timers-to-tensorboard
    --seed 1234
)

TRAINING_ARGS=(
    --micro-batch-size 1
    --global-batch-size 1024
    --train-iters 10
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
	--tensor-model-parallel-size 1
    --pipeline-model-parallel-size 1
    --sequence-parallel
    --no-async-tensor-model-parallel-allreduce
)

source examples/pretrain-allam/training/test_allam-7bv4-X/iterator_prob.sh

# torchrun ${\DISTRIBUTED_ARGS[@]}\ pretrain_gpt.py\ \
python pretrain_gpt.py \
    ${GPT_MODEL_ARGS[@]} \
    ${LOGISTICS_ARGS[@]} \
    ${TRAINING_ARGS[@]} \
    ${MODEL_PARALLEL_ARGS[@]} \
    ${DATA_PATH[@]}
  
  
  
  
  
