set -e

PRETRAINED_LLAMA_MODEL_PATH=$1
TOKENIZER_MODEL=$2
BIN_IDX_PATH=$3
DATA_CACHE=$4
CHECKPOINT_DIR=$5
TENSORBOARD_LOGS_PATH=$6
TP=$7
PP=$8

# DISTRIBUTED_ARGS=(
#     --nproc_per_node $GPUS_PER_NODE 
#     --nnodes $NUM_NODES 
#     --master_addr $MASTER_ADDR 
#     --master_port $MASTER_PORT
# )

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
    --save $CHECKPOINT_DIR 
    --load $PRETRAINED_LLAMA_MODEL_PATH 
    --tokenizer-model $TOKENIZER_MODEL
    --split 100,0,0 
    --log-interval 10
    --save-interval 1500 
    --eval-interval 1500
    --eval-iters 0
    --tensorboard-dir $TENSORBOARD_LOGS_PATH 
    --tensorboard-log-interval 100
    --data-cache-path $DATA_CACHE
    --log-validation-ppl-to-tensorboard 
)

TRAINING_ARGS=(
    --no-initialization
    --no-load-optim
    --no-load-rng
    --micro-batch-size 1 
    --global-batch-size 1024
    --train-iters 200
    --weight-decay 0.1 
    --adam-beta1 0.9 
    --adam-beta2 0.95 
    --init-method-std 0.006 
    --clip-grad 1.0 
    --lr 3.0e-5
    --min-lr 3.0e-5
    --lr-warmup-iters 20
    --lr-decay-style cosine 
    --use-flash-attn
    --bf16
    --attention-dropout 0.0
    --hidden-dropout 0.0
)
# --use-mcore-models

MODEL_PARALLEL_ARGS=(
	--tensor-model-parallel-size $TP
    --pipeline-model-parallel-size $PP
    --sequence-parallel
    --no-async-tensor-model-parallel-allreduce
)

source examples/pretrain-llama/training/llama_ve/llama70b_ve_init_emb_en_reasoning_ar_with_trans_from_scratch_test/iter_prob.sh

# $BIN_IDX_PATH/torchrun ${DISTRIBUTED_ARGS[@]} pretrain_gpt.py \
python pretrain_gpt.py \
    ${GPT_MODEL_ARGS[@]} \
    ${LOGISTICS_ARGS[@]} \
    ${TRAINING_ARGS[@]} \
    ${MODEL_PARALLEL_ARGS[@]} \
    ${DATA_PATH[@]}
  
  
  
  
  
