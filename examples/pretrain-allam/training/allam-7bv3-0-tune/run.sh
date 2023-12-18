PRETRAINED_MODEL_PATH=$1
TOKENIZER_MODEL=$2
BIN_IDX_PATH=$3
DATA_CACHE=$4
CHECKPOINT_DIR=$5
TENSORBOARD_LOGS_PATH=$6
ENG_TOK=$7

TRAIN_ITER=2500
GLOBAL_BATCH_SIZE=1024

SAVE_INTERVAL=250
LOG_INTERVAL=10
EVAL_INTERVAL=2500
EVAL_ITER=0
SPLIT_INFO='99996,2,2'
TOTAL_NUM_TOKENS=10_000_000_000

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
    --save $CHECKPOINT_DIR/$ENG_TOK-$AR_TOK
    --load $PRETRAINED_MODEL_PATH
    --load-iteration 120000
    --tokenizer-model $TOKENIZER_MODEL
    --split $SPLIT_INFO
    --log-interval $LOG_INTERVAL
    --save-interval $SAVE_INTERVAL 
    --eval-interval $EVAL_INTERVAL
    --eval-iters $EVAL_ITER
    --tensorboard-dir $TENSORBOARD_LOGS_PATH 
    --tensorboard-log-interval $LOG_INTERVAL
    --data-cache-path $DATA_CACHE
    --log-validation-ppl-to-tensorboard 
    --seed 1234
)

TRAINING_ARGS=(
    --micro-batch-size 1
    --global-batch-size $GLOBAL_BATCH_SIZE
    --train-iters $TRAIN_ITER    
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

echo "{\"en\": $ENG_TOK,\"ar\": $AR_TOK}" > examples/pretrain-allam/training/allam-7bv3-0-tune/lang_prob.json

if [ ! -f "examples/pretrain-allam/training/allam-7bv3-0-tune/data.json" ]; then
    python examples/data-processing/remote_list.py \
    --az-configs "examples/configs/azure_login_configs.json" \
    --input-folder-path "https://allamllmuksstandard.blob.core.windows.net/llm-data/data_repo/tokenize_by_v5_improved/meglm_tok_v5_improved_bin_idx/" \
    --export-data-signature "examples/pretrain-allam/training/allam-7bv3-0-tune/data.json"
fi

python examples/data-processing/data_ratio_from_file.py \
--prefix-paths-from-json "examples/pretrain-allam/training/allam-7bv3-0-tune/data.json" \
--domain-ratio-from-json "examples/pretrain-allam/training/allam-7bv3-0-tune/data_ratio.json" \
--lang-select-prob-json "examples/pretrain-allam/training/allam-7bv3-0-tune/lang_prob.json" \
--total-token $TOTAL_NUM_TOKENS \
--exclude-iterator-json "examples/pretrain-allam/training/allam-7bv3-0-tune/exclude_iterator.json" \
--prefix-for-file-path "\$BIN_IDX_PATH/" \
--export-script "examples/pretrain-allam/training/allam-7bv3-0-tune/iterator_prob.sh"

source "examples/pretrain-allam/training/allam-7bv3-0-tune/iterator_prob.sh"

# torchrun ${DISTRIBUTED_ARGS[@]} pretrain_gpt.py \
python pretrain_gpt.py \
    ${GPT_MODEL_ARGS[@]} \
    ${LOGISTICS_ARGS[@]} \
    ${TRAINING_ARGS[@]} \
    ${MODEL_PARALLEL_ARGS[@]} \
    ${DATA_PATH[@]}
  
  
  
  
  
