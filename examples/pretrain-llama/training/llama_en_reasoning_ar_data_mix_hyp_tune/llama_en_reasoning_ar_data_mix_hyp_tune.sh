set -e

PRETRAINED_LLAMA_MODEL_PATH=$1
TOKENIZER_MODEL=$2
export BIN_IDX_PATH=$3
DATA_CACHE=$4
CHECKPOINT_DIR=$5
TENSORBOARD_LOGS_PATH=$6


LR_RATE=1e-5
LR_WARMUP_ITERS=100
TRAIN_ITER=2500
GLOBAL_BATCH_SIZE=1024

SAVE_INTERVAL=100
LOG_INTERVAL=10
EVAL_INTERVAL=100
EVAL_ITER=100
SPLIT_INFO='998,1,1'
TOTAL_NUM_TOKENS=10_000_000_000
# DISTRIBUTED_ARGS=(
#     --nproc_per_node $GPUS_PER_NODE 
#     --nnodes $NUM_NODES 
#     --master_addr $MASTER_ADDR 
#     --master_port $MASTER_PORT
# )


for ENG_TOK in {2..10}
do
    AR_TOK=$((10 - $ENG_TOK))
    echo "Training with ENG_TOK: $ENG_TOK"
    echo "Training with AR_TOK: $AR_TOK"
    
    EXP_NAME=$ENG_TOK-$AR_TOK
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

    echo "{\"en\": $ENG_TOK,\"ar\": $AR_TOK}" > examples/pretrain-llama/training/llama_en_reasoning_ar_data_mix_hyp_tune/lang_prob.json

    python examples/data-processing/data_ratio_from_file.py \
    --prefix-paths-from-json "examples/pretrain-llama/training/llama_en_reasoning_ar_data_mix_hyp_tune/en_reasoning_and_arabic_files.json" \
    --domain-ratio-from-json "examples/pretrain-llama/training/llama_en_reasoning_ar_data_mix_hyp_tune/data_ratio.json" \
    --lang-select-prob-json "examples/pretrain-llama/training/llama_en_reasoning_ar_data_mix_hyp_tune/lang_prob.json" \
    --total-token $TOTAL_NUM_TOKENS \
    --exclude-iterator-json "examples/pretrain-llama/training/llama_en_reasoning_ar_data_mix_hyp_tune/exclude_iterator.json" \
    --prefix-for-file-path "\$BIN_IDX_PATH/" \
    --export-script "examples/pretrain-llama/training/llama_en_reasoning_ar_data_mix_hyp_tune/iterator_selection_prob.sh" \
    
    source examples/pretrain-llama/training/llama_en_reasoning_ar_data_mix_hyp_tune/iterator_selection_prob.sh

    # $BIN_IDX_PATH/$BIN_IDX_PATH/torchrun ${\DISTRIBUTED_ARGS[@]}\ pretrain_gpt.py\ \
    python pretrain_gpt.py \
        ${GPT_MODEL_ARGS[@]} \
        ${LOGISTICS_ARGS[@]} \
        ${TRAINING_ARGS[@]} \
        ${MODEL_PARALLEL_ARGS[@]} \
        ${DATA_PATH[@]}

    rm examples/pretrain-llama/training/llama_en_reasoning_ar_data_mix_hyp_tune/iterator_selection_prob.sh
    rm examples/pretrain-llama/training/llama_en_reasoning_ar_data_mix_hyp_tune/lang_prob.json
done