#!/bin/bash

print_help_and_exit() { echo -e "Usage:\n\t$0 {zero_out|retrofit|finetune} MODEL_PATH TOKENIZER_PATH"; exit 1; }
[ "$#" -ne 3 ] && print_help_and_exit

set -a

CUDA_DEVICE_MAX_CONNECTIONS=1
OMP_NUM_THREADS=1

STAGE=$1  # Allowed values: {zero_out, retrofit, finetune}
MODEL_PATH=$2  # Path to a model in Megatron format
TOKENIZER_MODEL_PATH=$3  # Path to tokenizer.model

: ${GPUS_PER_NODE:=8}
: ${SLURM_NNODES:=1}
: ${SLURM_NODEID:=0}
: ${MASTER_ADDR:="localhost"}
: ${MASTER_PORT:=6000}

# Data and checkpoint paths
: ${DATA_BLEND:="1.0 datasets/dmc_demo/llama2_tokenized/hf_wiki_20231101_en_train_text_document"}
: ${DATA_CACHE:="datasets/dmc_demo/cache_llama2"}

case $STAGE in
    "zero_out")
        OUTPUT_PATH="output/dmc_stage1"
        STAGE_ARGS="
            --dmc-is-stage-one \
            --train-iters 250 \
            --eval-interval 250 \
            --min-lr 3e-5 \
            --seed 1234 \
        "
        ;;
    "retrofit")
        OUTPUT_PATH="output/dmc_stage2"
        STAGE_ARGS="
            --train-samples $((1024*6000)) \
            `# --lr-decay-samples 2048000` \
            `# --lr-warmup-samples 0` \
            --eval-interval 1000 \
            --min-lr 3e-5 \
            --seed 1235 \
        "
        ;;
    "finetune")
        OUTPUT_PATH="output/dmc_stage3"
        STAGE_ARGS="
            --train-samples $((1024*2000)) \
            --eval-interval 1000 \
            --dmc-finetune \
            --dmc-cr 4.0 \
            --min-lr 3e-6 \
            --seed 1236 \
        "
        ;;
    *)
        print_help_and_exit
        ;;
esac

# Llama 2 7B
ARGS="
    `# Model` \
    --num-layers 32 \
    --hidden-size 4096 \
    --num-attention-heads 32 \
    --num-query-groups 32 \
    --ffn-hidden-size 11008
    --seq-length 4096 \
    --max-position-embeddings 4096 \
    `# Training` \
    --bf16 \
    --micro-batch-size 4 \
    --global-batch-size 1024 \
    --lr 3e-5 \
    --clip-grad 1.0 \
    --weight-decay 0.1 \
    --lr-decay-style cosine \
    --adam-beta1 0.9 \
    --adam-beta2 0.95 \
    --exit-duration-in-mins 220 \
    --no-load-optim \
    --no-load-rng \
    --finetune \
    --exit-signal-handler \
    --data-parallel-random-init \
    --use-checkpoint-args \
    --recompute-activations \
    `# GPT` \
    --use-flash-attn \
    --attention-dropout 0.0 \
    --hidden-dropout 0.0 \
    --untie-embeddings-and-output-weights \
    --use-rotary-position-embeddings \
    --normalization RMSNorm \
    --no-position-embedding \
    --transformer-impl transformer_engine \
    --no-masked-softmax-fusion \
    --disable-bias-linear \
    --no-bias-gelu-fusion \
    --swiglu \
    --group-query-attention \
    `# Distributed` \
    --tensor-model-parallel-size 4 \
    --pipeline-model-parallel-size 1 \
    --sequence-parallel \
    --distributed-backend nccl \
    `# Data` \
    --data-path ${DATA_BLEND} \
    --data-cache-path ${DATA_CACHE} \
    --tokenizer-type Llama2Tokenizer \
    --tokenizer-model ${TOKENIZER_MODEL_PATH} \
    --split 95,5,0 \
    `# Output` \
    --save-interval 250 \
    --log-interval 1 \
    --eval-iters 1 \
    --tensorboard-dir ${OUTPUT_PATH}/tensorboard \
    --save $OUTPUT_PATH \
    --load $MODEL_PATH \
"

DISTRIBUTED_ARGS="
    --nproc_per_node $GPUS_PER_NODE \
    --nnodes $SLURM_NNODES \
    --node_rank $SLURM_NODEID \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT \
"

torchrun $DISTRIBUTED_ARGS pretrain_gpt_with_dmc.py $ARGS $STAGE_ARGS
