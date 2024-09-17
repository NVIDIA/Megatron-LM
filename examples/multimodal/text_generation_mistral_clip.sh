#!/bin/bash

export NCCL_IB_SL=1
export CUDA_DEVICE_MAX_CONNECTIONS=1
export NVTE_APPLY_QK_LAYER_SCALING=0

INPUT_METADATA_PATH="placeholder"
GROUNDTRUTH_PATH="placeholder"
NUM_FRAMES=1

while [[ $# -gt 0 ]]; do
    case $1 in
        --input-image-path)
            INPUT_IMAGE_PATH="$2"
            shift
            shift
            ;;
        --input-metadata-path)
            INPUT_METADATA_PATH="$2"
            shift
            shift
            ;;
        --num-frames)
            NUM_FRAMES="$2"
            shift
            shift
            ;;
        -g|--groundtruth-path)
            GROUNDTRUTH_PATH="$2"
            shift
            shift
            ;;
        -o|--output-path)
            OUTPUT_PATH="$2"
            shift
            shift
            ;;
        -m|--model-path)
            MODEL_PATH="$2"
            shift
            shift
            ;;
        -t|--tokenizer-path)
            TOKENIZER_PATH="$2"
            shift
            shift
            ;;
        --task)
            TASK="$2"
            shift
            shift
            ;;
        -g|--gt-path)
            GROUNDTRUTH_PATH="$2"
            shift
            shift
            ;;
        -*|--*)
            echo "Invalid option $1"
            exit 1
            ;;
    esac
done

# Please modify these as needed.
NUM_PARTITIONS=0
START=0
END=0

for PARTITION_ID in $( eval echo {$START..$END} )
do
    torchrun --nproc_per_node 4 examples/multimodal/run_text_generation.py \
        --apply-layernorm-1p \
        --attention-softmax-in-fp32 \
        --use-flash-attn \
        --transformer-impl transformer_engine \
        --use-te \
        --use-checkpoint-args \
        --normalization RMSNorm \
        --language-model-type mistral_7b \
        --untie-embeddings-and-output-weights \
        --disable-bias-linear \
        --position-embedding-type rope \
        --rotary-percent 1.0 \
        --rotary-base 1000000 \
        --swiglu \
        --attention-dropout 0.0 \
        --hidden-dropout 0.0 \
        --tensor-model-parallel-size 4 \
        --pipeline-model-parallel-size 1 \
        --group-query-attention \
        --num-query-groups 8 \
        --num-layers 32 \
        --hidden-size 4096 \
        --ffn-hidden-size 14336 \
        --num-attention-heads 32 \
        --max-position-embeddings 4096 \
        --no-masked-softmax-fusion \
        --load ${MODEL_PATH} \
        --tokenizer-type HuggingFaceTokenizer \
        --tokenizer-model ${TOKENIZER_PATH} \
        --bf16 \
        --micro-batch-size 1 \
        --seq-length 2048 \
        --out-seq-length 12 \
        --temperature 1.0 \
        --img-h 336 \
        --img-w 336 \
        --patch-dim 14 \
        --seed 153 \
        --top_k 1 \
        --no-load-rng \
        --no-load-optim \
        --input-image-path ${INPUT_IMAGE_PATH} \
        --input-metadata-path ${INPUT_METADATA_PATH} \
        --num-partitions ${NUM_PARTITIONS} \
        --partition-id ${PARTITION_ID} \
        --output-path ${OUTPUT_PATH}-${TASK}-${PARTITION_ID}.jsonl \
        --gt-path ${GROUNDTRUTH_PATH} \
        --task ${TASK} \
        --disable-vision-class-token \
        --prompt-format mistral \
        --num-frames ${NUM_FRAMES}
done
