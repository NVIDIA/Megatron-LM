#!/bin/bash

export NCCL_IB_SL=1
export CUDA_DEVICE_MAX_CONNECTIONS=1
export NVTE_APPLY_QK_LAYER_SCALING=0

INPUT_IMAGE_PATH="placeholder"
GROUNDTRUTH_PATH="placeholder"

USE_TILING=0
USE_PIXEL_SHUFFLE_ONLY=0

while [[ $# -gt 0 ]]; do
    case $1 in
        --input-image-path)
            INPUT_IMAGE_PATH="$2"
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
        --use-tiling)
            USE_TILING=1
            shift
            shift
            ;;
        --use-pixel-shuffle-only)
            USE_PIXEL_SHUFFLE_ONLY=1
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

SEQ_LEN=1024     # Image embeddings sequence length.
DECODER_SEQ_LEN=8192    # Language model sequence length.
MAX_POS_EMBED=8192

# Additional arguments.
EXTRA_ARGS=""

if [[ $USE_TILING -eq 1 ]]; then
    EXTRA_ARGS+=" --pixel-shuffle --use-tiling --max-num-tiles 6 --use-thumbnail --use-tile-tags"
    SEQ_LEN=261     # Image embeddings sequence length (256 image embeddings + 5 tile tag embeddings).
fi

if [[ $USE_PIXEL_SHUFFLE_ONLY -eq 1 ]]; then
    EXTRA_ARGS+=" --pixel-shuffle"
    SEQ_LEN=256
fi

for PARTITION_ID in $( eval echo {$START..$END} )
do
    torchrun --nproc_per_node 8 examples/multimodal/run_text_generation.py \
        --attention-softmax-in-fp32 \
        --no-masked-softmax-fusion \
        --swiglu \
        --num-layers 60 \
        --hidden-size 7168 \
        --normalization RMSNorm \
        --num-attention-heads 56 \
        --exit-on-missing-checkpoint \
        --group-query-attention \
        --num-query-groups 8 \
        --ffn-hidden-size 20480 \
        --load ${MODEL_PATH} \
        --seq-length ${SEQ_LEN} \
        --decoder-seq-length ${DECODER_SEQ_LEN} \
        --max-position-embeddings ${MAX_POS_EMBED} \
        --tokenizer-type MultimodalTokenizer \
        --tokenizer-model <tokenizer model path> \
        --tokenizer-prompt-format nvlm-yi-34b \
        --vocab-size 64000 \
        --make-vocab-size-divisible-by 1 \
        --position-embedding-type rope \
        --rotary-percent 1.0 \
        --rotary-base 5000000 \
        --disable-bias-linear \
        --tensor-model-parallel-size 8 \
        --pipeline-model-parallel-size 1 \
        --language-model-type yi-34b \
        --vision-model-type internvit \
        --micro-batch-size 1 \
        --attention-dropout 0.0 \
        --hidden-dropout 0.0 \
        --bf16 \
        --freeze-LM \
        --freeze-ViT \
        --img-h 448 \
        --img-w 448 \
        --patch-dim 14 \
        --use-te \
        --transformer-impl transformer_engine \
        --use-checkpoint-args \
        --out-seq-length 16 \
        --temperature 1.0 \
        --patch-dim 14 \
        --seed 1234 \
        --top_k 1 \
        --no-load-rng \
        --no-load-optim \
        --num-partitions ${NUM_PARTITIONS} \
        --partition-id ${PARTITION_ID} \
        --output-path ${OUTPUT_PATH} \
        --gt-path ${GROUNDTRUTH_PATH} \
        --disable-vision-class-token \
        --input-image-path ${INPUT_IMAGE_PATH} \
        --gt-path ${GROUNDTRUTH_PATH} \
        ${EXTRA_ARGS} \
        --task ${TASK} \
        --image-tag-type nlvm \
        --ckpt-format torch
done
