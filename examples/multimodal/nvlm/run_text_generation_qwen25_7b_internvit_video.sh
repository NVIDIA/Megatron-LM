#!/bin/bash

export NCCL_IB_SL=1
export CUDA_DEVICE_MAX_CONNECTIONS=1
export NVTE_APPLY_QK_LAYER_SCALING=0
export TOKENIZERS_PARALLELISM="false"

INPUT_IMAGE_PATH="placeholder"
GROUNDTRUTH_PATH="placeholder"

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

SEQ_LEN=256
DECODER_SEQ_LEN=16384

EXTRA_ARGS=" --pixel-shuffle"


for PARTITION_ID in $( eval echo {$START..$END} )
do
    torchrun --nproc_per_node 8 examples/multimodal/run_text_generation.py \
        --attention-softmax-in-fp32 \
        --transformer-impl transformer_engine \
        --use-te \
        --use-checkpoint-args \
        --normalization RMSNorm \
        --norm-epsilon 1e-06 \
        --language-model-type=qwen2.5_7B \
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
        --num-query-groups 4 \
        --num-layers 28 \
        --hidden-size 3584 \
        --ffn-hidden-size 18944 \
        --add-qkv-bias \
        --num-attention-heads 28 \
        --max-position-embeddings 32768  \
        --no-masked-softmax-fusion \
        --load ${MODEL_PATH} \
        --tokenizer-type MultimodalTokenizer \
        --tokenizer-model Qwen/Qwen2.5-7B-Instruct \
        --tokenizer-prompt-format qwen2p5 \
        --bf16 \
        --micro-batch-size 1 \
        --seq-length ${SEQ_LEN} \
        --decoder-seq-length ${DECODER_SEQ_LEN} \
        --out-seq-length 128 \
        --temperature 1.0 \
        --img-h 448 \
        --img-w 448 \
        --patch-dim 14 \
        --seed 153 \
        --top_k 1 \
        --no-load-rng \
        --no-load-optim \
        --input-image-path ${INPUT_IMAGE_PATH} \
        --num-partitions ${NUM_PARTITIONS} \
        --partition-id ${PARTITION_ID} \
        --output-path ${OUTPUT_PATH} \
        --gt-path ${GROUNDTRUTH_PATH} \
        --task ${TASK} \
        ${EXTRA_ARGS} \
        --special-tokens "<image>" "<img>" "</img>" \
        --vision-model-type internvit \
        --num-frames ${NUM_FRAMES} \
        --ckpt-format torch
done
