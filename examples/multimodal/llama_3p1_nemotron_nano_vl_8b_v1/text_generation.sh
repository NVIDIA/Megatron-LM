#!/bin/bash

export NCCL_IB_SL=1
export CUDA_DEVICE_MAX_CONNECTIONS=1
export NVTE_APPLY_QK_LAYER_SCALING=0

INPUT_IMAGE_PATH="placeholder"
GROUNDTRUTH_PATH="placeholder"
NUM_FRAMES=1
TP=4
OUT_SEQ_LEN=1024
INFERENCE_MAX_SEQ_LEN=8192
USE_TILING=1
MAX_NUM_TILES=12

while [[ $# -gt 0 ]]; do
    case $1 in
        --tensor-model-parallel-size)
            TP="$2"
            shift
            shift
            ;;
        --input-image-path)
            INPUT_IMAGE_PATH="$2"
            shift
            shift
            ;;
        --num-frames)
            NUM_FRAMES="$2"
            shift
            shift
            ;;
        --out-seq-length)
            OUT_SEQ_LEN="$2"
            shift
            shift
            ;;
        --inference-max-seq-length)
            INFERENCE_MAX_SEQ_LEN="$2"
            shift
            shift
            ;;
        --max-num-tiles)
            MAX_NUM_TILES="$2"
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

SEQ_LEN=1024
DECODER_SEQ_LEN=16384

EXTRA_ARGS=""

if [[ $USE_TILING -eq 1 ]]; then
    EXTRA_ARGS+=" --pixel-shuffle --use-tiling --max-num-tiles ${MAX_NUM_TILES} --use-thumbnail"
    SEQ_LEN=256
fi

for PARTITION_ID in $( eval echo {$START..$END} )
do
    torchrun --nproc_per_node ${TP} examples/multimodal/run_text_generation.py \
        --attention-softmax-in-fp32 \
        --transformer-impl transformer_engine \
        --use-te \
        --use-checkpoint-args \
        --normalization RMSNorm \
        --language-model-type=llama3.1_8b \
        --untie-embeddings-and-output-weights \
        --disable-bias-linear \
        --position-embedding-type rope \
        --rotary-percent 1.0 \
        --rotary-base 500000 \
        --use-rope-scaling \
        --swiglu \
        --attention-dropout 0.0 \
        --hidden-dropout 0.0 \
        --tensor-model-parallel-size ${TP} \
        --pipeline-model-parallel-size 1 \
        --group-query-attention \
        --num-query-groups 8 \
        --num-layers 32 \
        --hidden-size 4096 \
        --ffn-hidden-size 14336 \
        --num-attention-heads 32 \
        --max-position-embeddings 131072 \
        --no-masked-softmax-fusion \
        --load ${MODEL_PATH} \
        --tokenizer-type MultimodalTokenizer \
        --tokenizer-model /lustre/fsw/portfolios/llmservice/projects/llmservice_nlp_fm/mcore_mmodal_models/models--meta-llama--Meta-Llama-3.1-8B-Instruct/snapshots/5206a32e0bd3067aef1ce90f5528ade7d866253f/ \
        --tokenizer-prompt-format llama3p1 \
        --bf16 \
        --micro-batch-size 1 \
        --seq-length ${SEQ_LEN} \
        --decoder-seq-length ${DECODER_SEQ_LEN} \
        --out-seq-length ${OUT_SEQ_LEN} \
        --inference-max-seq-length ${INFERENCE_MAX_SEQ_LEN} \
        --temperature 1.0 \
        --img-h 512 \
        --img-w 512 \
        --patch-dim 16 \
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
        --vision-model-type radio \
        --num-frames ${NUM_FRAMES} \
        --special-tokens "<image>" "<img>" "</img>" "<quad>" "</quad>" "<ref>" "</ref>" "<box>" "</box>" \
        --ckpt-format torch \
        --image-tag-type internvl \
        --disable-vision-class-token \
        --force-system-message \
        --exit-on-missing-checkpoint
done
