#!/bin/bash

export NCCL_IB_SL=1
export CUDA_DEVICE_MAX_CONNECTIONS=1
export NVTE_APPLY_QK_LAYER_SCALING=1


while [[ $# -gt 0 ]]; do
    case $1 in
        -i|--input-path)
            INPUT_PATH="$2"
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
        -g|--gt-path)
            GROUNDTRUTH_PATH="$2"
            shift
            shift
            ;;
        --default)
            DEFAULT=YES
            shift # past argument
            ;;
        -*|--*)
            echo "Invalid option $1"
            exit 1
            ;;
    esac
done

# Please modify these as needed.
NUM_PARTITIONS=100
START=0
END=0

for PARTITION_ID in $( eval echo {$START..$END} )
do
    torchrun --nproc_per_node 4 examples/multimodal/run_text_generation.py \
        --use-flash-attn \
        --language-model-type 8b \
        --apply-layernorm-1p \
        --untie-embeddings-and-output-weights \
        --disable-bias-linear \
        --position-embedding-type rope \
        --rotary-percent 0.5 \
        --squared-relu \
        --attention-dropout 0.0 \
        --hidden-dropout 0.0 \
        --tensor-model-parallel-size 4 \
        --pipeline-model-parallel-size 1 \
        --num-layers 32 \
        --hidden-size 4096 \
        --num-attention-heads 32 \
        --max-position-embeddings 4096 \
        --no-masked-softmax-fusion \
        --load ${MODEL_PATH} \
        --tokenizer-type GPTSentencePieceTokenizer \
        --tokenizer-model ${TOKENIZER_PATH} \
        --bf16 \
        --micro-batch-size 1 \
        --seq-length 99 \
        --out-seq-length 700 \
        --temperature 1.0 \
        --img-h 336 \
        --img-w 336 \
        --patch-dim 14 \
        --seed 153 \
        --top_k 1 \
        --disable-vision-class-token \
        --no-load-rng \
        --no-load-optim \
        --input-path ${INPUT_PATH} \
        --num-partitions ${NUM_PARTITIONS} \
        --partition-id ${PARTITION_ID} \
        --output-path ${OUTPUT_PATH}/${PART_ID}.jsonl \
        --gt-path ${GROUNDTRUTH_PATH}
done
