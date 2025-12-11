#!/bin/bash

# Your SBATCH commands here if using SLURM.

# Please launch this script from megatron-lm root.

# Train a multimodal model.

export NCCL_IB_SL=1
export CUDA_DEVICE_MAX_CONNECTIONS=1
export NCCL_ALGO=^NVLS
export TOKENIZERS_PARALLELISM=false

USER=$SLURM_JOB_USER

# Auto-detect batch or interactive mode.
which srun
BATCH=$((1-$?))

DEBUG=0

if [[ $BATCH -eq 0 ]]; then
    DATETIME=`date +'%y-%m-%d-%H-%M-%S'`
    MODEL_NAME="qwen2.5-7B-internvit-video-sft-nvlm-${DATETIME}"
else
    MODEL_NAME="qwen2.5-7B-internvitp-video-sft-nvlm"
    DEBUG=0
fi

WORKSPACE="<some dir>"
SOURCE=`pwd`
OUTPUT_BASE="${WORKSPACE}/output"
OUTPUT="${OUTPUT_BASE}/${MODEL_NAME}"

FINETUNE_DIR="${OUTPUT}/checkpoints"
LOGS_DIR="${OUTPUT}/logs"
TENSORBOARD_DIR="${OUTPUT}/tensorboard"

# From pretraining. The pretraining checkpoint should have tensor parallel size to 4.
LOAD_NAME="mcore-qwen2p5-7b-internvit-tp4"

CHECKPOINT_DIR="${WORKSPACE}/output/${LOAD_NAME}/checkpoints"

DATA_TRAIN="${SOURCE}/examples/multimodal/nvlm/sft_blend.yaml"

if [[ $DEBUG -eq 1 ]]; then
    MBZ=1
    BZ=1
    NW=0
    AD=0.0
    HD=0.0
    LI=1
    # This is just for interactive testing purposes. Do not use for proper training.
    EXTRA_ARGS="--freeze-LM"
    ALLOW_NONDETERMINISTIC=1
else
    MBZ=1
    BZ=256
    NW=8
    AD=0.0
    HD=0.0
    LI=5
    EXTRA_ARGS=""
    ALLOW_NONDETERMINISTIC=1
fi

USE_TILING=1
SEQ_LEN=1024
DECODER_SEQ_LEN=16384
MAX_POS_EMBED=32768
TRAIN_SAMPLES=6602173
WARMUP_SAMPLES=198065


if [[ $BATCH -eq 0 ]]; then
    # Runs out of GPU memory in interactive memory without this.
    EXTRA_ARGS+="--freeze-LM"
fi

if [[ $USE_TILING -eq 1 ]]; then
    EXTRA_ARGS+=" --pixel-shuffle --use-tiling --max-num-tiles 12 --use-thumbnail"
    SEQ_LEN=256
fi


OPTIONS=" \
    --swiglu \
    --use-distributed-optimizer \
    --num-workers ${NW} \
    --num-layers 28 \
    --hidden-size 3584 \
    --norm-epsilon 1e-06 \
    --normalization RMSNorm \
    --num-attention-heads 28 \
    --exit-duration-in-mins 110 \
    --group-query-attention \
    --num-query-groups 4 \
    --ffn-hidden-size 18944 \
    --add-qkv-bias \
    --seq-length ${SEQ_LEN} \
    --decoder-seq-length ${DECODER_SEQ_LEN} \
    --max-position-embeddings ${MAX_POS_EMBED} \
    --dataloader-seq-length ${DECODER_SEQ_LEN} \
    --tokenizer-type MultimodalTokenizer \
    --tokenizer-model Qwen/Qwen2.5-7B-Instruct \
    --tokenizer-prompt-format qwen2p5 \
    --pixel-shuffle \
    --position-embedding-type rope \
    --rotary-percent 1.0 \
    --rotary-base 1000000 \
    --disable-bias-linear \
    --pipeline-model-parallel-size 1 \
    --tensor-model-parallel-size 4 \
    --language-model-type qwen2.5_7B \
    --vision-model-type internvit \
    --micro-batch-size ${MBZ} \
    --global-batch-size ${BZ} \
    --lr 2e-6 \
    --min-lr 2.5e-7 \
    --train-samples ${TRAIN_SAMPLES} \
    --lr-warmup-samples ${WARMUP_SAMPLES} \
    --lr-decay-style cosine \
    --clip-grad 10 \
    --weight-decay 0.1 \
    --adam-beta1 0.9 \
    --adam-beta2 0.95 \
    --init-method-std 0.014 \
    --attention-dropout ${AD} \
    --hidden-dropout ${HD} \
    --eod-mask-loss \
    --bf16 \
    --tensorboard-dir ${TENSORBOARD_DIR} \
    --img-h 448 \
    --img-w 448 \
    --patch-dim 14 \
    --data-path ${DATA_TRAIN} \
    --dataloader-type external \
    --split 100,0,0 \
    --prompt-path ${SOURCE}/examples/multimodal/nvlm/nvlm_prompts.json \
    --log-interval ${LI} \
    --save-interval 500 \
    --eval-interval 500 \
    --eval-iters 10 \
    --log-params-norm \
    --log-num-zeros-in-grad \
    ${EXTRA_ARGS} \
    --save ${FINETUNE_DIR} \
    --load ${FINETUNE_DIR} \
    --pretrained-checkpoint ${CHECKPOINT_DIR} \
    --distributed-timeout-minutes 60 \
    --allow-missing-vision-projection-checkpoint \
    --dataloader-save ${FINETUNE_DIR}/dataloader \
    --disable-vision-class-token \
    --use-te \
    --ckpt-format torch \
    --num-frames 32 \
    --use-checkpoint-args \
    --image-tag-type internvl \
    --recompute-granularity full \
    --recompute-method block \
    --recompute-num-layers 28 \
    --recompute-vision \
"


export NVTE_ALLOW_NONDETERMINISTIC_ALGO=${ALLOW_NONDETERMINISTIC}
export NVTE_APPLY_QK_LAYER_SCALING=0

# Interactive or batch mode
if [[ $BATCH -eq 0 ]]; then
    torchrun --nproc_per_node 8 examples/multimodal/train.py ${OPTIONS}
else
    run_cmd="python -u ${SOURCE}/examples/multimodal/train.py ${OPTIONS}"

    DATETIME=`date +'date_%y-%m-%d_time_%H-%M-%S'`

    srun -l --verbose \
    --container-image <path to docker image> \
    --container-mounts "<some mount>" \
    --output=${LOGS_DIR}/%x_%j_$DATETIME.log \
    sh -c "${run_cmd}"

    set +x
fi
