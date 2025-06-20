#!/bin/bash

# Your SBATCH commands here if using SLURM.

# Please launch this script from megatron-lm root.

# Train a multimodal model.

export NCCL_IB_SL=1
export CUDA_DEVICE_MAX_CONNECTIONS=1
export NCCL_ALGO=^NVLS
export TOKENIZERS_PARALLELISM="false"


DEBUG=0

if [[ $BATCH -eq 0 ]]; then
    DATETIME=`date +'%y-%m-%d-%H-%M-%S'`
    MODEL_NAME="mcore-nous-yi34b-internvit-mlp-sft-${DATETIME}"
else
    MODEL_NAME="mcore-nous-yi34b-internvit-mlp-sft"
fi

WORKSPACE="<some dir>"
SOURCE=`pwd`
OUTPUT_BASE="${WORKSPACE}/output"
OUTPUT="${OUTPUT_BASE}/${MODEL_NAME}"

FINETUNE_DIR=${OUTPUT}/checkpoints
LOGS_DIR="${OUTPUT}/logs"
TENSORBOARD_DIR="${OUTPUT}/tensorboard"

LOAD_NAME="mcore-nous-yi34b-internvit-mlp"  # From pretraining
CHECKPOINT_DIR="${WORKSPACE}/output/${LOAD_NAME}/checkpoints"

DATA_TRAIN="${SOURCE}/examples/multimodal/nvlm/sft_blend.yaml"


if [[ $DEBUG -eq 1 ]]; then
    MBZ=1
    BZ=1
    NW=0
    LI=1
    AD=0.0
    HD=0.0
    ALLOW_NONDETERMINISTIC=1

    # Can run out of GPU memory in interactive memory without this.
    # This is just for interactive testing purposes. Do not use for proper training.
    EXTRA_ARGS=" --freeze-LM"
else
    MBZ=1
    BZ=128
    NW=2
    LI=5
    AD=0.0
    HD=0.0
    ALLOW_NONDETERMINISTIC=1

    EXTRA_ARGS=""
fi

SEQ_LEN=261     # Image embeddings sequence length (256 image embeddings + 5 tile tag embeddings).
DECODER_SEQ_LEN=3200    # Language model sequence length.
MAX_POS_EMBED=3200

OPTIONS=" \
    --swiglu \
    --use-distributed-optimizer \
    --num-workers ${NW} \
    --num-layers 60 \
    --hidden-size 7168 \
    --normalization RMSNorm \
    --num-attention-heads 56 \
    --exit-duration-in-mins 230 \
    --group-query-attention \
    --num-query-groups 8 \
    --ffn-hidden-size 20480 \
    --seq-length ${SEQ_LEN} \
    --decoder-seq-length ${DECODER_SEQ_LEN} \
    --max-position-embeddings ${MAX_POS_EMBED} \
    --tokenizer-type MultimodalTokenizer \
    --tokenizer-model NousResearch/Nous-Hermes-2-Yi-34B \
    --tokenizer-prompt-format nvlm-yi-34b \
    --vocab-size 64000 \
    --make-vocab-size-divisible-by 1 \
    --position-embedding-type rope \
    --rotary-percent 1.0 \
    --rotary-base 5000000 \
    --disable-bias-linear \
    --tensor-model-parallel-size 8 \
    --language-model-type yi-34b \
    --vision-model-type internvit \
    --micro-batch-size ${MBZ} \
    --global-batch-size ${BZ} \
    --train-samples 30000000 \
    --lr-decay-samples 25600000 \
    --lr-warmup-samples 83200 \
    --lr 2e-6 \
    --min-lr 2.5e-7 \
    --lr-decay-style cosine \
    --split 100,0,0 \
    --clip-grad 10 \
    --weight-decay 0.1 \
    --adam-beta1 0.9 \
    --adam-beta2 0.95 \
    --init-method-std 0.014 \
    --attention-dropout ${AD} \
    --hidden-dropout ${HD} \
    --untie-embeddings-and-output-weights \
    --eod-mask-loss \
    --bf16 \
    --tensorboard-dir=${TENSORBOARD_DIR} \
    --freeze-ViT \
    --img-h 448 \
    --img-w 448 \
    --patch-dim 14 \
    --data-path ${DATA_TRAIN} \
    --dataloader-type external \
    --dataloader-save ${FINETUNE_DIR}/dataloader \
    --prompt-path ${SOURCE}/examples/multimodal/nvlm/nvlm_prompts.json \
    --log-interval ${LI} \
    --load ${FINETUNE_DIR} \
    --save ${FINETUNE_DIR} \
    --pretrained-checkpoint ${CHECKPOINT_DIR} \
    --save-interval 5000 \
    --eval-interval 500 \
    --eval-iters 10 \
    --log-params-norm \
    --log-num-zeros-in-grad \
    ${EXTRA_ARGS} \
    --disable-vision-class-token \
    --use-te \
    --ckpt-format torch \
    --pixel-shuffle \
    --use-tiling \
    --max-num-tiles 6 \
    --use-thumbnail \
    --use-tile-tags \
    --image-tag-type nvlm
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
