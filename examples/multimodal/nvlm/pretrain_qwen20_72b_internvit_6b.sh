#!/bin/bash

# Your SBATCH commands here if using SLURM.

# Please launch this script from megatron-lm root.

# Train a multimodal model.

export NCCL_IB_SL=1
export CUDA_DEVICE_MAX_CONNECTIONS=1
export TOKENIZERS_PARALLELISM="false"

DEBUG=0

if [[ $BATCH -eq 0 ]]; then
    DATETIME=`date +'%y-%m-%d-%H-%M-%S'`
    MODEL_NAME="mcore-qwen20-72b-internvit-${DATETIME}"
else
    MODEL_NAME="mcore-qwen20-72b-internvit"
fi

WORKSPACE="<some dir>"
SOURCE=`pwd`
OUTPUT_BASE="${WORKSPACE}/output"
OUTPUT="${OUTPUT_BASE}/${MODEL_NAME}"

FINETUNE_DIR=${OUTPUT}/checkpoints
LOGS_DIR="${OUTPUT}/logs"
TENSORBOARD_DIR="${OUTPUT}/tensorboard"

CHECKPOINT_DIR="${WORKSPACE}/combined-qwen2.0-72b-instruct-internvit-6b-448px-1.5-tp8-te"

DATA_TRAIN="${SOURCE}/examples/multimodal/nvlm/pretrain_blend.yaml"

if [[ $DEBUG -eq 1 ]]; then
    MBZ=1
    BZ=1
    NW=0
    AD=0.0
    HD=0.0
    LI=1
    EXTRA_ARGS=""
    ALLOW_NONDETERMINISTIC=1
else
    MBZ=1
    BZ=2048
    NW=8
    AD=0.1
    HD=0.1
    LI=5
    EXTRA_ARGS=""
    ALLOW_NONDETERMINISTIC=1
fi

SEQ_LEN=256     # Image embeddings sequence length.
DECODER_SEQ_LEN=512     # Language model sequence length.
MAX_POS_EMBED=512


OPTIONS=" \
    --use-checkpoint-args \
    --exit-duration-in-mins 230 \
    --disable-bias-linear \
    --tokenizer-type MultimodalTokenizer \
    --tokenizer-model Qwen/Qwen2-72B-Instruct \
    --tokenizer-prompt-format qwen2p0 \
    --transformer-impl transformer_engine \
    --normalization RMSNorm \
    --norm-epsilon 1e-06 \
    --group-query-attention \
    --num-query-groups 8 \
    --no-masked-softmax-fusion \
    --attention-softmax-in-fp32 \
    --attention-dropout ${AD} \
    --hidden-dropout ${HD} \
    --untie-embeddings-and-output-weights \
    --position-embedding-type rope \
    --rotary-percent 1.0 \
    --rotary-base 1000000 \
    --swiglu \
    --tensor-model-parallel-size 8  \
    --pipeline-model-parallel-size 1  \
    --num-layers 80 \
    --hidden-size 8192 \
    --ffn-hidden-size 29568 \
    --add-qkv-bias \
    --num-attention-heads 64  \
    --use-distributed-optimizer \
    --use-te \
    --num-workers ${NW} \
    --seq-length ${SEQ_LEN} \
    --decoder-seq-length ${DECODER_SEQ_LEN} \
    --max-position-embeddings 32768 \
    --train-samples 122880000 \
    --lr-decay-samples 25600000 \
    --lr-warmup-samples 83200 \
    --micro-batch-size ${MBZ} \
    --global-batch-size ${BZ} \
    --lr 1e-4 \
    --min-lr 2.5e-5 \
    --lr-decay-style cosine \
    --log-interval ${LI} \
    --eval-iters 10 \
    --eval-interval 500 \
    --data-path ${DATA_TRAIN} \
    --prompt-path ${SOURCE}/examples/multimodal/nvlm/nvlm_prompts.json \
    --save-interval 5000 \
    --save ${FINETUNE_DIR} \
    --load ${FINETUNE_DIR} \
    --dataloader-save ${FINETUNE_DIR}/dataloader \
    --pretrained-checkpoint ${CHECKPOINT_DIR} \
    --split 100,0,0 \
    --clip-grad 10.0 \
    --weight-decay 0.1 \
    --adam-beta1 0.9 \
    --adam-beta2 0.95 \
    --init-method-std 0.014 \
    --bf16 \
    --eod-mask-loss \
    --freeze-ViT \
    --freeze-LM \
    --patch-dim 14 \
    --img-h 448 \
    --img-w 448 \
    --dataloader-type external \
    --tensorboard-dir ${TENSORBOARD_DIR} \
    --language-model-type qwen2.0_72B \
    ${EXTRA_ARGS} \
    --allow-missing-vision-projection-checkpoint \
    --vision-model-type internvit \
    --disable-vision-class-token \
    --log-params-norm \
    --log-num-zeros-in-grad \
    --ckpt-format torch \
    --pixel-shuffle \
    --image-tag-type nvlm
"


export NVTE_APPLY_QK_LAYER_SCALING=0
export NVTE_ALLOW_NONDETERMINISTIC_ALGO=${ALLOW_NONDETERMINISTIC}

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
