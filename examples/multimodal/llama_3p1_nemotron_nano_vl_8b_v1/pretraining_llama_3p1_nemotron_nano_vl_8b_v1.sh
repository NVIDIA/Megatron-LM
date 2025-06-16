#!/bin/bash

# Your SBATCH commands here if using SLURM.

# Please launch this script from megatron-lm root.

# Train a multimodal model.

export CUDA_DEVICE_MAX_CONNECTIONS=1

USER=$SLURM_JOB_USER

# Auto-detect batch or interactive mode.
which srun
BATCH=$((1-$?))

DEBUG=0
USE_TILING=1

# Remember to update model and job name if running in batch mode!!
if [[ $BATCH -eq 0 ]]; then
    DATETIME=`date +'%y-%m-%d-%H-%M-%S'`
    MODEL_NAME="interactive_pretraining_llama_3p1_nemotron_nano_vl_8b_v1_${DATETIME}"
    SPECIAL_TOKENS="--special-tokens <image> <img> </img> <quad> </quad> <ref> </ref> <box> </box>"
    DEBUG=1
else
    MODEL_NAME="pretraining_llama_3p1_nemotron_nano_vl_8b_v1"
    SPECIAL_TOKENS="--special-tokens \<image\> \<img\> \</img\> \<quad\> \</quad\> \<ref\> \</ref\> \<box\> \</box\>"
fi

WORKSPACE="<some dir>"
SOURCE=`pwd`
OUTPUT_BASE="${WORKSPACE}/output"
OUTPUT="${OUTPUT_BASE}/${MODEL_NAME}"

FINETUNE_DIR=${OUTPUT}/checkpoints
LOGS_DIR="${OUTPUT}/logs"
TENSORBOARD_DIR="${OUTPUT}/tensorboard"

TP=4

CHECKPOINT_DIR="${WORKSPACE}/output/${LOAD_NAME}/checkpoints"

DATA_TRAIN="${SOURCE}/examples/multimodal/llama_3p1_nemotron_nano_vl_8b_v1/pretrain_blend.yaml"

if [[ $DEBUG -eq 1 ]]; then
    MBZ=1
    BZ=1
    NW=0
    AD=0.0
    HD=0.0
    LI=1

    NONDETERMINISTIC_ATTN=1

    NUM_GPU=4
    export CUDA_VISIBLE_DEVICES=0,1,2,3
else
    MBZ=1
    BZ=1024
    NW=8
    AD=0.0
    HD=0.0
    LI=5
    EXTRA_ARGS=""
    NONDETERMINISTIC_ATTN=1
    NUM_GPU=8
fi

SEQ_LEN=1024
DECODER_SEQ_LEN=4096

if [[ $USE_TILING -eq 1 ]]; then
    EXTRA_ARGS+=" --pixel-shuffle --use-tiling --max-num-tiles 12 --use-thumbnail"
    SEQ_LEN=256
fi

OPTIONS=" \
    --use-checkpoint-args \
    --disable-bias-linear \
    --tokenizer-type MultimodalTokenizer \
    --tokenizer-model meta-llama/Llama-3.1-8B-Instruct \
    --transformer-impl transformer_engine \
    --normalization RMSNorm \
    --group-query-attention \
    --num-query-groups 8 \
    --no-masked-softmax-fusion \
    --attention-softmax-in-fp32 \
    --attention-dropout ${AD} \
    --hidden-dropout ${HD} \
    --untie-embeddings-and-output-weights \
    --position-embedding-type rope \
    --rotary-percent 1.0 \
    --rotary-base 500000 \
    --use-rope-scaling \
    --swiglu \
    --tensor-model-parallel-size ${TP}  \
    --pipeline-model-parallel-size 1  \
    --num-layers 32 \
    --hidden-size 4096 \
    --ffn-hidden-size 14336 \
    --num-attention-heads 32 \
    --use-distributed-optimizer \
    --use-te \
    --num-workers ${NW} \
    --exit-duration-in-mins 230 \
    --seq-length ${SEQ_LEN} \
    --decoder-seq-length ${DECODER_SEQ_LEN} \
    --max-position-embeddings 131072 \
    --train-samples 1491231 \
    --lr-warmup-samples 102400 \
    --micro-batch-size ${MBZ} \
    --global-batch-size ${BZ} \
    --lr 2e-4 \
    --min-lr 0.0 \
    --lr-decay-style cosine \
    --log-interval ${LI} \
    --eval-iters 10 \
    --eval-interval 500 \
    --data-path ${DATA_TRAIN} \
    --prompt-path ${SOURCE}/examples/multimodal/manual_prompts.json \
    --save-interval 5000 \
    --save ${FINETUNE_DIR} \
    --load ${FINETUNE_DIR} \
    --dataloader-save ${FINETUNE_DIR}/dataloader \
    --pretrained-checkpoint ${CHECKPOINT_DIR} \
    --split 100,0,0 \
    --clip-grad 1.0 \
    --weight-decay 1e-2 \
    --adam-beta1 0.9 \
    --adam-beta2 0.999 \
    --init-method-std 0.02 \
    --log-params-norm \
    --log-num-zeros-in-grad \
    --bf16 \
    --eod-mask-loss \
    --freeze-ViT \
    --freeze-LM \
    --patch-dim 16 \
    --img-h 512 \
    --img-w 512 \
    --dataloader-type external \
    --tensorboard-dir ${TENSORBOARD_DIR} \
    --language-model-type=llama3.1_8b \
    ${EXTRA_ARGS} \
    --distributed-timeout-minutes 60 \
    --allow-missing-vision-projection-checkpoint \
    --vision-model-type radio \
    --tokenizer-prompt-format llama3p1 \
    --use-loss-scaling \
    ${SPECIAL_TOKENS} \
    --ckpt-format torch \
    --image-tag-type internvl \
    --force-system-message \
    --disable-vision-class-token \
    --use-area-weighted-aspect-ratio \
    --inference-max-seq-length 32768 \
"

export NVTE_APPLY_QK_LAYER_SCALING=0
export NVTE_ALLOW_NONDETERMINISTIC_ALGO=${NONDETERMINISTIC_ATTN}

# Interactive or batch mode
if [[ $BATCH -eq 0 ]]; then
    torchrun --nproc_per_node ${NUM_GPU} examples/multimodal/train.py ${OPTIONS}
else
    run_cmd="python -u ${SOURCE}/examples/multimodal/train.py ${OPTIONS}"

    DATETIME=`date +'date_%y-%m-%d_time_%H-%M-%S'`

    srun -l --verbose \
    --container-image <path to docker image> \
    --container-mounts "<some mount>" \
    --output=${LOGS_DIR}/%x_%j_$DATETIME.log \
    sh -c "echo ${run_cmd}; ${run_cmd}"

    set +x
fi
