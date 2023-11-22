#!/bin/bash

set -u
unset NCCL_DEBUG
export CUDA_DEVICE_MAX_CONNECTIONS=1

######## Arguments. ########

if [ "$#" != 2 ]; then
    echo "expected 2 args, found ${#}."
    exit 1
fi
USE_CORE=$1
ADD_RETRIEVER=$2
NPROCS=8

# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# customize / begin.
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

REPO_DIR="/lustre/fsw/adlr/adlr-nlp/lmcafee/data/retro/megatrons/instructretro-test"

# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
# customize / end.
# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<







######## setup. ########

set -u

export CUDA_DEVICE_MAX_CONNECTIONS=1
export NCCL_IB_QPS_PER_CONNECTION=4
export NCCL_SOCKET_IFNAME=^vlan,lo
unset NCCL_DEBUG

######## data blend. ########

. /lustre/fsw/adlr/adlr-nlp/lmcafee/data/retro/megatrons/retro-mcore-test/scripts/843m/lawrence_blend_oci_soft.sh /lustre/fsw/adlr/adlr-nlp/lmcafee/data/retro/data/843m/english-custom

# echo $DATA_BLEND
# exit 0

######## args. ########

# --DDP-impl local \
# --sequence-parallel \
#     --data-path ${DATA_BLEND} \
# ARGS+=" --split-constraint 99,1,0 --split-constraint 98,2,0"
# --retro-split-preprocessing 98,2,0 \
ARGS=" \
    --log-interval 1 \
    --exit-interval 200 \
    --data-path ${DATA_BLEND} \
    \
    --recompute-activations \
    --use-flash-attn \
    --apply-layernorm-1p \
    --untie-embeddings-and-output-weights \
    --disable-bias-linear \
    --no-position-embedding \
    --use-rotary-position-embeddings \
    --rotary-percent 0.5 \
    --swiglu \
    --attention-dropout 0.0 \
    --hidden-dropout 0.0 \
    --exit-duration-in-mins 220 \
    --tensor-model-parallel-size 1 \
    --pipeline-model-parallel-size 1 \
    --num-layers 24 \
    --hidden-size 1024 \
    --num-attention-heads 16 \
    --seq-length 4096 \
    --max-position-embeddings 4096 \
    --micro-batch-size 2 \
    --global-batch-size 128 \
    --train-samples 25000000 \
    --lr-decay-samples 23750000 \
    --lr-warmup-samples 16667 \
    --lr 2.5e-5 \
    --min-lr 2.5e-6 \
    --lr-decay-style cosine \
    --eval-iters 32 \
    --eval-interval 1260 \
    --tokenizer-type GPTSentencePieceTokenizer \
    --tokenizer-model /lustre/fsw/adlr/adlr-nlp/adlr-nlp-sharing/nvllm-1.1t/utils/mt_nlg_plus_multilingual_ja_zh_the_stack_frac_015_256k.model \
    --split 99,1,0 \
    --clip-grad 1.0 \
    --weight-decay 0.1 \
    --adam-beta1 0.9 \
    --adam-beta2 0.95 \
    --init-method-std 0.007 \
    --log-params-norm \
    --log-num-zeros-in-grad \
    --bf16 \
"

# >>>
# CHECKPOINT_DIR="/lustre/fsw/adlr/adlr-nlp/lmcafee/data/retro/megatrons/retro-mcore-test/scripts/843m/checkpoints/continued/c${USE_CORE}-r${ADD_RETRIEVER}" # mr-model"
# TENSORBOARD_DIR="${CHECKPOINT_DIR}/tb"
# mkdir -p ${TENSORBOARD_DIR}

# if [ -f "$CHECKPOINT_DIR/latest_checkpointed_iteration.txt" ]; then
#     LOAD_DIR=$CHECKPOINT_DIR
#     LOAD_OPTION=""
# else
#     # LOAD_DIR="/lustre/fsw/adlr/adlr-nlp/adlr-nlp-sharing/nvllm-1.1t/checkpoints/gpt3-843m-multi-1.1t-gtc-llr"
#     LOAD_DIR="/lustre/fsw/adlr/adlr-nlp/lmcafee/data/retro/megatrons/retro-mcore-test/scripts/843m/checkpoints/core-gpt-te-843m"
#     LOAD_OPTION="--no-load-optim --finetune"
# fi

# # echo $LOAD_DIR

# ARGS+=" \
#   --save-interval 10 \
#   --save ${CHECKPOINT_DIR} \
#   --load ${LOAD_DIR} ${LOAD_OPTION} \
#   --tensorboard-dir ${TENSORBOARD_DIR} \
#   --log-validation-ppl-to-tensorboard \
# "
# <<<

######## retro. ########

# >>>
# if [ "$ADD_RETRIEVER" = "0" ]; then
#     SCRIPT=pretrain_gpt.py
# else
#     SCRIPT=pretrain_retro.py
#     # RETRO_WORKDIR=/lustre/fsw/adlr/adlr-nlp/boxinw/next-llm
#     RETRO_WORKDIR=/lustre/fsw/adlr/adlr-nlp/lmcafee/data/retro/workdirs/nextllm-soft
#     ARGS+=" \
#     --retro-workdir ${RETRO_WORKDIR} \
#     --retro-add-retriever \
#     --num-workers 32 \
#     "
# fi
if [ "$ADD_RETRIEVER" = "1" ]; then
    ARGS+=" --retro-add-retriever"
fi
# >>>
SCRIPT=pretrain_retro.py
ARGS+=" \
  --retro-workdir /lustre/fsw/adlr/adlr-nlp/lmcafee/data/retro/workdirs/nextllm-soft \
  --num-workers 32 \
"
# <<<

if [ "$USE_CORE" = "1" ]; then
    ARGS+=" --use-mcore-models"
fi

######## Command. ########

NODE_RANK=0
CMD="\
    cd ${REPO_DIR} && \
    export PYTHONPATH=${REPO_DIR}:/home/lmcafee/src/sandbox && \
    python -m torch.distributed.run \
    --nproc_per_node ${NPROCS} \
    --nnodes 1 \
    --node_rank ${NODE_RANK} \
    --master_addr ${MASTER_ADDR} \
    --master_port 6000 \
    ${SCRIPT} ${ARGS} \
"
echo "~~~~~~~~~~~~~~~~~~~~~~~~~~"
echo "CMD = '$CMD'."
echo "~~~~~~~~~~~~~~~~~~~~~~~~~~"
eval $CMD

# eof.
