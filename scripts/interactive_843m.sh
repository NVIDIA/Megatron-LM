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
NPROCS=1 # 8
export NWORKERS=32
# export NVTE_FLASH_ATTN=0

# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# customize / begin.
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

# ADD_RETRIEVER=1
REPO_DIR="/lustre/fsw/adlr/adlr-nlp/lmcafee/data/retro/megatrons/retro-mcore"
# OUTPUT_DIR="${REPO_DIR}/scripts/843m"
# CHECKPOINT_DIR="${OUTPUT_DIR}/checkpoints/c${USE_CORE}-r${ADD_RETRIEVER}"
# TENSORBOARD_DIR="${CHECKPOINT_DIR}/tb"
# LOG_DIR="${OUTPUT_DIR}/logs"

# mkdir -p ${TENSORBOARD_DIR}
# mkdir -p ${LOG_DIR}

# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
# customize / end.
# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<







######## setup. ########

set -u

export CUDA_DEVICE_MAX_CONNECTIONS=1
export NCCL_IB_QPS_PER_CONNECTION=4
export NCCL_SOCKET_IFNAME=^vlan,lo
unset NCCL_DEBUG

# if [ -f "$CHECKPOINT_DIR/latest_checkpointed_iteration.txt" ]
# then
# LOAD_DIR=$CHECKPOINT_DIR
# LOAD_OPTION=""
# else
#     LOAD_DIR="/lustre/fsw/adlr/adlr-nlp/adlr-nlp-sharing/nvllm-1.1t/checkpoints/gpt3-843m-multi-1.1t-gtc-llr"
#     LOAD_OPTION="--no-load-optim --finetune"
# fi

# echo $LOAD_DIR

######## data blend. ########

# . /lustre/fsw/adlr/adlr-nlp/boxinw/megatron-lm-pretrain/scripts/lawrence_blend_oci.sh
. /lustre/fsw/adlr/adlr-nlp/lmcafee/data/retro/megatrons/retro-mcore/scripts/843m/lawrence_blend_oci.sh

######## args. ########

# --DDP-impl local \
# --save-interval 1000 \
# --save ${CHECKPOINT_DIR} \
# --load ${LOAD_DIR} ${LOAD_OPTION} \
# --tensorboard-dir ${TENSORBOARD_DIR} \
# --log-validation-ppl-to-tensorboard \
# --sequence-parallel \
# TP=8 # 1
ARGS=" \
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
    --tensor-model-parallel-size ${TP} \
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
    --log-interval 1 \
    --eval-iters 32 \
    --eval-interval 1260 \
    --tokenizer-type GPTSentencePieceTokenizer \
    --tokenizer-model /lustre/fsw/adlr/adlr-nlp/adlr-nlp-sharing/nvllm-1.1t/utils/mt_nlg_plus_multilingual_ja_zh_the_stack_frac_015_256k.model \
    --data-path ${DATA_BLEND} \
    --split 98,2,0 \
    --split-constraint 99,1,0 \
    --split-constraint 98,2,0 \
    --clip-grad 1.0 \
    --weight-decay 0.1 \
    --adam-beta1 0.9 \
    --adam-beta2 0.95 \
    --init-method-std 0.007 \
    --log-params-norm \
    --log-num-zeros-in-grad \
    --bf16 \
"

######## retro. ########

if [ "$ADD_RETRIEVER" = "0" ]; then
    if [ "$USE_CORE" = "0" ]; then
	SCRIPT=pretrain_gpt.py
    else
	SCRIPT=pretrain_gpt_core.py
    fi
else
    RETRO_WORKDIR=/lustre/fsw/adlr/adlr-nlp/boxinw/next-llm
    ARGS="${ARGS} \
    --retro-workdir ${RETRO_WORKDIR} \
    --retro-add-retriever \
    --num-workers 32 \
    "
    SCRIPT=pretrain_retro.py
    if [ "$USE_CORE" = "1" ]; then
	ARGS="${ARGS} --retro-use-core"
    fi
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
