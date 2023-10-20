#!/bin/bash

if [ "$#" != 2 ]; then
    echo "expected 2 args."
    exit 1
fi

ADD_RETRIEVER=$1
TP=$2

######## setup. ########

set -u

export CUDA_DEVICE_MAX_CONNECTIONS=1
export NCCL_IB_QPS_PER_CONNECTION=4
export NCCL_SOCKET_IFNAME=^vlan,lo
unset NCCL_DEBUG

DIR=$(readlink -f `pwd`)
DATETIME=`date +'date_%y-%m-%d_time_%H-%M-%S'`
LOG_DIR=$DIR/logs
mkdir -p $LOG_DIR


######## retro. ########

REPO_DIR="${SHARE_DATA}/retro/megatrons/retro-mcore"

DATA_BLEND="/lustre/fsw/portfolios/adlr/users/lmcafee/retro/data/MTNLG/NIHExporter_shuf_text_document"
TRAIN_SAMPLES=200000
LR_DECAY_SAMPLES=175000
LR_WARMUP_SAMPLES=10000
EVAL_INTERVAL=2000
EVAL_ITERS=50
SEQ_LENGTH=512
MICRO_BATCH_SIZE=4 GLOBAL_BATCH_SIZE=256 # up til 2023/9/10
RETRO_WORKDIR=/lustre/fsw/portfolios/adlr/users/lmcafee/retro/workdirs/nih

NUM_LAYERS=12
HIDDEN_SIZE=512
NUM_ATTN_HEADS=8


if [ "$ADD_RETRIEVER" = "0" ]; then
    SCRIPT=pretrain_gpt.py
    ARGS=""
else
    ARGS=" \
    --retro-workdir ${RETRO_WORKDIR} \
    --retro-add-retriever \
    "
    SCRIPT=pretrain_retro.py
fi

######## args. ########

ARGS="${ARGS} \
    --attention-dropout 0.0 \
    --hidden-dropout 0.0 \
    --tensor-model-parallel-size ${TP} \
    --pipeline-model-parallel-size 1 \
    --num-layers ${NUM_LAYERS} \
    --hidden-size ${HIDDEN_SIZE} \
    --num-attention-heads ${NUM_ATTN_HEADS} \
    --seq-length ${SEQ_LENGTH} \
    --max-position-embeddings ${SEQ_LENGTH} \
    --micro-batch-size ${MICRO_BATCH_SIZE} \
    --global-batch-size ${GLOBAL_BATCH_SIZE} \
    --train-samples ${TRAIN_SAMPLES} \
    --lr-decay-samples ${LR_DECAY_SAMPLES} \
    --lr-warmup-samples ${LR_WARMUP_SAMPLES} \
    --lr 3.0e-4 \
    --min-lr 3.0e-5 \
    --lr-decay-style cosine \
    --log-interval 1 \
    --eval-interval ${EVAL_INTERVAL} \
    --eval-iters ${EVAL_ITERS} \
    --tokenizer-type GPTSentencePieceTokenizer \
    --tokenizer-model /lustre/fsw/portfolios/adlr/projects/adlr_nlp_arch/adlr_nlp_sharing/nvllm-1.1t/utils/mt_nlg_plus_multilingual_ja_zh_the_stack_frac_015_256k.model \
    --data-path ${DATA_BLEND} \
    --split 98,2,0 \
    --clip-grad 1.0 \
    --weight-decay 0.1 \
    --adam-beta1 0.9 \
    --adam-beta2 0.95 \
    --init-method-std 0.02 \
    --log-params-norm \
    --log-num-zeros-in-grad \
    --bf16 --DDP-impl local \
"

ARGS="${ARGS} --recompute-activations"
ARGS="${ARGS} --use-flash-attn"
ARGS="${ARGS} --apply-layernorm-1p"
ARGS="${ARGS} --untie-embeddings-and-output-weights"
ARGS="${ARGS} --disable-bias-linear"
ARGS="${ARGS} --no-position-embedding"
ARGS="${ARGS} --use-rotary-position-embeddings"
ARGS="${ARGS} --rotary-percent 0.5"
ARGS="${ARGS} --swiglu"
ARGS="${ARGS} --apply-residual-connection-post-layernorm"
ARGS="${ARGS} --num-workers 32 --exit-interval 500 --use-cpu-initialization"

# eof.
