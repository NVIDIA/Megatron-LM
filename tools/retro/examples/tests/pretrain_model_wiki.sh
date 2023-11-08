#!/bin/bash

set -u

unset NCCL_DEBUG
export CUDA_DEVICE_MAX_CONNECTIONS=1

######## GPT or Retro?. ########

# 0 : GPT.
# 1 : Retro

ADD_RETRIEVER=1

######## Megatron, Retro dirs. ########

REPO_DIR="/lustre/fs4/portfolios/adlr/users/boxinw/github-version/retro/Megatron-LM"
RETRO_WORKDIR="/lustre/fs4/portfolios/adlr/users/boxinw/workdirs/wiki"

######## Data. ########

DATA_HOME="/lustre/fs4/portfolios/adlr/users/boxinw/pretraining_data/"

WIK="${DATA_HOME}/MTNLG/Wikipedia_shuf_text_document"

DATA_BLEND=" \
  1 ${WIK} \
"
######## Args. ########

ARGS=" \
    --log-interval 1 \
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
    --seq-length 512 \
    --max-position-embeddings 512 \
    --micro-batch-size 16 \
    --global-batch-size 256 \
    --train-samples 200000 \
    --lr-decay-samples 175000 \
    --lr-warmup-samples 10000 \
    --lr 2.5e-5 \
    --min-lr 2.5e-6 \
    --lr-decay-style cosine \
    --eval-iters 50 \
    --eval-interval 2000 \
    --tokenizer-type GPTSentencePieceTokenizer \
    --tokenizer-model /lustre/fsw/portfolios/adlr/users/lmcafee/retro/misc/next-llm-tokenizer/mt_nlg_plus_multilingual_ja_zh_the_stack_frac_015_256k.model \
    --data-path ${DATA_BLEND} \
    --split 98,2,0 \
    --clip-grad 1.0 \
    --weight-decay 0.1 \
    --adam-beta1 0.9 \
    --adam-beta2 0.95 \
    --init-method-std 0.007 \
    --log-params-norm \
    --log-num-zeros-in-grad \
    --bf16 \
"

######## Retro. ########

if [ "$ADD_RETRIEVER" = "0" ]; then
    SCRIPT=pretrain_gpt.py
else
    ARGS="${ARGS} \
    --retro-workdir ${RETRO_WORKDIR} \
    --retro-add-retriever \
    "
    SCRIPT=pretrain_retro.py
fi

######## Command. ########

NPROCS=8 # Number of GPUs.
NODE_RANK=0
MASTER_ADDR=localhost
CMD="\
    pwd && cd ${REPO_DIR} && pwd && \
    export PYTHONPATH=$PYTHONPATH:${REPO_DIR} && \
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
