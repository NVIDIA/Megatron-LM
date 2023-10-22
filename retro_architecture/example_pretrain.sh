#!/bin/bash

#SBATCH -p luna
#SBATCH --nodes=1
#SBATCH -A adlr_nlp_llmnext
#SBATCH -t 0:15:00
#SBATCH --exclusive
#SBATCH --job-name=adlr_nlp_llmnext-lmcafee:lmcafee
#SBATCH --ntasks-per-node=8
#SBATCH --dependency=singleton

######## setup. ########

set -u

export CUDA_DEVICE_MAX_CONNECTIONS=1
export NCCL_IB_QPS_PER_CONNECTION=4
export NCCL_SOCKET_IFNAME=^vlan,lo
unset NCCL_DEBUG

######## data blend. ########

# REPO_DIR=/lustre/fsw/adlr/adlr-nlp/lmcafee/data/retro/megatrons/retro-mcore
REPO_DIR="/path/to/megatron"

ADD_RETRIEVER=1
# . /lustre/fsw/adlr/adlr-nlp/boxinw/megatron-lm-pretrain/scripts/lawrence_blend_oci.sh

######## args. ########

DATA_PATH="/lustre/fsw/adlr/adlr-nlp/lmcafee/data/retro/dataset-wiki-tiny/wiki-200k_text_document"

# --tokenizer-type GPTSentencePieceTokenizer \
# --tokenizer-model /lustre/fsw/adlr/adlr-nlp/adlr-nlp-sharing/nvllm-1.1t/utils/mt_nlg_plus_multilingual_ja_zh_the_stack_frac_015_256k.model \
# --split-constraint 99,1,0 \
# --split-constraint 98,2,0 \
# --sequence-parallel \
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
    --tensor-model-parallel-size 1 \
    --pipeline-model-parallel-size 1 \
    --num-layers 24 \
    --hidden-size 1024 \
    --num-attention-heads 16 \
    --seq-length 2048 \
    --max-position-embeddings 2048 \
    --micro-batch-size 4 \
    --global-batch-size 256 \
    --train-samples 100000 \
    --lr-decay-samples 99000 \
    --lr-warmup-samples 1000 \
    --lr 2.5e-5 \
    --min-lr 2.5e-6 \
    --lr-decay-style cosine \
    --log-interval 1 \
    --eval-iters 100 \
    --eval-interval 2000 \
    --tokenizer-type GPT2BPETokenizer \
    --vocab-file /lustre/fsw/adlr/adlr-nlp/lmcafee/data/retro/workdirs/wiki-tiny/gpt2-vocab.json \
    --merge-file /lustre/fsw/adlr/adlr-nlp/lmcafee/data/retro/workdirs/wiki-tiny/gpt2-merges.txt \
    --data-path ${DATA_PATH} \
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

######## retro. ########

if [ "$ADD_RETRIEVER" = "0" ]; then
    SCRIPT=pretrain_gpt.py
else
    # RETRO_WORKDIR=/lustre/fsw/adlr/adlr-nlp/boxinw/next-llm
    RETRO_WORKDIR=/lustre/fsw/adlr/adlr-nlp/lmcafee/data/retro/workdirs/wiki-tiny
    ARGS="${ARGS} \
    --retro-workdir ${RETRO_WORKDIR} \
    --retro-add-retriever \
    "
    SCRIPT=pretrain_retro.py
fi

######## Command. ########

SCRIPT_DIR="${REPO_DIR}/scripts/843m"
CMD=" \
    cd /lustre/fsw/adlr/adlr-nlp/lmcafee/data/retro/megatrons/retro-example && \
    ${SCRIPT_DIR}/bind.sh --cpu=${SCRIPT_DIR}/dgxa100_ccx.sh --mem=${SCRIPT_DIR}/dgxa100_ccx.sh python -u ${SCRIPT} ${ARGS} \
"
echo "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
echo $CMD
echo "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"

IMAGE="gitlab-master.nvidia.com/adlr/megatron-lm/lmcafee/retro-process-23.04"
MOUNTS="/lustre/fsw/adlr:/lustre/fsw/adlr"

# LOG_PATH="/lustre/fsw/adlr/adlr-nlp/lmcafee/data/retro/megatrons/retro-mcore/scripts/843m/example_logs/%j_example.log"
LOG_PATH="/path/to/logs/%j_example.log"

srun -l --export=ALL,PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python \
     --container-image $IMAGE \
     --container-mounts $MOUNTS \
     --output=$LOG_PATH \
     sh -c "${CMD}"

# eof.
