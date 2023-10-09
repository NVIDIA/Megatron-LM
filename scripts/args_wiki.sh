#!/bin/bash

set -u
unset NCCL_DEBUG

if [ "$#" != 3 ]; then
    echo "expected 3 args, found ${#}."
    exit 1
fi
USE_CORE=$1
ADD_RETRIEVER=$2
NUM_WORKERS=$3

ROOT_DIR=/lustre/fsw/portfolios/adlr/users/lmcafee

# >>>
# DATA_PATH=${ROOT_DIR}/corpus-530b/Wikipedia-shuf/Wikipedia_en_ftfy_id_shuf_text_document
# RETRO_WORKDIR=${ROOT_DIR}/retro/workdirs/wiki-mt-lower-mcore
DATA_PATH=${ROOT_DIR}/corpus-530b/wiki-tiny/wiki-200k_text_document
RETRO_WORKDIR=${ROOT_DIR}/retro/workdirs/wiki-tiny
VOCAB_FILE=${ROOT_DIR}/retro/misc/vocab/gpt2-vocab.json
MERGE_FILE=${ROOT_DIR}/retro/misc/vocab/gpt2-merges.txt
TOKENIZER_ARGS=" \
    --tokenizer-type GPT2BPETokenizer \
    --vocab-file ${VOCAB_FILE} \
    --merge-file ${MERGE_FILE} \
"
GLOBAL_BATCH_SIZE=256
# +++
# DATA_PATH=${ROOT_DIR}/retro/data/MTNLG/NIHExporter_shuf_text_document
# RETRO_WORKDIR=${ROOT_DIR}/retro/workdirs/nih
# TOKENIZER_ARGS=" \
#     --tokenizer-type GPTSentencePieceTokenizer \
#     --tokenizer-model /lustre/fsw/portfolios/adlr/projects/adlr_nlp_arch/adlr_nlp_sharing/nvllm-1.1t/utils/mt_nlg_plus_multilingual_ja_zh_the_stack_frac_015_256k.model \
# "
# # GLOBAL_BATCH_SIZE=16
# GLOBAL_BATCH_SIZE=256
# <<<

# CHECKPOINT_DIR=${RETRO_WORKDIR}/checkpoints/c${USE_CORE}-r${ADD_RETRIEVER}
# CHECKPOINT_DIR=${RETRO_WORKDIR}/checkpoints/c0-r${ADD_RETRIEVER}
# CHECKPOINT_DIR=${RETRO_WORKDIR}/checkpoints/c1-r${ADD_RETRIEVER}
# TENSORBOARD_DIR="${CHECKPOINT_DIR}/tb"
# mkdir -p ${TENSORBOARD_DIR}

# --loss-scale 1024 \
# --DDP-impl local \
# --fp16 \
    # --train-samples  2037248  \
    # --lr-decay-samples 166400000 \
    # --lr-warmup-samples 162761 \
NUM_LAYERS=12 # 4, [*12]
HIDDEN_SIZE=768 # 256, [512], *768
NUM_HEADS=12 # [4], 8, *12
MICRO_BATCH_SIZE=4 # [4], *8
LOG_INTERVAL=1 # 20
# SAVE_INTERVAL=2000 EXIT_INTERVAL=1000
# SAVE_INTERVAL=10 EXIT_INTERVAL=20
EXIT_INTERVAL=10
# ARGS=" \
#     --tensorboard-dir ${TENSORBOARD_DIR} \
#     --log-validation-ppl-to-tensorboard \
#     --save-interval ${SAVE_INTERVAL} \
#     --save ${CHECKPOINT_DIR} \
#     --load ${CHECKPOINT_DIR} \
#     \
ARGS=" \
    --exit-interval ${EXIT_INTERVAL} \
    \
    ${TOKENIZER_ARGS} \
    --tensor-model-parallel-size 1 \
    --pipeline-model-parallel-size 1 \
    --num-layers ${NUM_LAYERS} \
    --hidden-size ${HIDDEN_SIZE} \
    --num-attention-heads ${NUM_HEADS} \
    --seq-length 2048 \
    --max-position-embeddings 2048 \
    --micro-batch-size ${MICRO_BATCH_SIZE} \
    --global-batch-size ${GLOBAL_BATCH_SIZE} \
    --train-samples 100000  \
    --lr-decay-samples 99000 \
    --lr-warmup-samples 1000 \
    --lr 6.0e-4 \
    --min-lr 6.0e-5 \
    --lr-decay-style cosine \
    --log-interval ${LOG_INTERVAL} \
    --eval-iters 100 \
    --eval-interval 2000 \
    --data-path ${DATA_PATH} \
    --split 98,2,0 \
    --clip-grad 1.0 \
    --weight-decay 0.1 \
    --adam-beta1 0.9 \
    --adam-beta2 0.95 \
    --init-method-std 0.023 \
    --log-params-norm \
    --log-num-zeros-in-grad \
    --bf16 \
    --dataloader-type cyclic \
    --no-data-sharding \
"

if [ "$ADD_RETRIEVER" = "0" ]; then
    if [ "$USE_CORE" = "0" ]; then
	SCRIPT=pretrain_gpt.py
    else
	SCRIPT=pretrain_gpt_core.py
    fi
else
    # --retro-no-verify-neighbor-count \
    ARGS="${ARGS} \
    --retro-workdir ${RETRO_WORKDIR} \
    --retro-add-retriever \
    --retro-cyclic-train-iters 750000 \
    --num-workers ${NUM_WORKERS} \
    "
    # if [ "$USE_CORE" = "0" ]; then
    # 	SCRIPT=pretrain_retro.py
    # else
    # 	SCRIPT=pretrain_retro_core.py
    # fi
    SCRIPT=pretrain_retro.py
    if [ "$USE_CORE" = "1" ]; then
	ARGS="${ARGS} --retro-use-core"
    fi
fi

# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# run_cmd=" \
#     pwd && cd $SHARE_SOURCE/megatrons/megatron-lm-${REPO} && pwd && \
#     export PYTHONPATH=$PYTHONPATH:${SHARE_SOURCE}/megatrons/megatron-lm-${REPO}&&\
#     python -u ${SCRIPT} ${ARGS} \
# "

# echo "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
# echo $run_cmd
# echo "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"

# export FI_PROVIDER="efa"
# export FI_EFA_USE_DEVICE_RDMA=1
# export NCCL_ALGO=ring
# export NCCL_PROTO=simple
# export LD_LIBRARY_PATH=/opt/amazon/efa/lib:$LD_LIBRARY_PATH

# # IMAGE="nvcr.io#nvidia/pytorch:22.09-py3"
# # IMAGE="gitlab-master.nvidia.com/adlr/megatron-lm/boxinw/faissgpu"
# # IMAGE="gitlab-master.nvidia.com/lmcafee/sandbox-cluster/retro"
# IMAGE="gitlab-master.nvidia.com/lmcafee/sandbox-cluster/retro-train"
# # CONTAINER_MOUNTS="/home/lmcafee/src:/home/lmcafee/src,/gpfs/fs1/projects/gpu_adlr/datasets:/gpfs/fs1/projects/gpu_adlr/datasets"
# CONTAINER_MOUNTS="/home/lmcafee/src:/home/lmcafee/src,/mnt/fsx-outputs-chipdesign:/mnt/fsx-outputs-chipdesign"
# srun -l \
#      --container-image $IMAGE \
#      --container-mounts $CONTAINER_MOUNTS \
#      --output=$LOG_DIR/"%j_r${ADD_RETRIEVER}.log" \
#      sh -c "${run_cmd}"
# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
