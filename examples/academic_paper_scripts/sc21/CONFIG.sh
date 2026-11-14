#!/bin/bash


# SLURM options.
: "${SLURM_PARTITION:?Set SLURM_PARTITION, e.g. batch}"
: "${SLURM_ACCOUNT:?Set SLURM_ACCOUNT, e.g. my-slurm-account}"
export SLURM_PARTITION
export SLURM_ACCOUNT


# Source code.
: "${MEGATRON_CODE_DIR:?Set MEGATRON_CODE_DIR, e.g. /path/to/Megatron-LM}"
export MEGATRON_CODE_DIR


# This variable is used to mount the relevant part of the filesystem
# inside the docker container. Note that the `MEGATRON_CODE_DIR` and the
# launch directory already get mounted; this variable should be used to
# mount the directories that contain the data and tokenizer files.
: "${DOCKER_MOUNT_DIR:?Set DOCKER_MOUNT_DIR, e.g. /path/to/data-and-tokenizers}"
export DOCKER_MOUNT_DIR


# Data and tokenizer files.
: "${MEGATRON_DATA:?Set MEGATRON_DATA, e.g. /path/to/megatron_processed_data}"
: "${BPE_VOCAB_FILE:?Set BPE_VOCAB_FILE, e.g. /path/to/gpt2-vocab.json}"
: "${BPE_MERGE_FILE:?Set BPE_MERGE_FILE, e.g. /path/to/gpt2-merges.txt}"


# Megatron input parameters.
# `MEGATRON_EXTRA_PARAMS` can be used to provide any extra parameters
# that are not listed here. 
export MEGATRON_PARAMS=" ${MEGATRON_EXTRA_PARAMS} \
	--tensor-model-parallel-size ${TP} \
	--pipeline-model-parallel-size ${PP} \
	--micro-batch-size ${MBS} \
	--global-batch-size ${GBS} \
        --num-layers ${NLS} \
        --hidden-size ${HS} \
        --num-attention-heads ${NAH} \
	--DDP-impl ${DDP} \
	--data-path ${MEGATRON_DATA} \
	--vocab-file ${BPE_VOCAB_FILE} \
	--merge-file ${BPE_MERGE_FILE} \
        --log-interval 5 \
        --seq-length 2048 \
        --max-position-embeddings 2048 \
        --train-iters 500 \
        --lr-decay-iters 320 \
        --lr 0.0001 \
	--min-lr 0.00001 \
        --lr-decay-style cosine \
        --lr-warmup-fraction 0.01 \
        --split 969,30,1 \
        --eval-iters 100 \
        --eval-interval 1000 \
        --clip-grad 1.0 \
        --fp16 \
	--loss-scale 8192 "

