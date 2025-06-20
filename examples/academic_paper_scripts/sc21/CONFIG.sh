#!/bin/bash


# SLURM options.
export SLURM_PARTITION=<slurm partition, used to feed -p option in slurm>
export SLURM_ACCOUNT=<slurm account, used to feed -A option in slurm>


# Source code.
export MEGATRON_CODE_DIR=<megatron source code directory>


# This variable is used to mount the relevant part of the filesystem
# inside the docker container. Note that the `MEGATRON_CODE_DIR` and the
# launch directory already get mounted; this variable should be used to
# mount the directories that contain the data and tokenizer files.
export DOCKER_MOUNT_DIR=<megatron dataset and bpe tokenizer vocab path>


# Data and tokenizer files.
MEGATRON_DATA=<path to megatron processed data>
BPE_VOCAB_FILE=<path to bpe vocab file>
BPE_MERGE_FILE=<path to bpe merges file>


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


