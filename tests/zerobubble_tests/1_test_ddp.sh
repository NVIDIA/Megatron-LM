#!/bin/bash

source $(dirname "${BASH_SOURCE[0]}")/commons.sh
setup;

export WORLD_SIZE_IN_GPUS=8
export GLOBAL_BATCH_SIZE=24
export PIPELINE_SIZE=1
export LAYERS=4

export AIP_RUN_NAME=$(basename $0 | cut -d '.' -f 1)
launch

# Obtained by running the following command on main branch of upstream:
# CUDA_DEVICE_MAX_CONNECTIONS=1 torchrun --nnodes 1 --node_rank 0 --master_addr localhost --master_port 10086 --nproc_per_node=8 /code/pretrain_gpt.py --tensor-model-parallel-size 1 --pipeline-model-parallel-size 1 --num-layers 4 --hidden-size 4096 --num-attention-heads 32 --exit-interval 300 --seq-length 1024 --max-position-embeddings 2048 --micro-batch-size 1 --global-batch-size 24 --train-samples 146484375 --lr-decay-samples 126953125 --lr-warmup-samples 183105 --lr 6.0e-5 --min-lr 6.0e-6 --lr-decay-style cosine --log-interval 10 --eval-iters 40 --eval-interval 10000 --data-path /dataset/c4_text_document --tokenizer-type GPTSentencePieceTokenizer --tokenizer-model /tokenizers/tokenizer.model --split 98,2,0 --clip-grad 8.0 --weight-decay 0.1 --adam-beta1 0.9 --adam-beta2 0.95 --init-method-std 0.006 --no-barrier-with-level-1-timing --profile-step-start 150 --profile-step-end 170 --profile-ranks 0 1 2 3 4 5 6 7 --fp16
check_loss "7.381943E+00"