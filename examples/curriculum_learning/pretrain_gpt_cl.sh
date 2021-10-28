#!/bin/bash

# This is a dummy train script to show how to use curriculum
# learning, some parameters are not for actual GPT pretraining.

TARGET_GLOBAL_BATCH_SIZE=512
TRAIN_SAMPLES=146_484_375
LR=1.0e-4
MIN_LR=1.0e-5
LR_DECAY_SAMPLES=126_953_125
LR_WARMUP_SAMPLES=183_105
SEQLEN=1024

############################################################
# New configs for curriculum learning, see README.md
TRAIN_TOKENS=10_000_000_000
# LR_DECAY_TOKENS=LR_DECAY_SAMPLES*SEQLEN
LR_DECAY_TOKENS=130000000000
############################################################

LOG_INTERVAL=100
EVAL_ITERS=10
EVAL_INTERVAL=100
SAVE_INTERVAL=1000

VOCAB_PATH=/data/Megatron-LM/data/gpt2-vocab.json
MERGE_PATH=/data/Megatron-LM/data/gpt2-merges.txt
DATA_PATH=/data/Megatron-LM/data/indexed_datasets/megatron

MICRO_BATCH_SIZE=1
MP_SIZE=1
PP_SIZE=1

NUM_GPUS=128
echo ${NUM_GPUS}
if [[ $PP_SIZE -gt 0 ]]; then
    DP_SIZE=$(( ${NUM_GPUS} / (${PP_SIZE} * ${MP_SIZE}) ))
else
    DP_SIZE=$(( ${NUM_GPUS} / ${MP_SIZE} ))
fi
GRAD_ACC_STEPS=$(( ${TARGET_GLOBAL_BATCH_SIZE} / (${MICRO_BATCH_SIZE} * ${DP_SIZE}) ))

NAME="gpt-117M-pp${PP_SIZE}-mp${MP_SIZE}-bsz${TARGET_GLOBAL_BATCH_SIZE}-mbsz${MICRO_BATCH_SIZE}-cl"
current_time=$(date "+%Y.%m.%d-%H.%M.%S")
host="${HOSTNAME}"
TENSORBOARD_DIR="tensorboard/${NAME}_${host}_${current_time}"
mkdir -p ${TENSORBOARD_DIR}
CHECKPOINT_PATH="checkpoints/${NAME}"

megatron_options=" \
        --data-path ${DATA_PATH} \
        --vocab-file ${VOCAB_PATH} \
        --merge-file ${MERGE_PATH} \
        --data-impl mmap \
        --override-lr-scheduler \
        --adam-beta1 0.9 \
        --adam-beta2 0.95 \
        --tensor-model-parallel-size ${MP_SIZE} \
        --init-method-std 0.014 \
        --lr-decay-tokens ${LR_DECAY_TOKENS} \
        --lr-warmup-samples ${LR_WARMUP_SAMPLES} \
        --micro-batch-size ${MICRO_BATCH_SIZE} \
        --global-batch-size ${TARGET_GLOBAL_BATCH_SIZE} \
        --num-layers 12 \
        --hidden-size 768 \
        --num-attention-heads 16 \
        --seq-length ${SEQLEN} \
        --max-position-embeddings ${SEQLEN} \
        --train-samples ${TRAIN_SAMPLES} \
        --train-tokens ${TRAIN_TOKENS} \
        --lr ${LR} \
        --min-lr ${MIN_LR} \
        --lr-decay-style cosine \
        --split 98,2,0 \
        --log-interval ${LOG_INTERVAL} \
        --eval-interval ${EVAL_INTERVAL} \
        --eval-iters ${EVAL_ITERS} \
        --save-interval ${SAVE_INTERVAL} \
        --weight-decay 0.1 \
        --clip-grad 1.0 \
        --hysteresis 2 \
        --num-workers 0 \
        --checkpoint-activations \
        --fp16 \
        --load ${CHECKPOINT_PATH} \
        --save ${CHECKPOINT_PATH} \
        --tensorboard-queue-size 1 \
        --log-timers-to-tensorboard \
        --log-batch-size-to-tensorboard \
        --log-validation-ppl-to-tensorboard \
        --no-masked-softmax-fusion \
        --tensorboard-dir ${TENSORBOARD_DIR}"

config_json="ds_config_cl.json"

deepspeed_options=" \
		    --deepspeed \
		    --deepspeed_config ${config_json} \
		    --pipeline-model-parallel-size ${PP_SIZE} \
		    --partition-activations"

run_cmd="deepspeed ../../pretrain_gpt.py ${megatron_options} ${deepspeed_options} &>> ${NAME}.log"
echo ${run_cmd}
eval ${run_cmd}
set +x
