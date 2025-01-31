#!/bin/bash
###############################################################################
# Copyright (c) 2024, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
#################################################################################
set -e

CURRENT_DIR="$( cd "$( dirname "$0" )" && pwd )"
MEGATRON_PATH=$( dirname $( dirname ${CURRENT_DIR}))
export PYTHONPATH=${MEGATRON_PATH}:${MEGATRON_PATH}/PAI-Megatron-LM-240718:$PYTHONPATH
export CUDA_DEVICE_MAX_CONNECTIONS=1
echo $CURRENT_DIR

ENV=dsw

# Here are some configs controled by env
if [ -z ${MP_DATASET_TYPE} ];then
    MP_DATASET_TYPE="idxmap"
fi

if [ -z ${MP_AC_LAYERS} ];then
    MP_AC_LAYERS=1
fi

EXPERIMENT_DIR="experiment"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)


MODEL_NAME=DeepSeek-V2-Lite

DATA_DIR="/workspace/data"
MODEL=deepseek-ai/${MODEL_NAME}

MODEL_SIZE=16B
BATCH_SIZE=4
GLOBAL_BATCH_SIZE=256
LR=1e-5
MIN_LR=1e-6
SEQ_LEN=2048
PAD_LEN=2048
PR=bf16
TP=1
PP=1  
CP=1
EP=8  
SP=true
DO=true
FL=true
SFT=false   
AC=sel #full
OPTIMIZER_OFFLOAD=false
SAVE_INTERVAL=5000
DATASET_PATH=${DATA_DIR}/deepseekv2-train-datasets/mmap_deepseekv2_datasets_text_document
VALID_DATASET_PATH=${DATA_DIR}/deepseekv2-train-datasets/mmap_deepseekv2_datasets_text_document
PRETRAIN_CHECKPOINT_PATH=${DATA_DIR}/deepseek-ckpts/DeepSeek-V2-Lite
TRAIN_ITERS=20
LR_WARMUP_ITERS=2
LR_DECAY_ITERS=$(( ${TRAIN_ITERS} - ${LR_WARMUP_ITERS}))
OUTPUT_BASEPATH=${EXPERIMENT_DIR}/deepseek-ckpts/test_ft


TRAIN_LOG=${EXPERIMENT_DIR}/MI300X-$MODEL_NAME-${PR}-seq${SEQ_LEN}-tp${TP}pp${PP}ep${EP}-mbs${MBS}gbs${GBS}-ac_${AC}-do_${DO}-fa_${FL}-sp_${SP}-${TIMESTAMP}.log

if [ $ENV = dsw ]; then
export HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
MASTER_ADDR=localhost
MASTER_PORT=$(shuf -n 1 -i 10000-65535)
NNODES=1
NODE_RANK=0
GPUS_PER_NODE=8

elif [ $ENV = dlc ]; then

NNODES=${WORLD_SIZE}
NODE_RANK=${RANK}
GPUS_PER_NODE=${KUBERNETES_CONTAINER_RESOURCE_GPU}

fi

if [ -z ${MP_VP} ]; then
    vp_options=""
else
    vp_options=" \
        --num-layers-per-virtual-pipeline-stage ${MP_VP}"
fi

if [ -z ${MP_SFT_PACKING} ]; then
    MP_SFT_PACKING=false
fi

if [ $MODEL_SIZE = 236B ]; then

HIDDEN_SIZE=5120
NUM_ATTN_HEADS=128
NUM_LAYERS=60
INTERMEDIATE_SIZE=12288
MOE_INTERMEDIATE_SIZE=1536
MAX_POSITION_EMBEDDINGS=${SEQ_LEN}
EXTRA_VOCAB_SIZE=2400
Q_LORA_RANK=1536
KV_LORA_RANK=512
QK_NOPE_HEAD_DIM=128
QK_ROPE_HEAD_DIM=64
V_HEAD_DIM=128
ROPE_THETA=10000
SCALE_FACTOR=40
NUM_EXPERTS=160
ROUTER_TOPK=6
NUM_SHARED_EXPERTS=2
MOE_LAYER_FREQ=1

moe_options=" \
    --moe-ffn-hidden-size ${MOE_INTERMEDIATE_SIZE} \
    --enable-shared-expert \
    --moe-layer-freq ${MOE_LAYER_FREQ} \
    --num-shared-experts ${NUM_SHARED_EXPERTS} \
    --moe-router-topk ${ROUTER_TOPK} \
    --num-experts ${NUM_EXPERTS} \
    --moe-aux-loss-coeff 1e-2 \
    --expert-model-parallel-size ${EP} \
    --q-lora-rank ${Q_LORA_RANK} \
    --kv-lora-rank ${KV_LORA_RANK} \
    --qk-nope-head-dim ${QK_NOPE_HEAD_DIM} \
    --qk-rope-head-dim ${QK_ROPE_HEAD_DIM} \
    --v-head-dim ${V_HEAD_DIM} \
    --moe-router-load-balancing-type aux_loss"


elif [ $MODEL_SIZE = 16B ]; then

HIDDEN_SIZE=2048
NUM_ATTN_HEADS=16
NUM_LAYERS=27
INTERMEDIATE_SIZE=10944
MOE_INTERMEDIATE_SIZE=1408
MAX_POSITION_EMBEDDINGS=${SEQ_LEN}
EXTRA_VOCAB_SIZE=2400
KV_LORA_RANK=512
QK_NOPE_HEAD_DIM=128
QK_ROPE_HEAD_DIM=64
V_HEAD_DIM=128
ROPE_THETA=10000
SCALE_FACTOR=40
NUM_EXPERTS=64
ROUTER_TOPK=6
NUM_SHARED_EXPERTS=2
MOE_LAYER_FREQ=1

moe_options=" \
    --moe-ffn-hidden-size ${MOE_INTERMEDIATE_SIZE} \
    --enable-shared-expert \
    --moe-layer-freq ${MOE_LAYER_FREQ} \
    --num-shared-experts ${NUM_SHARED_EXPERTS} \
    --moe-router-topk ${ROUTER_TOPK} \
    --num-experts ${NUM_EXPERTS} \
    --moe-aux-loss-coeff 1e-2 \
    --expert-model-parallel-size ${EP} \
    --kv-lora-rank ${KV_LORA_RANK} \
    --qk-nope-head-dim ${QK_NOPE_HEAD_DIM} \
    --qk-rope-head-dim ${QK_ROPE_HEAD_DIM} \
    --v-head-dim ${V_HEAD_DIM} \
    --moe-router-load-balancing-type aux_loss"

fi

TP_COMM_OVERLAP=$(( ($TP > 1) ? 1 : 0 ))
comm_overlap_option="\
    --overlap-grad-reduce \
    --overlap-param-gather"

if [ $TP_COMM_OVERLAP -eq 1 ]; then
    comm_overlap_option="\
        --tp-comm-overlap \
        --overlap-grad-reduce \
        --overlap-param-gather"
fi

if [ $AC = full ]; then
    activation_checkpoint_options=" \
		    --recompute-method uniform \
		    --recompute-granularity full \
            --recompute-num-layers 27"
elif [ $AC = sel ]; then
    activation_checkpoint_options=" \
        --recompute-activations"
elif [ $AC = none ]; then
    activation_checkpoint_options=" \
    "
fi

if [ $PR = fp16 ]; then
    pr_options=" \
		    --fp16 \
            --apply-query-key-layer-scaling"
    export NVTE_APPLY_QK_LAYER_SCALING=1
elif [ $PR = bf16 ]; then
    pr_options=" \
        --bf16"
elif [ $PR = fp8 ]; then
    pr_options=" \
        --bf16
        --fp8-hybrid \
        --fp8-amax-compute-algo max \
        --fp8-amax-history-len 1024 \
        --transformer-impl transformer_engine"
fi

if [ $OPTIMIZER_OFFLOAD != false ] && [ $DO = false ]; then
    echo "Offload optimizer is valid only if \$DO=true"
    DO=true
fi

if [ $DO = true ]; then
    do_options=" \
		    --use-distributed-optimizer"

elif [ $DO = false ]; then
    do_options=" \
                    "
fi

if [ $SP = true ] && [ $TP -gt 1 ]; then
    sp_options=" \
		    --sequence-parallel"

elif [ $SP = false ]; then
    sp_options=" \
                    "
fi

if [ $FL = true ]; then
    flash_options=" \
		    --use-flash-attn"

elif [ $FL = false ]; then
    flash_options=" \
                    "
fi

if [ $PRETRAIN_CHECKPOINT_PATH != none ]; then
    load_options=" \
            --load $PRETRAIN_CHECKPOINT_PATH"
fi

if [ $OPTIMIZER_OFFLOAD = 'static' ]; then
    offload_option=" \
        --optimizer hybridadam \
        --optimizer-offload-policy static \
        --optimizer-offload-fraction 1.0"
elif [ $OPTIMIZER_OFFLOAD = 'auto' ]; then
    offload_option=" \
        --optimizer hybridadam \
        --optimizer-offload-policy auto"
else
    offload_option=""
fi

sft_option="--train-mode pretrain"

NAME="${ENV}-finetune-mcore-deepseek-${MODEL_SIZE}-lr-${LR}-bs-${BATCH_SIZE}-seqlen-${SEQ_LEN}-pr-${PR}-tp-${TP}-pp-${PP}-ac-${AC}-do-${DO}-sp-${SP}"
mkdir -p "${OUTPUT_BASEPATH}/tensorboard/"
mkdir -p "${OUTPUT_BASEPATH}/checkpoint/"
mkdir -p "${OUTPUT_BASEPATH}/log/"
current_time=$(date "+%Y.%m.%d-%H.%M.%S")
TENSORBOARD_DIR="${OUTPUT_BASEPATH}/tensorboard/${NAME}_${current_time}"
mkdir -p ${TENSORBOARD_DIR}

SAVED_PRETRAIN_CHECKPOINT_PATH="${OUTPUT_BASEPATH}/checkpoint/${NAME}"

megatron_options="  \
	--log-throughput \
	--no-gradient-accumulation-fusion \
	--no-async-tensor-model-parallel-allreduce \
        --save ${SAVED_PRETRAIN_CHECKPOINT_PATH} \
        --train-data-path ${DATASET_PATH} \
        --valid-data-path ${VALID_DATASET_PATH} \
        --test-data-path ${VALID_DATASET_PATH} \
        --lr ${LR} \
        --min-lr ${MIN_LR} \
        --lr-decay-style cosine \
        --adam-beta1 0.9 \
        --adam-beta2 0.95 \
        --weight-decay 0.1 \
        --clip-grad 1.0 \
        --init-method-std 0.008 \
        --attention-dropout 0.0 \
        --hidden-dropout 0.0 \
        --dataloader-type cyclic \
        --lr-decay-iters ${LR_DECAY_ITERS} \
        --lr-warmup-iters ${LR_WARMUP_ITERS} \
        --train-iters ${TRAIN_ITERS} \
        --micro-batch-size ${BATCH_SIZE} \
        --global-batch-size ${GLOBAL_BATCH_SIZE} \
        --num-layers ${NUM_LAYERS} \
        --hidden-size ${HIDDEN_SIZE} \
        --num-attention-heads ${NUM_ATTN_HEADS} \
        --ffn-hidden-size ${INTERMEDIATE_SIZE} \
        --seq-length ${SEQ_LEN} \
        --max-position-embeddings ${MAX_POSITION_EMBEDDINGS} \
        --max-padding-length ${PAD_LEN} \
        --log-interval 1 \
        --eval-interval 10000 \
        --eval-iters -1 \
        --save-interval ${SAVE_INTERVAL} \
        --tensorboard-queue-size 1 \
        --tensorboard-dir ${TENSORBOARD_DIR} \
        --log-timers-to-tensorboard \
        --log-batch-size-to-tensorboard \
        --log-validation-ppl-to-tensorboard \
        --tensor-model-parallel-size ${TP} \
        --pipeline-model-parallel-size ${PP} \
        --no-load-optim \
        --no-load-rng \
        --num-workers 8 \
        --extra-vocab-size ${EXTRA_VOCAB_SIZE} \
        --patch-tokenizer-type DeepSeekV2Tokenizer \
        --tokenizer-type DeepSeekV2Tokenizer \
        --dataset LLama-Pretrain-Idxmap \
        --swiglu \
        --normalization RMSNorm \
        --norm-epsilon 1e-06 \
        --use-rotary-position-embeddings \
        --no-bias-swiglu-fusion \
        --no-rope-fusion \
        --position-embedding-type rope \
        --untie-embeddings-and-output-weights \
        --disable-bias-linear \
        --use-mcore-models \
        --use-legacy-models \
        --ckpt-format torch \
        --rotary-base ${ROPE_THETA} \
        --rotary-scaling-factor ${SCALE_FACTOR} \
        --transformer-impl transformer_engine \
        --eod-mask-loss
        "


DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE --nnodes $NNODES --node_rank $NODE_RANK --master_addr $MASTER_ADDR --master_port $MASTER_PORT"

run_cmd="torchrun $DISTRIBUTED_ARGS examples/deepseek_v2/pretrain_deepseek.py 
 ${megatron_options} ${pr_options} ${load_options} ${activation_checkpoint_options} \
 ${do_options} ${sp_options} ${moe_options} ${offload_option} ${sft_option} ${vp_options} ${flash_options}"

run_cmd="$run_cmd | tee $TRAIN_LOG"
echo ${run_cmd}
eval ${run_cmd}
set +x

echo 'import argparse
import numpy as np

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
                        prog="Process Log")
    parser.add_argument("filename")
    args = parser.parse_args()

    with open(args.filename) as f:
        lines = f.readlines()
    lines = lines[1:-1]
    lines = [float(a) for a in lines]
    mean = np.mean(np.array(lines))
    print(mean)' > mean_log_value.py


echo '============================================================================================================'
grep -Eo 'throughput per GPU [^|]*' $TRAIN_LOG | sed -E 's/.*throughput per GPU \(TFLOP\/s\/GPU\): ([0-9\.]+).*/\1/' > tmp.txt
echo "throughput per GPU: $(python mean_log_value.py tmp.txt)" |& tee -a $TRAIN_LOG
THROUGHPUT=$(python mean_log_value.py tmp.txt)
rm tmp.txt

echo '============================================================================================================'
grep -Eo 'elapsed time per iteration [^|]*' $TRAIN_LOG | sed -E 's/.*elapsed time per iteration \(ms\): ([0-9\.]+).*/\1/' > tmp.txt
echo "elapsed time per iteration: $(python mean_log_value.py tmp.txt)" |& tee -a $TRAIN_LOG

TIME_PER_ITER=$(python mean_log_value.py tmp.txt 2>/dev/null | awk '{printf "%.6f", $0}')
PERFORMANCE=$(awk -v bs="$BS" -v sl="$SEQ_LENGTH" -v tpi="$TIME_PER_ITER" -v ws="$WORLD_SIZE" 'BEGIN {printf "%.6f", bs * sl * 1000/ (tpi * ws)}')
echo "tokens/GPU/s: $PERFORMANCE" |& tee -a $TRAIN_LOG
rm tmp.txt
