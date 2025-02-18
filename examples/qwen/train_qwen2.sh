#!/bin/bash
###############################################################################
# Copyright (c) 2024, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
#################################################################################
# set -x

# set envs 
export GPU_MAX_HW_QUEUES=2
export TORCH_NCCL_HIGH_PRIORITY=1
export NCCL_CHECKS_DISABLE=1
export NCCL_IB_HCA=rdma0,rdma1,rdma2,rdma3,rdma4,rdma5,rdma6,rdma7 
export NCCL_IB_GID_INDEX=3
export NCCL_CROSS_NIC=0
export CUDA_DEVICE_MAX_CONNECTIONS=1
export NCCL_PROTO=Simple
export RCCL_MSCCL_ENABLE=0
export TOKENIZERS_PARALLELISM=false
export HSA_NO_SCRATCH_RECLAIM=1


# parsing input arguments
for ARGUMENT in "$@"
do
   KEY=$(echo $ARGUMENT | cut -f1 -d=)

   KEY_LENGTH=${#KEY}
   VALUE="${ARGUMENT:$KEY_LENGTH+1}"

   export "$KEY"="$VALUE"
done


TIME_STAMP=$(date +"%Y-%m-%d_%H-%M-%S")
EXP_NAME="${EXP_NAME:-perf}"

TEE_OUTPUT="${TEE_OUTPUT:-1}"
USE_FLASH_ATTN="${USE_FLASH_ATTN:-1}"
NO_TRAINING="${NO_TRAINING:-0}" # NO_TRAINING=1: for computing metrics only
ENABLE_PROFILING="${ENABLE_PROFILING:-0}" #enable pytorch profiling
ENABLE_ROPE="${ENABLE_ROPE:-1}"
echo "NO_TRAINING=$NO_TRAINING"

CWD=`pwd`
GPUS_PER_NODE=`python3 -c "import torch; print(torch.cuda.device_count())"`

# single node config, Change for multinode config
MASTER_ADDR="${MASTER_ADDR:-localhost}"
MASTER_PORT="${MASTER_PORT:-6000}"
NNODES="${NNODES:-1}"
NODE_RANK="${NODE_RANK:-0}"
WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES))

if [ "${NNODES:-1}" -gt 1 ]; then
    export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-ens5}"
    export GLOO_SOCKET_IFNAME="${GLOO_SOCKET_IFNAME:-ens50f0}"
    echo "NCCL and GLOO socket interfaces set."
else
    echo "Single node setup, skipping NCCL and GLOO socket interface settings."
fi

MODEL_SIZE="${MODEL_SIZE:-1.5}"
TP="${TP:-8}"
PP="${PP:-1}"
CP="${CP:-1}"
MBS="${MBS:-2}"
BS="${BS:-8}"
SEQ_LENGTH="${SEQ_LENGTH:-2048}"
MAX_POSITION_EMBEDDINGS="${MAX_POSITION_EMBEDDINGS:-131072}"
TOTAL_ITERS="${TOTAL_ITERS:-10}"
SEQ_PARALLEL="${SEQ_PARALLEL:-1}" 
CONTI_PARAMS="${CONTI_PARAMS:-0}"
TE_FP8="${TE_FP8:-0}"  # 0: disable FP8, 1: enable FP8
GEMM_TUNING="${GEMM_TUNING:-1}"
MOCK_DATA="${MOCK_DATA:-0}"
MCORE="${MCORE:-1}"
RECOMPUTE_ACTIVATIONS="${RECOMPUTE_ACTIVATIONS:-none}" # full or sel
RECOMPUTE_NUM_LAYERS="${RECOMPUTE_NUM_LAYERS:-8}" # only work with full recomputation
DIST_OPTIM="${DIST_OPTIM:-1}" # 0: disable distributed optimizer, 1: enable distributed optimizer
OPTIMIZER="${OPTIMIZER:-adam}" # adam or sgd, by default adam 

if [ "$TOTAL_ITERS" -lt 4 ]; then
    echo "Must give number of iteration greater than 3 to generate peformance data. Exiting..."
    exit 1
fi

TEMP_DIR="temp"
mkdir -p $TEMP_DIR
CHECKPOINT_PATH=${CHECKPOINT_PATH:-"$TEMP_DIR/ckpts"}

DATA_DIR="${DATA_DIR:-./${TEMP_DIR}/qwen-datasets}" # change to where the dataset is stored
DATA_PATH=${DATA_PATH:-"$DATA_DIR/wudao_qwenbpe_text_document"}
DEFAULT_LOG_DIR="${TEMP_DIR}/${NNODES}nodes_rank${NODE_RANK}_train_${MODEL_SIZE}_mbs${MBS}_gbs${BS}_seqlen${SEQ_LENGTH}_tp${TP}_pp${PP}_cp${CP}_iter${TOTAL_ITERS}/TE_FP8_${TE_FP8}/${TIME_STAMP}"
LOG_DIR="${LOG_DIR:-${DEFAULT_LOG_DIR}}"
TRAIN_LOG="${LOG_DIR}/output_${EXP_NAME}.log"
mkdir -p $LOG_DIR
echo $TRAIN_LOG

# gemm tuning 
if [ "$GEMM_TUNING" -eq 1 ]; then
   export TE_HIPBLASLT_TUNING_RUN_COUNT=10
   export TE_HIPBLASLT_TUNING_ALGO_COUNT=50
fi

if [ "$SEQ_LENGTH" -le 8192 ]; then
  ds_works=8
else
  ds_works=24
fi

if [ $MODEL_SIZE = 0.5 ]; then
    HIDDEN_SIZE=896
    FFN_HIDDEN_SIZE=4864
    NUM_LAYERS=24
    NUM_HEADS=14
    NUM_KV_HEADS=2
    TOKENIZER_MODEL=${TOKENIZER_MODEL:-"Qwen/Qwen2-0.5B"}
elif [ $MODEL_SIZE = 1.5 ]; then
    HIDDEN_SIZE=1536
    FFN_HIDDEN_SIZE=8960
    NUM_LAYERS=28
    NUM_HEADS=12
    NUM_KV_HEADS=2
    TOKENIZER_MODEL=${TOKENIZER_MODEL:-"Qwen/Qwen2-1.5B"}
elif [ $MODEL_SIZE = 7 ]; then
    HIDDEN_SIZE=3584
    FFN_HIDDEN_SIZE=18944
    NUM_LAYERS=28
    NUM_HEADS=28
    NUM_KV_HEADS=4
    TOKENIZER_MODEL=${TOKENIZER_MODEL:-"Qwen/Qwen2-7B"}
elif [ $MODEL_SIZE = 72 ]; then
    HIDDEN_SIZE=8192
    FFN_HIDDEN_SIZE=29568
    NUM_LAYERS=80
    NUM_HEADS=64
    NUM_KV_HEADS=8
    TOKENIZER_MODEL=${TOKENIZER_MODEL:-"Qwen/Qwen2-72B"}
else
    echo "Model size not supported. Supported sizes are 0.5, 1.5, 7, 72"
    exit 1
fi

echo "TOKENIZER_MODEL $TOKENIZER_MODEL"
GROUP_SIZE=$(( ${NUM_HEADS} / ${NUM_KV_HEADS} ))
NUM_GROUPS=$(( ${NUM_HEADS} / ${GROUP_SIZE} ))

PROFILING_DIR="${LOG_DIR}/trace_${EXP_NAME}"

GPT_ARGS="
    --tensor-model-parallel-size ${TP} \
    --pipeline-model-parallel-size ${PP} \
    --context-parallel-size ${CP} \
    --num-layers $NUM_LAYERS \
    --hidden-size $HIDDEN_SIZE \
    --ffn-hidden-size $FFN_HIDDEN_SIZE \
    --num-attention-heads $NUM_HEADS \
    --seq-length $SEQ_LENGTH \
    --max-position-embeddings $MAX_POSITION_EMBEDDINGS \
    --untie-embeddings-and-output-weights \
    --position-embedding-type rope \
    --no-position-embedding \
    --disable-bias-linear \
    --swiglu \
    --init-method-std 0.02 \
    --attention-dropout 0.0 \
    --hidden-dropout 0.0 \
    --normalization RMSNorm \
    --micro-batch-size $MBS \
    --global-batch-size $BS \
    --train-iters $TOTAL_ITERS \
    --no-async-tensor-model-parallel-allreduce \
    --bf16 \
    --no-masked-softmax-fusion \
    --disable-bias-linear \
"

# DONT CHANGE THESE ARGS
QWEN_ARGS="
    --add-qkv-bias \
    --rotary-percent 1.0 \
    --rotary-base 1000000 \
    --rotary-seq-len-interpolation-factor 1 \
"

TRAIN_ARGS="--lr 1e-4 \
        --min-lr 1e-5 \
        --lr-decay-iters 320000 \
        --lr-decay-style cosine \
        --weight-decay 1.0e-1 \
        --clip-grad 1.0 \
        --optimizer ${OPTIMIZER} \
"

DATA_ARGS="
    --tokenizer-type HuggingFaceTokenizer \
    --tokenizer-model ${TOKENIZER_MODEL} \
    --dataloader-type cyclic \
    --save-interval 200000 \
    --tensorboard-dir $LOG_DIR \
    --log-interval 1 \
    --eval-interval 320000 \
    --eval-iters 10 \
    --num-workers $ds_works
"

if [ "$MOCK_DATA" -eq 1 ]; then
    echo Using mock data.
    DATA_ARGS="$DATA_ARGS --mock-data"
else
    echo Using data from $DATA_PATH
    DATA_ARGS="$DATA_ARGS --data-path $DATA_PATH"
fi

OUTPUT_ARGS="
    --log-interval 1 \
    --save-interval 5000 \
    --log-throughput \
    --no-save-optim \
    --eval-iters -1   
"

DISTRIBUTED_ARGS="
    --nproc_per_node $GPUS_PER_NODE \
    --nnodes $NNODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT \
"

CKPT_LOAD_ARGS="--exit-on-missing-checkpoint \
        --no-load-optim \
        --use-checkpoint-args \
        --no-load-rng"


EXTRA_ARGS="
    --group-query-attention \
    --num-query-groups $NUM_GROUPS \
    --no-gradient-accumulation-fusion \
    --distributed-backend nccl \
    --distributed-timeout-minutes 120 \
"

if [ $RECOMPUTE_ACTIVATIONS = full ]; then
    EXTRA_ARGS="$EXTRA_ARGS --recompute-method uniform --recompute-granularity full --recompute-num-layers ${RECOMPUTE_NUM_LAYERS}"
elif [ $RECOMPUTE_ACTIVATIONS = sel ]; then
    EXTRA_ARGS="$EXTRA_ARGS --recompute-activations"
fi

if [ $DIST_OPTIM -eq 1 ]; then
    EXTRA_ARGS="$EXTRA_ARGS --use-distributed-optimizer --overlap-param-gather --overlap-grad-reduce"
fi

if [ "$ENABLE_PROFILING" -eq 1 ]; then
EXTRA_ARGS="$EXTRA_ARGS --profile --use-pytorch-profiler --tensorboard-dir $LOG_DIR"
fi

if [ "$USE_FLASH_ATTN" -eq 1 ]; then
EXTRA_ARGS="$EXTRA_ARGS --use-flash-attn"
fi

if [ "$SEQ_PARALLEL" -eq 1 ]; then
EXTRA_ARGS="$EXTRA_ARGS --sequence-parallel"
fi

if [ "$CONTI_PARAMS" -eq 1 ]; then
EXTRA_ARGS="$EXTRA_ARGS --use-contiguous-parameters-in-local-ddp"
fi

if [ "$MCORE" -eq 1 ]; then
EXTRA_ARGS="$EXTRA_ARGS --use-mcore-models"
fi

if [ "$ENABLE_ROPE" -eq 1 ]; then
EXTRA_ARGS="$EXTRA_ARGS --position-embedding-type rope"
fi

if [ "$TE_FP8" -eq 1 ]; then
EXTRA_ARGS="$EXTRA_ARGS --transformer-impl=transformer_engine \
    --fp8-margin=0 \
    --fp8-format=hybrid \
    --fp8-interval=1 \
    --fp8-amax-history-len=1024 \
    --fp8-amax-compute-algo=max \
    --attention-softmax-in-fp32 \
"
fi

run_cmd="
    torchrun $DISTRIBUTED_ARGS pretrain_gpt.py \
        $GPT_ARGS \
        $QWEN_ARGS \
        $DATA_ARGS \
        $OUTPUT_ARGS \
        $EXTRA_ARGS \
        $TRAIN_ARGS \
"

if [ "$TEE_OUTPUT" -eq 0 ]; then 
    run_cmd="$run_cmd >& $TRAIN_LOG"
else
    run_cmd="$run_cmd |& tee $TRAIN_LOG"
fi

if [ "$NO_TRAINING" -eq 0 ]; then 
    eval $run_cmd
fi

echo 'import argparse
import numpy as np

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
                        prog="Process Log")
    parser.add_argument("filename")
    args = parser.parse_args()

    with open(args.filename) as f:
        lines = f.readlines()
    lines = lines[2:-1]
    lines = [float(a) for a in lines]
    mean = np.mean(np.array(lines))
    print(mean)' > mean_log_value.py

# echo '============================================================================================================'
grep -Eo 'throughput per GPU [^|]*' $TRAIN_LOG | sed -E 's/.*throughput per GPU \(TFLOP\/s\/GPU\): ([0-9\.]+).*/\1/' > ${TEMP_DIR}/tmp.txt
PERFORMANCE=$(python3 mean_log_value.py ${TEMP_DIR}/tmp.txt)
echo "throughput per GPU: $PERFORMANCE" |& tee -a $TRAIN_LOG
rm ${TEMP_DIR}/tmp.txt

# echo '============================================================================================================'
grep -Eo 'elapsed time per iteration [^|]*' $TRAIN_LOG | sed -E 's/.*elapsed time per iteration \(ms\): ([0-9\.]+).*/\1/' > ${TEMP_DIR}/tmp.txt
ETPI=$(python3 mean_log_value.py ${TEMP_DIR}/tmp.txt)
echo "elapsed time per iteration: $ETPI" |& tee -a $TRAIN_LOG


TIME_PER_ITER=$(python3 mean_log_value.py ${TEMP_DIR}/tmp.txt 2>/dev/null | awk '{printf "%.6f", $0}')
TGS=$(awk -v bs="$BS" -v sl="$SEQ_LENGTH" -v tpi="$TIME_PER_ITER" -v ws="$WORLD_SIZE" 'BEGIN {printf "%.6f", bs * sl * 1000/ (tpi * ws)}')
echo "tokens/GPU/s: $TGS" |& tee -a $TRAIN_LOG
rm ${TEMP_DIR}/tmp.txt

