#!/bin/bash
###############################################################################
# Copyright (c) 2024, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
#################################################################################
#set -x

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
    export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-ens51np0}"
    export GLOO_SOCKET_IFNAME="${GLOO_SOCKET_IFNAME:-ens51np0}"
    echo "NCCL and GLOO socket interfaces set."
else
    echo "Single node setup, skipping NCCL and GLOO socket interface settings."
fi

MODEL_SIZE="${MODEL_SIZE:-70}"
TP="${TP:-8}"
PP="${PP:-1}"
CP="${CP:-1}"
MBS="${MBS:-2}"
BS="${BS:-8}"
SEQ_LENGTH="${SEQ_LENGTH:-2048}"
TOTAL_ITERS="${TOTAL_ITERS:-10}"
SEQ_PARALLEL="${SEQ_PARALLEL:-1}" 
CONTI_PARAMS="${CONTI_PARAMS:-0}"
TE_FP8="${TE_FP8:-0}"  # 0: disable FP8, 1: enable FP8
GEMM_TUNING="${GEMM_TUNING:-1}"
MCORE="${MCORE:-1}"
OPTIMIZER="${OPTIMIZER:-adam}"
FSDP="${FSDP:-0}"
RECOMPUTE="${RECOMPUTE:-0}"
ROPE_FUSION="${ROPE_FUSION:-1}" # 1: use rope-fusion, 0: no-rope-fusion
MOCK_DATA="${MOCK_DATA:-1}" # 1: use mock data, 0: use real data

if [ "$FSDP" -eq 1 ]; then
    if [ "$TP" -gt 1 ]; then
        echo "It is not recommended to use FSDP and TP together. Disabling TP."
        TP=1
        echo "Resetting TP=$TP"
    fi
fi

EXPERIMENT_DIR="experiment"
mkdir -p $EXPERIMENT_DIR
CHECKPOINT_PATH=${CHECKPOINT_PATH:-"$EXPERIMENT_DIR/ckpts"}

DATA_DIR="${DATA_DIR:-/root/.cache/data}"
TOKENIZER_MODEL="${TOKENIZER_MODEL:-"$DATA_DIR/tokenizer_llama3"}"
# Download the tokenizer model
if ! [ -d "$TOKENIZER_MODEL" ]; then
  mkdir -p $TOKENIZER_MODEL
  HF_TOKEN="${HF_TOKEN:-hf_xxxx}" #set huggingface access token to be able to download tokenizer
  wget --header="Authorization: Bearer $HF_TOKEN" -O $TOKENIZER_MODEL/special_tokens_map.json https://huggingface.co/meta-llama/Llama-3.1-8B/resolve/main/special_tokens_map.json
  wget --header="Authorization: Bearer $HF_TOKEN" -O $TOKENIZER_MODEL/tokenizer.json https://huggingface.co/meta-llama/Llama-3.1-8B/resolve/main/tokenizer.json
  wget --header="Authorization: Bearer $HF_TOKEN" -O $TOKENIZER_MODEL/tokenizer.model https://huggingface.co/meta-llama/Llama-3.1-8B/resolve/main/original/tokenizer.model
  wget --header="Authorization: Bearer $HF_TOKEN" -O $TOKENIZER_MODEL/tokenizer_config.json https://huggingface.co/meta-llama/Llama-3.1-8B/resolve/main/tokenizer_config.json

  echo "Tokenizer files downloaded successfully to $TOKENIZER_MODEL."
else
  echo "Folder $TOKENIZER_MODEL already exists. Skipping download."
fi

DATA_PATH=${DATA_PATH:-"$DATA_DIR/bookcorpus_text_sentence"}

MAX_POSITION_EMBEDDINGS=131072

DEFAULT_LOG_DIR="${EXPERIMENT_DIR}/${NNODES}nodes_rank${NODE_RANK}_train_${MODEL_SIZE}B_mbs${MBS}_bs${BS}_tp${TP}_pp${PP}_cp${CP}_iter${TOTAL_ITERS}/TE_FP8_${TE_FP8}/${TIME_STAMP}"
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

if [[ $MODEL_SIZE -eq 8 ]]; then #llama3.1-8B
        HIDDEN_SIZE=4096 # e.g. llama-13b: 5120
        FFN_HIDDEN_SIZE=14336 # e.g. llama-13b: 13824
        NUM_LAYERS=32 # e.g. llama-13b: 40
        NUM_HEADS=32 # e.g. llama-13b: 40
        NUM_KV_HEADS=8 
        SEQ_LENGTH=$SEQ_LENGTH
elif [[ $MODEL_SIZE -eq 70 ]]; then
        HIDDEN_SIZE=8192 # e.g. llama-13b: 5120
        FFN_HIDDEN_SIZE=28672 # e.g. llama-13b: 13824
        NUM_LAYERS=80 # e.g. llama-13b: 40
        NUM_HEADS=64 # e.g. llama-13b: 40
        NUM_KV_HEADS=8 # llama3 70B uses GQA 
        SEQ_LENGTH=$SEQ_LENGTH
        MAX_POSITION_EMBEDDINGS=$MAX_POSITION_EMBEDDINGS
else
        echo "Model size not supported."
        exit 1
fi

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
"

if [ "$RECOMPUTE" -eq 1 ]; then
    GPT_ARGS="$GPT_ARGS --recompute-num-layers $NUM_LAYERS \
        --recompute-granularity full \
        --recompute-method block \
        "
fi 

if [ "$ROPE_FUSION" -eq 0 ]; then
    GPT_ARGS="$GPT_ARGS --no-rope-fusion"
fi

TRAIN_ARGS="--lr 1e-4 \
        --min-lr 1e-5 \
        --lr-decay-iters 320000 \
        --lr-decay-style cosine \
        --weight-decay 1.0e-1 \
        --clip-grad 1.0 \
        --ckpt-format torch_dist \
"


if [ "$OPTIMIZER" == "adam" ]; then
    TRAIN_ARGS="$TRAIN_ARGS --optimizer adam \
        --adam-beta1 0.9 \
        --adam-beta2 0.95 \
        "
else
    TRAIN_ARGS="$TRAIN_ARGS --optimizer sgd \
        "
fi

DATA_ARGS="
    --tokenizer-type HuggingFaceTokenizer \
    --tokenizer-model ${TOKENIZER_MODEL} \
    --dataloader-type cyclic \
    --save-interval 200000 \
    --tensorboard-dir $LOG_DIR \
    --log-interval 1 \
    --eval-interval 320000 \
    --eval-iters 10 \
    --num-workers $ds_works \
"
# For multi-node runs DATA_CACHE_PATH should point to a common path accessible by all the nodes (for eg, an NFS directory)
DATA_CACHE_PATH="/home/cache"

if [ "$MOCK_DATA" -eq 1 ];then
    DATA_ARGS="$DATA_ARGS --mock-data --data-cache-path $DATA_CACHE_PATH"
else
    DATA_ARGS="$DATA_ARGS --data-path $DATA_PATH --data-cache-path ${DATA_CACHE_PATH}"
fi

OUTPUT_ARGS="
    --log-interval 1 \
    --save-interval 5000 \
    --log-throughput \
    --no-save-optim \
    --eval-iters -1   
"
#  --save $CHECKPOINT_PATH \

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
    --overlap-grad-reduce \
"

if [ "$FSDP" -eq 1 ]; then
    EXTRA_ARGS="$EXTRA_ARGS --use-torch-fsdp2"
    if [ "$SEQ_PARALLEL" -eq 1 ]; then
        echo "Warning: Sequence Parallelism and FSDP2 have conflicting CUDA_MAX_CONNECTIONS requirements. It is recommended not to use them together."
        echo "FSDP2 and sequence parallel are on. Disabling sequence parallel."
        SEQ_PARALLEL=0
    fi
else
    if [ "$OPTIMIZER" == "adam" ]; then
        EXTRA_ARGS="$EXTRA_ARGS --use-distributed-optimizer --overlap-param-gather"
    fi
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
grep -Eo 'throughput per GPU [^|]*' $TRAIN_LOG | sed -E 's/.*throughput per GPU \(TFLOP\/s\/GPU\): ([0-9\.]+).*/\1/' > tmp.txt
PERFORMANCE=$(python3 mean_log_value.py tmp.txt)
echo "throughput per GPU: $PERFORMANCE" |& tee -a $TRAIN_LOG
rm tmp.txt

# echo '============================================================================================================'
grep -Eo 'elapsed time per iteration [^|]*' $TRAIN_LOG | sed -E 's/.*elapsed time per iteration \(ms\): ([0-9\.]+).*/\1/' > tmp.txt
ETPI=$(python3 mean_log_value.py tmp.txt)
echo "elapsed time per iteration: $ETPI" |& tee -a $TRAIN_LOG

TIME_PER_ITER=$(python3 mean_log_value.py tmp.txt 2>/dev/null | awk '{printf "%.6f", $0}')
TGS=$(awk -v bs="$BS" -v sl="$SEQ_LENGTH" -v tpi="$TIME_PER_ITER" -v ws="$WORLD_SIZE" 'BEGIN {printf "%.6f", bs * sl * 1000/ (tpi * ws)}')
echo "tokens/GPU/s: $TGS" |& tee -a $TRAIN_LOG
rm tmp.txt

# Extract memory usage
grep -Eo 'mem usages: [^|]*' "$TRAIN_LOG" | sed -E 's/.*mem usages: ([0-9\.]+).*/\1/' > tmp.txt
MEMUSAGE=$(python3 mean_log_value.py tmp.txt)
echo "mem usages: $MEMUSAGE" |& tee -a "$TRAIN_LOG"
rm tmp.txt