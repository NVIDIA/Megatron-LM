#!/usr/bin/bash
ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}" )/../.." && pwd )"

set -e
set -x

# TOOD (yiakwy) : add NCCL args
export NCCL_IB_HCA=mlx5
export NCCL_IB_TC=136
export NCCL_IB_SL=5
export NCCL_IB_GID_INDEX=3
export NCCL_IB_TIMEOUT=22
export NCCL_IB_SPLIT_DATA_ON_QPS=1
export NCCL_IB_QPS_PER_CONNECTION=8
export NCCL_IB_RETRY_CNT=13
export NCCL_SOCKET_IFNAME=ens #/*NOTE*/
export NCCL_DEBUG=INFO

# export NCCL_IB_HCA=ibp
# export UCX_NET_DEVICES=ibp0:1,ibp1:1,ibp2:1,ibp3:1,ibp4:1,ibp5:1,ibp6:1,ibp7:1
export SHARP_COLL_ENABLE_PCI_RELAXED_ORDERING=1
export NCCL_COLLNET_ENABLE=0

# This is needed to avoid NCCL to use ifiniband, which the cluster is not ready
export NCCL_IB_DISABLE=1

DIST_ENV=${DIST_ENV:-dsw}
echo "DIST_ENV : $DIST_ENV"

MEGATRON_PATCH_PATH=$MEGATRON_PATCH_PATH
echo "MEGATRON_PATCH_PATH : $MEGATRON_PATCH_PATH"
if [ -z "$MEGATRON_PATCH_PATH" ]; then
    echo "Error : MEGATRON_PATCH_PATH is not set"
    MEGATRON_PATCH_PATH=$ROOT
    echo "setting MEGATRON_PATCH_PATH to $MEGATRON_PATCH_PATH" 
else
    echo "Ok"
fi

MEGATRON_PATH=${MEGATRON_PATCH_PATH}

# add python path
export PYTHONPATH=${MEGATRON_PATH}:${MEGATRON_PATCH_PATH}:$PYTHONPATH

# TODO (yaikwy) : this is required by sequence parallelism , consider to deprecate the option in favor of context parallel
export CUDA_DEVICE_MAX_CONNECTIONS=1

if [ $DIST_ENV = dsw ]; then
# export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export CUDA_VISIBLE_DEVICES=4,5,6,7
MASTER_ADDR=localhost
MASTER_PORT=$(shuf -n 1 -i 10000-65535)
NNODES=1
NODE_RANK=0
# GPUS_PER_NODE=8
GPUS_PER_NODE=4

elif [ $DIST_ENV = dlc ]; then

NNODES=${WORLD_SIZE}
NODE_RANK=${RANK}
GPUS_PER_NODE=${KUBERNETES_CONTAINER_RESOURCE_GPU}

fi

DISTRIBUTED_ARGS="
    --nproc_per_node $GPUS_PER_NODE \
    --nnodes $NNODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT
"

# TODO (yiakwy) : decoupled from llama-2-7b model, used to define GPT control args : 
#   --num-layers, 
#   --hidden-size,
#   --num-attention-heads,
#   --ffn-hidden-size
MODEL_SIZE=${MODEL_SIZE:-7B}
echo "MODEL_SIZE : $MODEL_SIZE"
if [ -z "$MODEL_SIZE" ]; then
    echo "Error : MODEL_SIZE is not set"
    exit 1
fi

# TODO (yiakwy) : renamed to MICRO_BSZ, this is micro batch size NOT batch size
BATCH_SIZE=${MICRO_BATCH_SIZE:-1}
echo "MICRO BATCH SIZE: $BATCH_SIZE"

# TODO (yiakwy) : renamed to GLOBAL_BSZ
GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE:-2048}
echo "GLOBAL BATCH SIZE: $GLOBAL_BATCH_SIZE"

# peak lr
LR=${LR:-9e-7}
# minimum lr
MIN_LR=${MIN_LR:-9e-8}

# TODO (yiakwy) : finetune adam beta
B1=0.9
B2=0.95

# 2K
SEQ_LEN=${SEQ_LEN:-2048}
echo "SEQ_LEN : $SEQ_LEN"

# TODO (yiakwy) : add new tokens to vocabulary to scale up, standard llama has vocabulary size of 32k
EXTRA_VOCAB_SIZE=${EXTRA_VOCAB_SIZE:-0}

# precision
PR=${PR:-bf16}

TP=${TP:-1}
echo "TP : $TP"

PP=${PP:-1}
echo "PP : $PP"

# TODO (yiakwy) : add DP, ACC
DP=$((NNODES * GPUS_PER_NODE / TP / PP))
ACC=$((GLOBAL_BATCH_SIZE / DP / BATCH_SIZE))
echo "DP : $DP"
echo "ACC : $ACC"

# TODO (yiakwy) : rename to AUTO_RECOMPUTE_OPT
AC=${AUTO_RECOMPUTE_OPT:-sel} 
echo "AUTO_RECOMPUTE_OPT : $AC"

# TODO (yiakwy) : rename to DIST_OPT, --use-distributed-optimizer
DO=${DIST_OPT:-true}

# TODO (yiakwy) : rename to USE_FLASH_ATTN, --use-flash-attn
FL=${USE_FLASH_ATTN:-true}

# TODO (yiakwy) : disable SP, in favor of CP
SP=${SP:-false}

# TODO (yiakwy) : add support TE equivalent functions (FP8) in ROCm
TE=${TE:-false}

SAVE_INTERVAL=${SAVE_INTERVAL:-50}

# megatron dataset path
DATASET_PATH=${DATASET_PATH}
echo "DATASET_PATH : $DATASET_PATH"
if [ -z "$DATASET_PATH" ];then
  echo "WARN : DATASET_PATH is not set, using mocked dataset"
  load_dataset="--mock-data"
else
  load_dataset="--train-data-path ${DATASET_PATH}"
fi

# TODO (yiakwy) : add control, this is only used in SFT task
PRETRAIN_CHECKPOINT_PATH=${PRETRAIN_CHECKPOINT_PATH}
echo "PRETRAIN_CHECKPOINT_PATH : $PRETRAIN_CHECKPOINT_PATH"
if [ -z "$PRETRAIN_CHECKPOINT_PATH" ];then
  echo "NOTE : PRETRAIN_CHECKPOINT_PATH is not set, switch to pretrain mode"
  EXTRA_ARGS=""

  load_options=""
else
  EXTRA_ARGS="
    --finetune \
    --no-load-optim
  "

  load_options="
    --load $PRETRAIN_CHECKPOINT_PATH"
fi

# TODO (yiakwy) : add support blending dataset to scale up to 1 trillion tokens, in this test alibaba recommends 1/10 B tokens
TRAIN_TOKENS=${TRAIN_TOKENS:-10000000}
echo "TRAIN_TOKENS : $TRAIN_TOKENS"

WARMUP_TOKENS=${WARMUP_TOKENS:-1000}

DEFAULT_OUTPUT_BASEPATH="/workspace/logs/llama-2-7b_tp${TP}_pp${PP}_dp${DP}_MBSZ${BATCH_SIZE}_ACC${ACC}-profiling"

OUTPUT_BASEPATH=$DEFAULT_OUTPUT_BASEPATH

if [ $MODEL_SIZE = 7B ]; then

NUM_LAYERS=32
HIDDEN_SIZE=4096
NUM_ATTN_HEADS=32
INTERMEDIATE_SIZE=11008
MAX_POSITION_EMBEDDINGS=4096

gqa_options=""

elif [ $MODEL_SIZE = 13B ]; then

NUM_LAYERS=40
HIDDEN_SIZE=5120
NUM_ATTN_HEADS=40
INTERMEDIATE_SIZE=13824
MAX_POSITION_EMBEDDINGS=4096

gqa_options=""

elif [ $MODEL_SIZE = 70B ]; then

NUM_LAYERS=80
HIDDEN_SIZE=8192
NUM_ATTN_HEADS=64
INTERMEDIATE_SIZE=28672
MAX_POSITION_EMBEDDINGS=4096

gqa_options=" \
		--group-query-attention \
		--num-query-groups 8"
fi

if [ $AC = full ]; then
    activation_checkpoint_options=" \
		--recompute-method uniform \
		--recompute-granularity full"
elif [ $AC = sel ]; then
    activation_checkpoint_options=" \
        --recompute-activations"
elif [ $AC = none ]; then
    activation_checkpoint_options=" \
                    "
fi

if [ $PR = fp16 ]; then
    pr_options=" \
		--fp16"
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

if [ $DO = true ]; then
    do_options=" \
		--use-distributed-optimizer"

elif [ $DO = false ]; then
    do_options=" \
                    "
fi

if [ $FL = true ]; then
    flash_options=" \
		--use-flash-attn"

elif [ $FL = false ]; then
    flash_options=" \
                    "
fi

if [ $TE = true ]; then
    te_options=" \
		--transformer-impl transformer_engine"

elif [ $TE = false ]; then
    te_options=" \
        --transformer-impl local"
fi

if [ $SP = true ] && [ $TP -gt 1 ]; then
    sp_options=" \
		--sequence-parallel"

elif [ $SP = false ]; then
    sp_options=" \
                    "
fi

mkdir -p $OUTPUT_BASEPATH/index_mapping

DATA_ARGS="
  $load_dataset \
  --data-cache-path $OUTPUT_BASEPATH/index_mappings \
  --num-workers 5
"

# TODO (yiakwy) : add timing log level
LOOGING_ARGS="
  --timing-log-level 2 \
  --log-throughput
"

MODEL=Llama-2-7b-hf
TOK=Llama2Tokenizer
TOKENIZER_MODEL=/workspace/models/$MODEL/tokenizer.model

TRAIN_ITERS=$(( ${TRAIN_TOKENS} / ${GLOBAL_BATCH_SIZE} / ${SEQ_LEN} ))
LR_WARMUP_ITERS=$(( ${WARMUP_TOKENS} / ${GLOBAL_BATCH_SIZE} / ${SEQ_LEN} ))
LR_DECAY_ITERS=$(( ${TRAIN_TOKENS} / ${GLOBAL_BATCH_SIZE} / ${SEQ_LEN} ))

NAME="${DIST_ENV}-pretrain-megatron-llama-2-7b-${MODEL_SIZE}-lr-${LR}-bs-${BATCH_SIZE}-seqlen-${SEQ_LEN}-pr-${PR}-tp-${TP}-pp-${PP}-ac-${AC}-do-${DO}-sp-${SP}-tt-${TRAIN_TOKENS}-wt-${WARMUP_TOKENS}"
mkdir -p "${OUTPUT_BASEPATH}/tensorboard/"
mkdir -p "${OUTPUT_BASEPATH}/checkpoint/"
mkdir -p "${OUTPUT_BASEPATH}/log/"
current_time=$(date "+%Y.%m.%d-%H.%M.%S")
TENSORBOARD_DIR="${OUTPUT_BASEPATH}/tensorboard/${NAME}_${current_time}"
mkdir -p ${TENSORBOARD_DIR}

SAVED_PRETRAIN_CHECKPOINT_PATH="${OUTPUT_BASEPATH}/checkpoint/${NAME}"

megatron_options="  \
        --tensor-model-parallel-size ${TP} \
        --pipeline-model-parallel-size ${PP} \
        --seq-length ${SEQ_LEN} \
        --max-position-embeddings ${MAX_POSITION_EMBEDDINGS} \
        --num-layers ${NUM_LAYERS} \
        --hidden-size ${HIDDEN_SIZE} \
        --num-attention-heads ${NUM_ATTN_HEADS} \
        --ffn-hidden-size ${INTERMEDIATE_SIZE} \
        --swiglu \
        --normalization RMSNorm \
        --optimizer adam \
        --micro-batch-size ${BATCH_SIZE} \
        --global-batch-size ${GLOBAL_BATCH_SIZE} \
        --lr ${LR} \
        --min-lr ${MIN_LR} \
        --lr-decay-style linear \
        --adam-beta1 $B1 \
        --adam-beta2 $B2 \
        --weight-decay 0.1 \
        --clip-grad 1.0 \
        --init-method-std 0.01 \
        --lr-decay-iters ${LR_DECAY_ITERS} \
        --lr-warmup-iters ${LR_WARMUP_ITERS} \
        --train-iters ${TRAIN_ITERS} \
        --seed 1234 \
        --tokenizer-type $TOK \
        --tokenizer-model $TOKENIZER_MODEL \
        --use-rotary-position-embeddings \
        --rotary-percent 1.0 \
        --no-load-rng \
        --no-masked-softmax-fusion \
        --position-embedding-type rope \
        --untie-embeddings-and-output-weights \
        --disable-bias-linear \
        --initial-loss-scale 4096 \
        --attention-softmax-in-fp32 \
        --use-legacy-models
        "

OUTPUT_ARGS="
        --log-interval 1 \
        --eval-interval 10000 \
        --eval-iters 0 \
        --save-interval ${SAVE_INTERVAL}
"

LOGGING_ARGS="
        --log-timers-to-tensorboard \
        --log-batch-size-to-tensorboard \
        --log-validation-ppl-to-tensorboard \
        --tensorboard-queue-size 1 \
        --tensorboard-dir ${TENSORBOARD_DIR}
"

cat $0 > ${OUTPUT_BASEPATH}/launch_script.sh
git config --global --add safe.directory $ROOT
echo "COMMIT_ID=$(git rev-parse HEAD)" >> ${OUTPUT_BASEPATH}/commit_id.txt

torchrun $DISTRIBUTED_ARGS $MEGATRON_PATH/pretrain_gpt.py \
 ${megatron_options} \
 ${pr_options} \
 ${load_options} \
 ${te_options} \
 ${activation_checkpoint_options} \
 ${do_options} \
 ${flash_options} \
 ${sp_options} \
 ${gqa_options} \
 ${EXTRA_ARGS} \
 ${LOOGING_ARGS} \
 ${DATA_ARGS} \
 ${OUTPUT_ARGS} \
 ${LOGGING_ARGS} \
 &> ${OUTPUT_BASEPATH}/log/${NODE_RANK}.log

set +e
set +x
