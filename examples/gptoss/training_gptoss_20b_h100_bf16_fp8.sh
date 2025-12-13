#!/usr/bin/bash
ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}" )/../.." && pwd )"

set -e
set -x

## NCCL config
export NCCL_IB_HCA=mlx5_0,mlx5_3,mlx5_4,mlx5_5,mlx5_6,mlx5_9,mlx5_10,mlx5_11
# export NCCL_IB_HCA=mlx5
# export NCCL_TOPO_DUMP_FILE=topo.xml
# traffic class for QoS tunning
export NCCL_IB_TC=136
# service level that maps virtual lane
export NCCL_IB_SL=5
export NCCL_IB_GID_INDEX=3
export NCCL_IB_CUDA_SUPPORT=1
export NCCL_IB_TIMEOUT=22
# for HKUST supper pod, and this sets the TCP/IP-based interface for fallback of socket-based NCCL communication
# 'ibp154s0'(tcp), 'ibp170s0f0'(tcp), 'ibp192s0'(tcp), 'ibp206s0'(tcp), 'ibp220s0'(tcp), 'ibp24s0'(tcp), 'ibp41s0f0'(tcp), 'ibp64s0'(tcp), 'ibp79s0'(tcp), 'ibp94s0'(tcp)
# NOTE(yiakwy) : see ib device and roce device mapping via ibdev2netdev
export NCCL_SOCKET_IFNAME=ibp24s0,ibp41s0f0,ibp64s0,ibp79s0,ibp94s0,ibp154s0,ibp170s0f0,ibp192s0
# export UCX_NET_DEVICES=$NCCL_SOCKET_IFNAME
export NCCL_SOCKET_IFNAME=ibp #/*NOTE*/
export NCCL_DEBUG=DEBUG

export SHARP_COLL_ENABLE_PCI_RELAXED_ORDERING=1
export NCCL_COLLNET_ENABLE=1

export NCCL_IB_DISABLE=1

DIST_ENV=${DIST_ENV:-dsw}
echo "DIST_ENV : $DIST_ENV"


MEGATRON_PATH=$ROOT # ${MEGATRON_PATCH_PATH}


# add python path
export PYTHONPATH=${MEGATRON_PATH}:$PYTHONPATH

export CUDA_DEVICE_MAX_CONNECTIONS=1

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

if [ $DIST_ENV = dsw ]; then

MASTER_ADDR=localhost
MASTER_PORT=$(shuf -n 1 -i 10000-65535)
NNODES=1
NODE_RANK=0
GPUS_PER_NODE=8

elif [ $DIST_ENV = dlc ]; then

NNODES=${WORLD_SIZE}
NODE_RANK=${RANK}
GPUS_PER_NODE=${KUBERNETES_CONTAINER_RESOURCE_GPU}

fi

# megatron world size def
WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES))

## Model Parallel
TP=1
PP=1
CP=1
EDP=8

# TODO (yiakwy) : rename to DIST_OPT, --use-distributed-optimizer
DO=${DIST_OPT:-true}

TE=${TE:-true} # ${TE:-false}

PRETRAIN_CHECKPOINT_PATH="/raid/gpt-oss-20b-BF16-to-mcore_bridge-tp1-pp1-cp1-ep8-bf16/iter_0000000"
# PRETRAIN_CHECKPOINT_PATH="/raid/gpt-oss-20b-BF16-to-mcore_bridge-tp1-pp2-cp1-ep4-bf16/iter_0000000"

OUTPUT_BASEPATH=output


DISTRIBUTED_ARGS="
    --nproc_per_node $GPUS_PER_NODE \
    --nnodes $NNODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT \
"

EXTRA_ARGS="
    --finetune \
    --no-load-optim \
    --no-load-rng
"

load_options="
    --load $PRETRAIN_CHECKPOINT_PATH"

DATASET_PATH=${DATASET_PATH}
echo "DATASET_PATH : $DATASET_PATH"
if [ -z "$DATASET_PATH" ];then
  echo "WARN : DATASET_PATH is not set, using mocked dataset"
  load_dataset="--mock-data"
else
  load_dataset="--train-data-path ${DATASET_PATH}"
fi


if [ $DO = true ]; then
    do_options=" \
		--use-distributed-optimizer"
elif [ $DO = false ]; then
    do_options=" \
                    "
fi


if [ $TE = true ]; then
    te_options=" \
		--transformer-impl transformer_engine"
elif [ $TE = false ]; then
    te_options=" \
        --transformer-impl local"
fi


#     --fp8-param-gather \
pr_options=" \
    --bf16
    --fp8-format hybrid \
    --fp8-param-gather \
    --fp8-amax-compute-algo max \
    --fp8-amax-history-len 1024"


# pr_options=" \
#     --bf16
#     --fp8-format hybrid"


# NOTE (yiakwy) : wierd options, see https://github.com/NVIDIA/Megatron-LM/commit/a2d8c806b35bc708b13e6c069e19e5dfb49b8481#r171154625
#     --seq-length 4096 \
#     --max-position-embeddings 40960 \

#     --seq-length 131072 \
GPT_OSS_SFT_MODEL_ARGS=" \
    --no-masked-softmax-fusion \
    --untie-embeddings-and-output-weights \
    --no-rope-fusion \
    --normalization RMSNorm \
    --num-layers 24 \
    --hidden-size 2880 \
    --ffn-hidden-size 2880 \
    --num-attention-heads 64 \
    --group-query-attention \
    --num-query-groups 8 \
    --kv-channels 64 \
    --seq-length 4096 \
    --max-position-embeddings 4096 \
    --make-vocab-size-divisible-by 128 \
    --use-mcore-models \
    --rotary-percent 1.0 \
    --rotary-base 150000 \
    --no-bias-gelu-fusion \
    --sequence-parallel \
    --export-force-local-attention \
    --no-bias-dropout-fusion \
    --padded-vocab-size 201088 \
    --quick-geglu \
    --glu-linear-offset 1.0 \
    --softmax-type learnable \
    --window-attn-skip-freq 2 \
    --activation-func-clamp-value 7.0 \
    --window-size 128,0 \
    --enable-gpt-oss \
"

MODEL_ARGS=(
    --init-method-std 0.01
    --hidden-dropout 0.0
    --attention-dropout 0.0
    --rope-type yarn
    --position-embedding-type yarn
)

#    --moe-aux-loss-coeff 1e-2
#    --moe-router-pre-softmax
MOE_ARGS=(
    --num-experts 32
    --moe-router-load-balancing-type none # options: aux_loss, sinkhorn, None. Default is aux_loss.
    --moe-router-topk 4
    --moe-aux-loss-coeff 0.0
    --moe-grouped-gemm
    --moe-permute-fusion
    --moe-ffn-hidden-size 2880
    --moe-router-dtype fp32
    --moe-token-dispatcher-type alltoall
    --moe-router-score-function softmax
)

mkdir -p $OUTPUT_BASEPATH/index_mapping

DATA_ARGS=(
    $load_dataset
    --data-cache-path $OUTPUT_BASEPATH/index_mappings
    --num-workers 5
    --tokenizer-type HuggingFaceTokenizer
    --tokenizer-model "$PRETRAIN_CHECKPOINT_PATH/tokenizer"
)

TRAINING_ARGS=(
    --micro-batch-size 1
    --global-batch-size 128
    --lr 1e-4
    --train-iters 500000
    --lr-decay-iters 320000
    --lr-decay-style cosine
    --min-lr 1.0e-5
    --weight-decay 0.1
    --lr-warmup-iters 500
    --clip-grad 1.0
    --overlap-grad-reduce
    --overlap-param-gather
)

MODEL_PARALLEL_ARGS=(
    --tensor-model-parallel-size $TP
    --pipeline-model-parallel-size $PP
    --context-parallel-size $CP
    --expert-model-parallel-size $EDP
    --sequence-parallel
    --use-distributed-optimizer
    --disable-gloo-process-groups
    --enable-gpt-oss # see options added by modelopts
)

mkdir -p "${OUTPUT_BASEPATH}/checkpoint/"
mkdir -p "${OUTPUT_BASEPATH}/tensorboard/"
mkdir -p "${OUTPUT_BASEPATH}/log/"
current_time=$(date "+%Y.%m.%d-%H.%M.%S")

OUTPUT_ARGS="
        --log-interval 1 \
        --eval-interval 1000 \
        --eval-iters 0 \
        --save-interval ${SAVE_INTERVAL:-50}
        --save "${OUTPUT_BASEPATH}/checkpoint" \
"

LOGGING_ARGS="
        --timing-log-level 2 \
        --log-throughput
        --log-timers-to-tensorboard \
        --log-validation-ppl-to-tensorboard \
        --tensorboard-queue-size 1 \
        --tensorboard-dir "${OUTPUT_BASEPATH}/tensorboard" \
"

if [ -n "${WANDB_API_KEY}" ]; then
    LOGGING_ARGS+=(
        --wandb-project ${WANDB_PROJECT:-"GptOSS-20b-bf16-SFT"}
        --wandb-exp-name ${WANDB_NAME:-"gptoss-20b-bf16"} 
    )
fi

export PYTORCH_ALLOC_CONF=expandable_segments:True
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# TODO (yiakwy) : use modelopt finetune
torchrun ${DISTRIBUTED_ARGS[@]} $MEGATRON_PATH/pretrain_gpt.py \
    $load_options \
    $GPT_OSS_SFT_MODEL_ARGS \
    ${MODEL_ARGS[@]} \
    $do_options \
    $te_options \
    ${MOE_ARGS[@]} \
    ${DATA_ARGS[@]} \
    ${TRAINING_ARGS[@]} \
    --recompute-activations \
    $pr_options \
    ${MODEL_PARALLEL_ARGS[@]} \
    ${OUTPUT_ARGS} \
    ${LOGGING_ARGS} 2>&1 | tee ${OUTPUT_BASEPATH}/log/megatron_trainning.Rank_${RANK}.log