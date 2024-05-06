#! /bin/bash
if [ -z "$MLP_WORKER_NUM" ]; then
    GPUS_PER_NODE=8
    WORLD_SIZE=8
    NNODES=1
    NODE_RANK=0
else
    GPUS_PER_NODE=$MLP_GPU
    WORLD_SIZE=$(($MLP_WORKER_NUM * $MLP_GPU))
    NNODES=$MLP_WORKER_NUM
    NODE_RANK=$MLP_ROLE_INDEX
fi

# MASTER_ADDR is the first in SLURM_NODELIST
if [ -z "$MLP_WORKER_0_HOST" ]; then
    MASTER_ADDR=localhost
    MASTER_PORT=27878
else
    MASTER_ADDR=$MLP_WORKER_0_HOST
    MASTER_PORT=$MLP_WORKER_0_PORT
fi


NCCL_ALGO=RING
NCCL_IB_GID_INDEX=3
NCCL_IB_RETRY_CNT=7
NCCL_IB_TIME_OUT=32
NCCL_DEBUG=INFO
GLOO_SOCKET_IFNAME=eth1
NCCL_SOCKET_IFNAME=eth1
CUDA_DEVICE_MAX_CONNECTIONS=1
OPTIONS_NCCL="CUDA_DEVICE_MAX_CONNECTIONS=1 NCCL_ALGO=RING NCCL_IB_GID_INDEX=3 NCCL_IB_RETRY_CNT=7 NCCL_IB_TIME_OUT=32 NCCL_DEBUG=INFO GLOO_SOCKET_IFNAME=eth1 NCCL_SOCKET_IFNAME=eth1"

CHECKPOINT=./checkpoints/llama2_megatron_training_first36

export CUDA_DEVICE_MAX_CONNECTIONS=1

OPTIONS_NCCL="CUDA_DEVICE_MAX_CONNECTIONS=1 NCCL_ALGO=RING NCCL_IB_GID_INDEX=3 NCCL_IB_RETRY_CNT=7 NCCL_IB_TIME_OUT=32 NCCL_DEBUG=INFO GLOO_SOCKET_IFNAME=eth1 NCCL_SOCKET_IFNAME=eth1"

DISTRIBUTED_ARGS="
    --nproc_per_node $GPUS_PER_NODE \
    --nnodes $NNODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT
"

run_cmd="${OPTIONS_NCCL} torchrun $DISTRIBUTED_ARGS run_text_generation_server_mm.py   \
       --tensor-model-parallel-size 8  \
       --pipeline-model-parallel-size 2 \
       --first-pipeline-num-layers 36 \
       --load $CHECKPOINT \
       --vit-load /path/to/save/eva2-clip-224-mcore-tp8-pp1 \
       --num-layers 80  \
       --hidden-size 8192  \
       --num-attention-heads 64  \
       --group-query-attention \
       --num-query-groups 8 \
       --ffn-hidden-size 28672 \
       --max-position-embeddings 4096  \
       --vocab-size 32000 \
       --position-embedding-type rope \
       --no-position-embedding \
       --transformer-impl transformer_engine \
       --use-mcore-models \
       --disable-bias-linear \
       --tokenizer-type Llama2Tokenizer  \
       --bf16  \
       --micro-batch-size 1  \
       --seq-length 4096  \
       --image-seq-length 256 \
       --mock-data \
       --tokenizer-model /path/to/Llama-2-70b-chat-hf/tokenizer.model \
       --use-checkpoint-args \
       --no-load-optim \
       --no-load-rng \
       --swiglu \
       --normalization RMSNorm \
       --untie-embeddings-and-output-weights \
       --seed 42"
echo ${run_cmd}
eval ${run_cmd}