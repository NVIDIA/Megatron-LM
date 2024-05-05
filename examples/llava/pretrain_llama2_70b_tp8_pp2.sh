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

CHECKPOINT_PATH=/path/to/save/llama2-70b-chat-megatron-mcore-tp8-pp2-first36

DISTRIBUTED_ARGS="
    --nproc_per_node $GPUS_PER_NODE \
    --nnodes $NNODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT
"

GPT_ARGS="
    --tensor-model-parallel-size 8 \
    --pipeline-model-parallel-size 2 \
    --first-pipeline-num-layers 36 \
    --num-layers 80 \
    --hidden-size 8192 \
    --num-attention-heads 64 \
    --seq-length 1024 \
    --image-seq-length 257 \
    --max-position-embeddings 1024 \
    --micro-batch-size 2 \
    --global-batch-size 64 \
    --swiglu \
    --transformer-impl transformer_engine \
    --normalization RMSNorm \
    --untie-embeddings-and-output-weights \
    --position-embedding-type rope \
    --no-position-embedding \
    --disable-bias-linear \
    --group-query-attention \
    --num-query-groups 8 \
    --vocab-size 32000 \
    --tokenizer-type Llama2Tokenizer \
    --tokenizer-model /path/to/Llama-2-70b-chat-hf/tokenizer.model \
    --ffn-hidden-size 28672 \
    --no-load-optim \
    --no-load-rng \
    --use-checkpoint-args \
    --use-distributed-optimizer \
    --no-save-optim \
    --lr 0.00001 \
    --train-iters 5000 \
    --lr-decay-iters 320000 \
    --lr-decay-style cosine \
    --min-lr 1.0e-5 \
    --weight-decay 1e-2 \
    --lr-warmup-fraction .01 \
    --clip-grad 1.0 \
    --bf16 \
    --use-mcore-models
"

DATA_ARGS="
    --train-data-path /path/to/llava_instruct_150k_datas/metadata.json \
"

OUTPUT_ARGS="
    --log-interval 50 \
    --save-interval 500 \
    --eval-interval 500 \
    --eval-iters 10
"

LOG_DIR="./output"
mkdir -p ${LOG_DIR}/${MLP_TASK_ID}_${JOB_NAME}
run_cmd="${OPTIONS_NCCL} torchrun $DISTRIBUTED_ARGS pretrain_llava.py \
    $GPT_ARGS \
    $DATA_ARGS \
    $OUTPUT_ARGS \
    --distributed-backend nccl \
    --save ./checkpoints/llama2_megatron_training_first36/ \
    --load $CHECKPOINT_PATH \
    --vit-load /path/to/save/eva2-clip-224-mcore-tp8-pp1"

echo ${run_cmd}
eval ${run_cmd} >${LOG_DIR}/${MLP_TASK_ID}_${JOB_NAME}/${MLP_ROLE_INDEX}.out 2>${LOG_DIR}/${MLP_TASK_ID}_${JOB_NAME}/${MLP_ROLE_INDEX}.err

set +x
echo "DONE with job $MLP_TASK_ID, index $MLP_ROLE_INDEX on `hostname`"
