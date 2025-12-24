#!/bin/bash

# Simulate DeepSeek v3 

export CUDA_DEVICE_MAX_CONNECTIONS=1
export NCCL_DEBUG=WARN

GPUS_PER_NODE=8
# Change for multinode config
MASTER_ADDR=localhost
MASTER_PORT=29527
NUM_NODES=1
NODE_RANK=0
WORLD_SIZE=$(($GPUS_PER_NODE*$NUM_NODES))

export TIME_STAMP=$(date '+%Y%m%d-%H%M')

BASEDIR="/dnn_training_sys/users/lisiyuan.li/autolab/workspace/simulation"
LOG_PATH="${BASEDIR}/simulate_deepseek_v3/${TIME_STAMP}/logs"
OUTPUT_PATH="${BASEDIR}/simulate_deepseek_v3/${TIME_STAMP}/outputs"
mkdir -p $LOG_PATH
mkdir -p $OUTPUT_PATH

# Backup script to log directory
if [[ $NODE_RANK -eq 0 ]]; then
    cp -r ${0} ${LOG_PATH}
fi

CHECKPOINT_PATH=$1 #<Specify path>
TENSORBOARD_LOGS_PATH=$2 #<Specify path>
VOCAB_FILE=$3 #<Specify path to file>/gpt2-vocab.json
MERGE_FILE=$4 #<Specify path to file>/gpt2-merges.txt
DATA_PATH=$5 #<Specify path and file prefix>_text_document

DISTRIBUTED_ARGS=(
    --nproc_per_node $GPUS_PER_NODE 
    --nnodes $NUM_NODES 
    --master_addr $MASTER_ADDR 
    --master_port $MASTER_PORT
)

DEEPSEEK_MODEL=(
    --num-layers 14
    --seq-length 4096 
    --max-position-embeddings 4096 
    --max-position-embeddings 4096 
    --hidden-size 7168 
    --ffn-hidden-size 18432 
    --num-attention-heads 128 
    --kv-channels 128 

    --position-embedding-type rope 
    --rotary-base 10000 
    --make-vocab-size-divisible-by 3232 
    --attention-backend fused # Can use (flash/fused/unfused/local)
    --moe-router-load-balancing-type seq_aux_loss 
    --moe-router-topk 8 
    --moe-grouped-gemm 
    --moe-aux-loss-coeff 1e-4 
    --moe-router-group-topk 4 
    --moe-router-num-groups 8 
    --moe-router-pre-softmax 
    #--moe-router-padding-for-quantization 
    --moe-router-topk-scaling-factor 2.5 
    --moe-router-score-function sigmoid 
    --moe-router-enable-expert-bias 
    --moe-router-bias-update-rate 1e-3 
    --moe-router-dtype fp32 
    #--moe-permute-fusion 
    #--moe-router-fusion 
    --q-lora-rank 1536 
    --kv-lora-rank 512
    --qk-head-dim 128 
    --qk-pos-emb-head-dim 64 
    --v-head-dim 128 
    --disable-bias-linear

    --num-experts 256 
    --moe-layer-freq "[0]*1+[1]*13" 
    --moe-ffn-hidden-size 2048 
    --moe-shared-expert-intermediate-size 2048

    --normalization RMSNorm 
    --norm-epsilon 1e-6 
    --swiglu 
    --untie-embeddings-and-output-weights 
    --multi-latent-attention 
    --attention-dropout 0.0 
    --hidden-dropout 0.0 
)

TRAINING_ARGS=(
    --micro-batch-size 1 
    --global-batch-size 384 
    --train-iters 500000 
    --weight-decay 0.1 
    --adam-beta1 0.9 
    --adam-beta2 0.95 
    --init-method-std 0.006 
    --clip-grad 1.0 
    --bf16
    --lr 6.0e-5 
    --lr-decay-style cosine 
    --min-lr 6.0e-6
    --lr-warmup-fraction .001 
    --lr-decay-iters 430000 
)

MODEL_PARALLEL_ARGS=(
	--tensor-model-parallel-size 1
	--pipeline-model-parallel-size 4
    --expert-model-parallel-size 8
    --expert-tensor-parallel-size 1
    --pipeline-model-parallel-layout "Ett|(tt|)*6L"
    --moe-token-dispatcher-type flex
    --moe-flex-dispatcher-backend deepep
    --sequence-parallel
)

DATA_ARGS=(
    --mock-data
    --tokenizer-type NullTokenizer
    --vocab-size 128256
)

COMPUTE_ARGS=(
    --use-distributed-optimizer 
    --overlap-grad-reduce 
    --overlap-param-gather 
    --use-mcore-models

    --no-save-optim 
    --no-check-for-nan-in-loss-and-grad 
    --cross-entropy-loss-fusion 
    --cross-entropy-fusion-impl te 
    --manual-gc 
    --manual-gc-interval 10 
    --enable-experimental 
    --transformer-impl transformer_engine 

    # --fp8-recipe mxfp8 
    # --fp8-format e4m3 
    # --fp8-param-gather 
    # --reuse-grad-buf-for-mxfp8-param-ag
)

SIMULATE_ARGS=(
    --simulate-global-step
    --execute-mode "router_balanced"
    --simulate-result-dir ${OUTPUT_PATH}
)

# SIMULATE_ARGS=(
#     --simulate-global-step
#     --skip-execute
#     --load-result-dir /dnn_training_sys/users/lisiyuan.li/autolab/workspace/simulation/simulate_deepseek_v3/20251213-2025/outputs
# )

CMD="torchrun ${DISTRIBUTED_ARGS[@]} pretrain_gpt.py \
    ${DEEPSEEK_MODEL[@]} \
    ${TRAINING_ARGS[@]} \
    ${COMPUTE_ARGS[@]} \
    ${MODEL_PARALLEL_ARGS[@]} \
    ${DATA_ARGS[@]} \
    ${SIMULATE_ARGS[@]}"

$CMD 2>&1 | tee ${LOG_PATH}/log_${NODE_RANK}.log
