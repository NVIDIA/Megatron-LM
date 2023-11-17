#!/bin/bash

declare -A name2ip
name2ip[GPU6268]=172.18.135.11
name2ip[GPU6292]=172.18.135.12
name2ip[GPU626E]=172.18.135.13
name2ip[GPU6284]=172.18.135.14
name2ip[GPUC194]=172.18.135.15
name2ip[GPU6278]=172.18.135.16
name2ip[GPU627A]=172.18.135.17
name2ip[GPU6282]=172.18.135.18

declare -A rank
rank[GPU6268]=1
rank[GPU6292]=2
rank[GPU626E]=3
rank[GPU6284]=4
rank[GPUC194]=5
rank[GPU6278]=6
rank[GPU627A]=0
rank[GPU6282]=7

current_node=$(hostname)
current_rank=${rank[$current_node]}

master_node=NOT_FOUND
for k in "${!rank[@]}"; do
    if [ ${rank[$k]} = 0 ]; then
        master_node="$k"
    fi
done
if [ $master_node = NOT_FOUND ]; then
    echo "Must specify rank 0 node!"
    exit 1
fi
echo "Current node: $current_node. Current rank: $current_rank. Master node: $master_node. Master address: ${name2ip[$master_node]}"

# Runs the "345M" parameter model

export CUDA_DEVICE_MAX_CONNECTIONS=1
export TP_SOCKET_IFNAME=ibs108
export GLOO_SOCKET_IFNAME=ibs108

export WANDB_API_KEY=6f0443d34d41df289b878635a247d89381c06271
#export NCCL_DEBUG=INFO

GPUS_PER_NODE=8
# Change for multinode config
export MASTER_ADDR=${name2ip[$master_node]}
export MASTER_PORT=6000
NNODES=8
NODE_RANK=$current_rank
WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES))

CHECKPOINT_PATH=/checkpoints/megarun/ckpts_1p3b
VOCAB_FILE=/datasets/SlimPajama-627B_megatron/gpt-neox-20b-tokenizer/vocab.json
MERGE_FILE=/datasets/SlimPajama-627B_megatron/gpt-neox-20b-tokenizer/merges.txt
DATA_PATH=/datasets/SlimPajama-627B_megatron/gpt-neox-20b-tokenizer/train_text_document

WANDB_PROJECT=moe
WANDB_EXP_NAME=final_moe_1p3b_8e_600B_slimpj_120k
WANDB_SAVE_DIR=/checkpoints/megarun/wandb


DISTRIBUTED_ARGS="
    --nproc_per_node $GPUS_PER_NODE \
    --nnodes $NNODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT
"

GPT_ARGS="
    --num-layers 24 \
    --hidden-size 2048 \
    --num-attention-heads 16 \
    --seq-length 2048 \
    --max-position-embeddings 2048 \
    --micro-batch-size 6 \
    --global-batch-size 1152 \
    --lr 0.0001 \
    --override-opt_param-scheduler \
    --train-iters 290000 \
    --lr-decay-iters 290000 \
    --lr-decay-style cosine \
    --min-lr 1.0e-5 \
    --weight-decay 0.0 \
    --lr-warmup-fraction .01 \
    --clip-grad 1.0 \
    --fp16 \
    --num-experts 8 \
    --expert-model-parallel-size 8 \
    --recompute-granularity selective \
    --use-flash-attn \
    --accumulate-allreduce-grads-in-fp32 \
    --attention-dropout 0.0 \
    --hidden-dropout 0.0 \
    --swiglu \
    --position-embedding-type rope \
    --use-rotary-position-embeddings \
    --adam-beta2 0.95
    "
    #--fp8-format hybrid \
    #--transformer-impl transformer_engine 
#"

DATA_ARGS="
    --data-path $DATA_PATH \
    --vocab-file $VOCAB_FILE \
    --merge-file $MERGE_FILE \
    --split 949,50,1 \
    --num-workers 0 \
    --distributed-timeout-minutes 120
"

OUTPUT_ARGS="
    --log-interval 1 \
    --save-interval 5000 \
    --eval-interval 500 \
    --eval-iters 50 \
    --wandb-project $WANDB_PROJECT \
    --wandb-exp-name $WANDB_EXP_NAME \
    --wandb-save-dir $WANDB_SAVE_DIR
"

torchrun $DISTRIBUTED_ARGS /opt/Megatron-LM/pretrain_gpt.py \
    $GPT_ARGS \
    $DATA_ARGS \
    $OUTPUT_ARGS \
    --distributed-backend nccl \
    --save $CHECKPOINT_PATH \
    --load $CHECKPOINT_PATH
    
