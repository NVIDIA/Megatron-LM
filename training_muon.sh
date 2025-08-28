#!/bin/bash

# Runs the "340M" parameter model with Distributed Muon
# See more details at: https://github.com/MoonshotAI/Moonlight/blob/master/Moonlight.pdf

export CUDA_DEVICE_MAX_CONNECTIONS=1

GPUS_PER_NODE=8
# Change for multinode config
MASTER_ADDR=localhost
MASTER_PORT=6000
NUM_NODES=1
NODE_RANK=0
WORLD_SIZE=$(($GPUS_PER_NODE*$NUM_NODES))

CHECKPOINT_PATH=/lustre/fsw/portfolios/coreai/users/boxiangw/checkpoints
# TENSORBOARD_LOGS_PATH=/PATH/TO/TB
TOKENIZER_MODEL=/lustre/fsw/portfolios/coreai/projects/coreai_mlperf_training/data/c4/405b/tokenizer
# data is preprocessed as described in Megatron-LM' readme
DATA_PATH=/lustre/fsw/portfolios/coreai/projects/coreai_mlperf_training/data/c4/405b/c4-train.en_6_text_document

BLEND_PATH="/lustre/fs1/portfolios/coreai/users/skyw/scripts/megatron/621m/8B_15T_phase1_final_v2.json"

DISTRIBUTED_ARGS=(
    --nproc_per_node $GPUS_PER_NODE 
    --nnodes $NUM_NODES 
    --master_addr $MASTER_ADDR 
    --master_port $MASTER_PORT
)

GQA_ARGS=(
    --group-query-attention 
    --num-attention-heads 16 
    --num-query-groups 8 
    --kv-channels 128
)

MLP_ARGS=(
    --squared-relu 
    --num-layers 24 
    --hidden-size 1024 
    --ffn-hidden-size 4096 
    --normalization RMSNorm
)


OPT_ARGS=(
    --micro-batch-size 1 
    --global-batch-size 512 
    --lr-decay-style cosine 
    --optimizer dist_muon
    --ckpt-format torch 
    --lr-warmup-samples 2048000 

    --train-samples 24414062 
    --lr-decay-samples 23193359 
    --lr 1.0e-3 
    --min-lr 1.0e-5

    --weight-decay 0.1 
    --clip-grad 1.0 
    --init-method-std 0.02 
    --lr-decay-style cosine 
    --muon-scale-mode spectral 
    --tokenizer-type HuggingFaceTokenizer
    --tokenizer-model $TOKENIZER_MODEL
    --data-path $DATA_PATH
)

options=(
    --attention-backend flash 
    --distributed-timeout-minutes 2 
    --use-mcore-models 
    --data-cache-path /lustre/fsw/portfolios/coreai/users/boxiangw/muon-dev/megatron-lm/data_cache 
    --no-mmap-bin-files 
    --untie-embeddings-and-output-weights 
    --disable-bias-linear 
    --position-embedding-type rope 
    --rotary-base 1000000 
    --rotary-percent 1.0 
    ${MLP_ARGS[@]} 
    ${GQA_ARGS[@]} 
    --attention-dropout 0.0 
    --hidden-dropout 0.0 
    --exit-duration-in-mins 230 
    --tensor-model-parallel-size 1 
    --pipeline-model-parallel-size 1 
    --seq-length 4096 
    --max-position-embeddings 4096 
    --log-interval 10 
    --eval-iters 10 
    --eval-interval 25 
    --save-interval 50 
    --ckpt-format torch_dist 
    --log-progress  
    --timing-log-option minmax 
    --log-params-norm 
    --log-num-zeros-in-grad 
    --log-throughput 
    ${OPT_ARGS[@]} 
    --bf16 
    --no-create-attention-mask-in-dataloader 
    --manual-gc 
    --num-workers 2 
    --log-straggler 
    --disable-straggler-on-startup 
    --straggler-minmax-count 16 
    --check-weight-hash-across-dp-replicas-interval 20000 
    --tensorboard-dir /lustre/fsw/portfolios/coreai/users/boxiangw/tensorboard 
    --load ${CHECKPOINT_PATH} 
    --save ${CHECKPOINT_PATH} 
    --wandb-project "muon-test" 
    --wandb-exp-name "muon-621m-mbs1-gbs512-torch_dist" 
    --wandb-save-dir /lustre/fsw/portfolios/coreai/users/boxiangw/muon-dev/megatron-lm/wandb/muon-621m 
)

torchrun ${DISTRIBUTED_ARGS[@]} pretrain_gpt.py \
    ${options[@]} 