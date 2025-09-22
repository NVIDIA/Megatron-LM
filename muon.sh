#!/bin/bash

# To use this script, please
# 1. Add your wandb api key and entity to the script (optional)
# 2. Add your tokenizer model path and data path to the script
# 3. Select your optimizer from [muon, dist_muon, adam], defaults to dist_muon


# TODO: Paths and parameters to be added
export WANDB_API_KEY=<your_wandb_api_key> # optional
export WANDB_ENTITY=<your_wandb_entity> # optional
optimizer=dist_muon # choose from [muon, dist_muon, adam]
TOKENIZER_MODEL="your_path"
DATA_PATH="your_path"

DATETIME=`date +'date_%y-%m-%d_time_%H-%M-%S'`

DIR=$PWD
EXP_NAME="bf16_100B_${optimizer}_lr_1.0e-3_N${SLURM_NNODES}"
RUN_DIR="${DIR}/experiments/${EXP_NAME}"
LOG_DIR="${RUN_DIR}/logs"
CHECKPOINT_DIR="${RUN_DIR}/checkpoints"
TENSORBOARD_DIR="${RUN_DIR}/tensorboard"
DATACACHE_DIR="${DIR}/data-cache-phase-1/"
MEGATRON_PATH="${DIR}/Megatron-LM"
EMERGING_OPTIMIZERS_PATH="${DIR}/Emerging-Optimizers"
export TRITON_CACHE_DIR=${RUN_DIR}/triton_cache
export TORCHINDUCTOR_CACHE_DIR=${RUN_DIR}/inductor_cache

mkdir -p ${LOG_DIR}
mkdir -p ${CHECKPOINT_DIR}
mkdir -p ${DATACACHE_DIR}
mkdir -p ${TENSORBOARD_DIR}
mkdir -p ${TRITON_CACHE_DIR}
mkdir -p ${TORCHINDUCTOR_CACHE_DIR}

echo ${LOG_DIR}
echo ${TENSORBOARD_DIR}

# Optimizer related parameters
OPTIMIZER_ARGS=(
    --optimizer $optimizer
    --muon-momentum 0.9
    --muon-scale-mode spectral
    --muon-fp32-matmul-prec medium
    --muon-num-ns-steps 5
)

# MoE related parameters
MOE_ARGS=(
    --num-experts 8
    --moe-router-topk 2
    --moe-router-load-balancing-type aux_loss
    --moe-aux-loss-coeff 1e-2
    --moe-grouped-gemm
    --moe-token-dispatcher-type alltoall
    # --overlap-param-gather # not supported in Muon yet
    --overlap-grad-reduce
)

# Samples and learning rates for different token horizons
options=(
    --global-batch-size 512
    --micro-batch-size 1
    --seq-length 1024
    --max-position-embeddings 1024
    --tensor-model-parallel-size 1
    --context-parallel-size 1
    --pipeline-model-parallel-size 1
    --expert-model-parallel-size 8

    --num-layers 30
    --hidden-size 4096
    --ffn-hidden-size 14336
    --num-attention-heads 32
    --group-query-attention
    --num-query-groups 8
    --swiglu
    --norm-epsilon 1.0e-05
    --normalization RMSNorm
    --disable-bias-linear
    --attention-dropout 0.0
    --hidden-dropout 0.0
    --position-embedding-type rope
    --rotary-base 1000000
    --rotary-percent 1.0
    --untie-embeddings-and-output-weights
    --use-mcore-models
    --attention-backend fused

    --init-method-std 0.0134
    --lr 0.00015
    --min-lr 0.00001
    --clip-grad 1.0
    --adam-beta1 0.9
    --adam-beta2 0.95
    --weight-decay 0.1

    --distributed-timeout-minutes 60
    --ckpt-format torch
    --data-cache-path ${DATACACHE_DIR}
    --exit-duration-in-mins 235
    --train-samples 1953125
    --lr-decay-samples 1949218
    --lr-warmup-samples 3907
    --lr-decay-style cosine
    --split 99,1,0
    --no-create-attention-mask-in-dataloader
    --no-mmap-bin-files
    --manual-gc

    --log-interval 10
    --eval-iters 32
    --eval-interval 100
    --log-throughput

    --bf16
    --num-workers 1

    --tokenizer-type HuggingFaceTokenizer
    --tokenizer-model $TOKENIZER_MODEL
    --data-path $DATA_PATH

    --wandb-project "muon-test"
    --wandb-exp-name "muon-test"
    --wandb-save-dir ${RUN_DIR}/wandb/muon-test
    --log-timers-to-tensorboard
    --log-memory-to-tensorboard
    --log-validation-ppl-to-tensorboard
    --tensorboard-dir ${RUN_DIR}/tensorboard

    ${OPTIMIZER_ARGS[@]}

    ${MOE_ARGS[@]}
)


export PYTHONPATH=${MEGATRON_PATH}:${EMERGING_OPTIMIZERS_PATH}:${PYTHONPATH}
echo $PYTHONPATH
torchrun --nproc_per_node=8 ${MEGATRON_PATH}/pretrain_gpt.py ${options[@]}