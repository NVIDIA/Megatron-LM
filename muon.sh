#!/bin/bash

#SBATCH -p batch
#SBATCH --account=your_account
#SBATCH --exclusive
#SBATCH -t 4:00:00
#SBATCH --mem=0
#SBATCH --ntasks-per-node=8
#SBATCH --dependency=singleton
#SBATCH --job-name=your_job_name


# 621M baseline size
# 0: Number of parameters in transformer layers in billions:  0.35
# 0: Number of parameters in embedding layers in billions: 0.27
# 0: Total number of parameters in billions: 0.62
# 0: Number of parameters in most loaded shard in billions: 0.6209
# 0: Theoretical memory footprints: weight and optimizer=3774.61 MB

# export NCCL_DEBUG=INFO
# export NCCL_DEBUG_SUBSYS=COLL
# export NCCL_DEBUG_FILE="/results/nccl-debug-${RANK}"
export WANDB_API_KEY=<YOUR_WANDB_API_KEY>
export WANDB_ENTITY=<YOUR_WANDB_ENTITY>

export NCCL_IB_SL=1
export NCCL_IB_TIMEOUT=19
export NCCL_IB_QPS_PER_CONNECTION=4
export UB_TIMEOUT=720
export CUDA_DEVICE_MAX_CONNECTIONS=1
export NVTE_FWD_LAYERNORM_SM_MARGIN=16
export NVTE_BWD_LAYERNORM_SM_MARGIN=16
# with NVTE_FUSED_ATTN=0, have to use --attention-backend flash
export NVTE_FUSED_ATTN=1  
export NCCL_P2P_NET_CHUNKSIZE=2097152

export NCCL_SHM_DISABLE=1
export NCCL_PROTO=simple
export NCCL_NVLS_ENABLE=0

DATETIME=`date +'date_%y-%m-%d_time_%H-%M-%S'`

DIR="your_path_to_workspace"

optimizer=dist_muon

EXP_NAME="621M_bf16_100B_${optimizer}_tp8_split_qkv_bs_1536_lr_1.0e-3_N${SLURM_NNODES}"
RUN_DIR="${DIR}/experiments/${EXP_NAME}"
LOG_DIR="${RUN_DIR}/logs"
CHECKPOINT_DIR="${RUN_DIR}/checkpoints"
TENSORBOARD_DIR="${RUN_DIR}/tensorboard"
DATACACHE_DIR="${DIR}/data-cache-phase-1/"
MEGATRON_PATH="your_path_to_megatron_lm"
EMERGING_OPTIMIZERS_PATH="your_path_to_emerging_optimizers"
export TRITON_CACHE_DIR=${RUN_DIR}/triton_cache
#export TRITON_CACHE_MANAGER=megatron.core.ssm.triton_cache_manager:ParallelFileCacheManager
export TORCHINDUCTOR_CACHE_DIR=${RUN_DIR}/inductor_cache

mkdir -p ${LOG_DIR}
mkdir -p ${CHECKPOINT_DIR}
mkdir -p ${DATACACHE_DIR}
mkdir -p ${TENSORBOARD_DIR}
mkdir -p ${TRITON_CACHE_DIR}
mkdir -p ${TORCHINDUCTOR_CACHE_DIR}

echo ${LOG_DIR}
echo ${TENSORBOARD_DIR}

TOKENIZER_MODEL="your_path"
DATA_PATH="your_path"

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
    --seq-length 2048
    --max-position-embeddings 2048
    --tensor-model-parallel-size 1
    --context-parallel-size 1
    --pipeline-model-parallel-size 1
    --expert-model-parallel-size 8

    --num-layers 10
    --hidden-size 2048
    --ffn-hidden-size 8192
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
    --data-cache-path ${MEGATRON_PATH}/data_cache
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
    --wandb-save-dir ${MEGATRON_PATH}/wandb/muon-test
    --log-timers-to-tensorboard
    --log-memory-to-tensorboard
    --log-validation-ppl-to-tensorboard
    --tensorboard-dir ${MEGATRON_PATH}/tensorboard

    --optimizer $optimizer
    ${MOE_ARGS[@]}
)


export PYTHONPATH=${MEGATRON_PATH}:${EMERGING_OPTIMIZERS_PATH}
echo $PYTHONPATH
torchrun --nproc_per_node=8 ${MEGATRON_PATH}/pretrain_gpt.py ${options[@]}