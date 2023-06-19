#!/bin/bash
#SBATCH --job-name=starcoderplus
#SBATCH --nodes=64
#SBATCH --ntasks-per-node=1
#SBATCH --exclusive
#SBATCH --gres=gpu:8
#SBATCH --partition=production-cluster
#SBATCH --output=/fsx/leandro/logs/starcoderplus/bcs-%x-%j.out

set -x -e
source /admin/home/leandro/.bashrc

conda activate megatron

echo "START TIME: $(date)"

# File Path setup
SCRIPT_REPO=/fsx/leandro/git/Megatron-LM-BC
pushd $SCRIPT_REPO

LOG_PATH=$SCRIPT_REPO/main_log.txt

# Training setup
GPUS_PER_NODE=8
MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
MASTER_PORT=6000
NNODES=$SLURM_NNODES
NODE_RANK=$SLURM_PROCID
WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES))

# File path setup
STARCODER_PATH=/fsx/boomcode/starcoder/
CHECKPOINT_PATH=/fsx/boomcode/starcoderplus/$SLURM_JOB_ID
TOKENIZER_FILE=/fsx/boomcode/tokenizer-starcoder/tokenizer.json
WEIGHTS_TRAIN=/fsx/boomcode/datamix/train_data_paths.txt.tmp
WEIGHTS_VALID=/fsx/boomcode/datamix/valid_data_paths.txt.tmp

mkdir -p $CHECKPOINT_PATH/tensorboard

GPT_ARGS="\
        --tensor-model-parallel-size 4 \
       --pipeline-model-parallel-size 4 \
       --sequence-parallel \
       --num-layers 40 \
       --hidden-size 6144 \
       --num-attention-heads 48 \
       --attention-head-type multiquery \
       --init-method-std 0.01275 \
       --seq-length 8192 \
       --max-position-embeddings 8192 \
       --attention-dropout 0.1 \
       --hidden-dropout 0.1 \
       --micro-batch-size 1 \
       --global-batch-size 512 \
       --lr 0.0001 \
       --min-lr 0.00001 \
       --train-iters 400000 \
       --lr-decay-iters 150000 \
       --lr-decay-style cosine \
       --lr-warmup-iters 1000 \
       --weight-decay .1 \
       --adam-beta2 .95 \
       --clip-grad 1.0 \
       --bf16 \
       --use-flash-attn \
       --fim-rate 0.5 \
       --log-interval 10 \
       --save-interval 2500 \
       --eval-interval 2500 \
       --eval-iters 2 \
       --valid-num-workers 0 \
       --override-opt_param-scheduler \
       --no-load-optim \
       --no-load-rng \
       --finetune \
"

TENSORBOARD_ARGS="--tensorboard-dir ${CHECKPOINT_PATH}/tensorboard"

CMD=" \
    $SCRIPT_REPO/pretrain_gpt.py \
    $GPT_ARGS \
    --tokenizer-type TokenizerFromFile \
    --tokenizer-file $TOKENIZER_FILE \
    --save $CHECKPOINT_PATH \
    --load $STARCODER_PATH \
    --train-weighted-split-paths-path $WEIGHTS_TRAIN \
    --valid-weighted-split-paths-path $WEIGHTS_VALID \
    --structured-logs \
    --structured-logs-dir $CHECKPOINT_PATH/logs \
    $TENSORBOARD_ARGS \
    --wandb-entity-name lvwerra \
    --wandb-project-name starcoder-plus \
    "

export LAUNCHER="python -u -m torch.distributed.run \
    --nproc_per_node $GPUS_PER_NODE \
    --nnodes $NNODES \
    --rdzv_endpoint $MASTER_ADDR:$MASTER_PORT \
    --rdzv_backend c10d \
    --max_restarts 0 \
    --tee 3 \
    "

echo $CMD

# hide duplicated errors using this hack - will be properly fixed in pt-1.12
# export TORCHELASTIC_ERROR_FILE=/tmp/torch-elastic-error.json

# force crashing on nccl issues like hanging broadcast
export NCCL_ASYNC_ERROR_HANDLING=1
# export NCCL_DEBUG=INFO
# export NCCL_DEBUG_SUBSYS=COLL
# export NCCL_SOCKET_NTHREADS=1
# export NCCL_NSOCKS_PERTHREAD=1
# export CUDA_LAUNCH_BLOCKING=1

# AWS specific
export NCCL_PROTO=simple
export RDMAV_FORK_SAFE=1
export FI_EFA_FORK_SAFE=1
export FI_EFA_USE_DEVICE_RDMA=1
export FI_PROVIDER=efa
export FI_LOG_LEVEL=1
export NCCL_IB_DISABLE=1
export NCCL_SOCKET_IFNAME=ens

export CUDA_HOME=/usr/local/cuda-11.6

# srun error handling:
# --wait=60: wait 60 sec after the first task terminates before terminating all remaining tasks
# --kill-on-bad-exit=1: terminate a step if any task exits with a non-zero exit code
SRUN_ARGS=" \
    --wait=60 \
    --kill-on-bad-exit=1 \
    "

# py-spy top -s -i -n -- $LAUNCHER --node_rank $SLURM_PROCID --role $SLURMD_NODENAME: $CMD
clear; srun $SRUN_ARGS --jobid $SLURM_JOB_ID bash -c "$LAUNCHER --node_rank \$SLURM_PROCID --role \$SLURMD_NODENAME: $CMD" 2>&1 | tee $LOG_PATH

echo "END TIME: $(date)"
