#!/bin/bash

# ========== Configuration ==========
NUM_NODES=1
NODE_RANK=0
NUM_GPUS_PER_NODE=1
MASTER_ADDR="localhost"
MASTER_PORT=29501

# ========== Recommended Exports ==========
export CUDA_DEVICE_ORDER="PCI_BUS_ID"
export NCCL_PROTO="Simple,LL128"
export NCCL_DEBUG="INFO"
export NCCL_SOCKET_IFNAME="eth0"
export NCCL_NET_PLUGIN=none
export PYTHONUNBUFFERED="1"
export CUDA_LAUNCH_BLOCKING="1"
export CUDA_DEVICE_MAX_CONNECTIONS="1"

# ========== Print setup ==========
echo "Launching with:"
echo " - MASTER_ADDR: $MASTER_ADDR"
echo " - MASTER_PORT: $MASTER_PORT"
echo " - NODE_RANK: $NODE_RANK / $NUM_NODES"
echo " - GPUs per node: $NUM_GPUS_PER_NODE"

# Parallelism configuration
TP=1  # Tensor Parallel
PP=1  # Pipeline Parallel  
CP=1  # Context Parallel

# Build experiment name with parallelism config
EXP_NAME="test"

options="\
  --micro-batch-size 8 \
  --global-batch-size 512 \
  --rampup-batch-size 64 64 4882 \
  --train-samples 210449 \
  --data-path /home/aiccu/Megatron-LM/data/merged_falcon_english_32k/merged_0 \
  --data-cache-path /gcs/data/data-cache-path \
  --tokenizer-type HuggingFaceTokenizer \
  --tokenizer-model tiiuae/Falcon-H1-0.5B-Instruct \
  --vocab-size 32784 \
  --make-vocab-size-divisible-by 1 \
  --tensorboard-dir /gcs/data/tok-dir \
  --log-validation-ppl-to-tensorboard \
  --log-timers-to-tensorboard \
  --log-throughput \
  --log-interval 10 \
  --no-mmap-bin-files \
  --split 1000,0,0 \
  --fp32-residual-connection \
  \
  --disable-bias-linear \
  --num-layers 72 \
  --hidden-size 1024 \
  --ffn-hidden-size 2048 \
  --num-attention-heads 8 \
  --group-query-attention \
  --num-query-groups 2 \
  --seq-length 2048 \
  --max-position-embeddings 2048 \
  --rotary-base 100000000000 \
  --position-embedding-type rope \
  --no-rope-fusion \
  --disable-bias-linear \
  \
  --spec megatron.core.models.mamba.mamba_layer_specs mamba_stack_spec \
  --mamba-state-dim 128 \
  --mamba-head-dim 64 \
  --mamba-num-groups ${TP} \
  --reset-position-ids \
  \
  --weight-decay 0.1 \
  --optimizer adam \
  --adam-beta1 0.9 \
  --adam-beta2 0.95 \
  --adam-eps 1e-16 \
  --use-distributed-optimizer \
  --clip-grad 1.0 \
  --bf16 \
  --init-method-std 0.02 \
  --lr 128e-5 \
  --lr-decay-style WSD \
  --lr-wsd-decay-samples 15137 \
  --lr-wsd-decay-style exponential \
  --min-lr 0.0 \
  --lr-warmup-init 0.0 \
  --lr-warmup-fraction 0.1 \
  --ckpt-format torch \
  \
  --tensor-model-parallel-size ${TP} \
  --pipeline-model-parallel-size ${PP} \
  --context-parallel-size ${CP} \
  --overlap-param-gather \
  --overlap-grad-reduce \
  --no-gradient-accumulation-fusion \
  --no-masked-softmax-fusion \
  \
  --attention-softmax-in-fp32 \
  --untie-embeddings-and-output-weights \
  --swiglu \
  --normalization RMSNorm \
  --norm-epsilon 1e-5 \
  --attention-dropout 0.0 \
  --hidden-dropout 0.0 \
  --use-flash-attn \
  \
  --distributed-timeout-minutes 90 \
  --num-workers 16 \
  --num-dataset-builder-threads 32 \
  \
  --no-create-attention-mask-in-dataloader \
  --mid-level-dataset-surplus 0.005 \
  \
  --parallel-hybrid-ratio 0.5 \
  --hybrid-attention-ratio 0.0 \
  --hybrid-mlp-ratio 0.5 \
  \
  --save /gcs/data/save \
  --save-interval 420 \
  --wandb-project mlm-final-pr \
  --wandb-exp-name final-pr \
  \
  --disable-msc
  --dataloader-type single \
  --eval-iters 0 \
  --no-load-optim \
  --no-load-rng \
  --seed 52 \
  --override-opt_param-scheduler"

# ========== Run ==========
source ~/miniconda3/etc/profile.d/conda.sh
conda activate megatron
export CUDA_VISIBLE_DEVICES=0

$(which torchrun) \
    --nproc_per_node=$NUM_GPUS_PER_NODE \
    --nnodes=$NUM_NODES \
    --node_rank=$NODE_RANK \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    pretrain_mamba.py ${options}