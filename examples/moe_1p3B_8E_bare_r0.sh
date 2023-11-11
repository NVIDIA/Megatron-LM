#!/bin/bash

# Runs the "345M" parameter model

export CUDA_DEVICE_MAX_CONNECTIONS=1

export WANDB_API_KEY=6f0443d34d41df289b878635a247d89381c06271
#export NCCL_DEBUG=INFO

GPUS_PER_NODE=8
# Change for multinode config
export MASTER_ADDR="64.156.70.219"
export MASTER_PORT=6000
NNODES=2
NODE_RANK=1
WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES))

CHECKPOINT_PATH=/media/16TBNVME/qanthony/ckpts
VOCAB_FILE=/mnt/shared/datasets/SlimPajama-627B_megatron/gpt-neox-20b-tokenizer/vocab.json
MERGE_FILE=/mnt/shared/datasets/SlimPajama-627B_megatron/gpt-neox-20b-tokenizer/merges.txt
DATA_PATH=/mnt/shared/datasets/SlimPajama-627B_megatron/gpt-neox-20b-tokenizer/train_text_document

WANDB_PROJECT=moe
WANDB_EXP_NAME=moe_1p3b_16e_slimpj_test
WANDB_SAVE_DIR=/media/16TBNVME/qanthony/wandb

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
    --micro-batch-size 8 \
    --global-batch-size 1024 \
    --lr 0.00025 \
    --train-iters 570000 \
    --lr-decay-iters 570000 \
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
    --swiglu\
    --profile \
    --profile-step-start 1 \
    --profile-step-end 2 \
    --profile-ranks 0 1 2 3 4 5 6 7 8
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
    --save-interval 20000 \
    --eval-interval 500 \
    --eval-iters 50 \
    --wandb-project $WANDB_PROJECT \
    --wandb-exp-name $WANDB_EXP_NAME \
    --wandb-save-dir $WANDB_SAVE_DIR
"

torchrun $DISTRIBUTED_ARGS /home/cirrascale/beren/torchrun_env/Megatron-LM/pretrain_gpt.py \
    $GPT_ARGS \
    $DATA_ARGS \
    $OUTPUT_ARGS \
    --distributed-backend nccl \
    --save $CHECKPOINT_PATH \
    --load $CHECKPOINT_PATH


#mpirun -np 8 --mca btl smcuda,^lo,^openib --mca coll smcuda -mca btl_base_verbose 10 --mca mtl_base_verbose 99 --mca btl_tcp_if_include eth0 -x CONDA_PREFIX -x LIBRARY_PATH -x CUDNN_LIBRARY -x CUDNN_INCLUDE_DIR -x CONDA_DEFAULT_ENV -x WANDB_API_KEY -x CUDA_DEVICE_MAX_CONNECTIONS -x PATH -x LD_LIBRARY_PATH -x CPATH --hostfile hosts nsys profile -t nvtx,cuda -o nsys_mpi/nsys_output_mpi-%q{OMPI_COMM_WORLD_RANK} --force-overwrite true --capture-range=cudaProfilerApi /mnt/shared/qanthony/miniconda3/envs/deepspeed/bin/python /mnt/shared/qanthony/Megatron-LM/pretrain_gpt.py \
