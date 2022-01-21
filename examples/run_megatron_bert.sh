#!/bin/bash
GPUS_PER_NODE=1
# Change for multinode config
MASTER_ADDR=localhost
MASTER_PORT=12003
NNODES=1
NODE_RANK=0
WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES))

prefix='compare_oneflow_loss_reproduce'

DATA_BASE=/home/wang/data/t5/

DATA_PATH=$DATA_BASE/dataset/loss_compara_content_sentence
CHECKPOINT_PATH=$DATA_BASE/${prefix}_ckpt
TENSORBOARD_PATH=$DATA_BASE/${prefix}_tensorboard
VOCAB_PATH=$DATA_BASE/dataset/bert-base-chinese-vocab.txt

# PRE_CHECKPOINT_PATH=/cognitive_comp/gaoxinyu/megatron_model/bert-cn-wwm/${prefix}_ckpt_pre
# PRE_CHECKPOINT_PATH=/workspace/idea_model/small_model/compare_oneflow_loss_ckpt_pre

DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE --nnodes $NNODES --node_rank $NODE_RANK --master_addr $MASTER_ADDR --master_port $MASTER_PORT"


options=" \
       --tensor-model-parallel-size 1 \
       --pipeline-model-parallel-size 1 \
       --num-layers 5 \
       --hidden-size 384 \
       --num-attention-heads 16 \
       --micro-batch-size 16 \
       --global-batch-size 16 \
       --seq-length 512 \
       --max-position-embeddings 512 \
       --train-iters 10000 \
       --save $CHECKPOINT_PATH \
       --data-path $DATA_PATH \
       --vocab-file $VOCAB_PATH \
       --data-impl mmap \
       --split 949,50,1 \
       --distributed-backend nccl \
       --lr 0.0001 \
       --lr-decay-style linear \
       --min-lr 1.0e-5 \
       --lr-decay-iters 9000 \
       --weight-decay 1e-2 \
       --clip-grad 1.0 \
       --lr-warmup-fraction .01 \
       --log-interval 100 \
       --save-interval 20000 \
       --eval-interval 20000 \
       --eval-iters 1000 \
       --tensorboard-dir ${TENSORBOARD_PATH} \
       --vocab-extra-ids 100 \
       --num-workers 0 \
       --decoder-seq-length 512
       # --tokenizer-type BertCNWWMTokenizer"


export CUDA_VISIBLE_DEVICES=0
# SCRIPT_PATH=/cognitive_comp/gaoxinyu/Megatron/Megatron-LM/
SCRIPT_PATH=~/workspace/Megatron-LM
PY=python3
run_cmd="$PY -m torch.distributed.launch $DISTRIBUTED_ARGS  ${SCRIPT_PATH}/pretrain_t5.py $@ ${options}"



# SINGULARITY_PATH=/cognitive_comp/ganruyi/pytorch21_06_py3_docker_image_v2.sif
#singularity exec --nv -B /cognitive_comp/gaoxinyu/:/cognitive_comp/gaoxinyu/ SINGULARITYPATH{run_cmd}
eval $run_cmd
set +x
