#! /bin/bash
echo "------ARGUMENTS LIST --------"
for ARGUMENT in "$@"
do
   KEY=$(echo $ARGUMENT | cut -f1 -d=)

   KEY_LENGTH=${#KEY}
   VALUE="${ARGUMENT:$KEY_LENGTH+1}"

   export "$KEY"="$VALUE"
   echo "$KEY=$VALUE"
done
echo "---------------------------------"

set -x 

# Change for multinode config
GPUS_PER_NODE=8
MASTER_ADDR=localhost
MASTER_PORT=6000
NODE_RANK=0
WORLD_SIZE=$(($GPUS_PER_NODE*$NUM_NODES))
command="export CUDA_DEVICE_MAX_CONNECTIONS=1;"


# Runs the "345M" parameter model
DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE --nnodes $NUM_NODES"

torch_run_cmd="torchrun $DISTRIBUTED_ARGS \
       pretrain_bert.py \
       --num-layers 24 \
       --hidden-size 1024 \
       --num-attention-heads 16 \
       --log-params-norm \
       --log-num-zeros-in-grad \
       --log-validation-ppl-to-tensorboard \
       --log-timers-to-tensorboard \
       --tensorboard-dir ${TENSORBOARD_DIR} \
       --micro-batch-size 4 \
       --global-batch-size 128 \
       --seq-length 512 \
       --max-position-embeddings 512 \
       --train-iters $MAX_STEPS \
       --timing-log-level 2 \
       --lr-decay-iters 990000 \
       --save $CHECKPOINT_PATH \
       --load $CHECKPOINT_PATH \
       --data-path $DATA_PATH \
       --vocab-file /workspace/data/bert_data/vocab.txt \
       --split 949,50,1 \
       --distributed-backend nccl \
       --lr 0.0001 \
       --min-lr 0.00001 \
       --lr-warmup-fraction 0.01 \
       --log-interval 1 \
       --save-interval 10000 \
       --eval-interval 1000 \
       --eval-iters 10 \
       --tensor-model-parallel-size $TP_SIZE \
       --pipeline-model-parallel-size $PP_SIZE \
       ${VP_SIZE:+--num-layers-per-virtual-pipeline-stage "$VP_SIZE"} \
       --no-gradient-accumulation-fusion \
       --fp16 "

command="$command $torch_run_cmd"
echo "-------------------- THE FINAL PRETRAIN SCRIPT COMMAND THAT WILL BE RUN ------------"
echo "$command"
echo "-----------------------------------------------------------------------------"

echo "$command" > $SCRIPTS_DIR/pretrain_bert_distributed_command.sh
eval $command