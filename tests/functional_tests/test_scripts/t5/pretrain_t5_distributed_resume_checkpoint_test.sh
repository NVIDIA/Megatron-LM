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
if [[ -n $MBS ]]; then MBS=4; fi
if [[ -n $GBS ]]; then GBS=32; fi

GPUS_PER_NODE=8
# Change for multinode config
MASTER_ADDR=localhost
MASTER_PORT=6000
NODE_RANK=0
WORLD_SIZE=$(($GPUS_PER_NODE*$NUM_NODES))

command="export CUDA_DEVICE_MAX_CONNECTIONS=1;"

TRANSFORMER_IMPL=local
TRAINING_DTYPE=fp16

if [[ $USE_CORE -eq 1 ]]; then
       echo "Running using megatron core"
       TRANSFORMER_IMPL=local
       TRAINING_DTYPE=bf16
       command="$command export NVTE_ALLOW_NONDETERMINISTIC_ALGO=0;"
       USE_MCORE=1
       export NVTE_ALLOW_NONDETERMINISTIC_ALGO=0
fi

if [[ $USE_TE -eq 1 ]]; then
       echo "Running with TransformerEngine ..."
       TRANSFORMER_IMPL=transformer_engine
       TRAINING_DTYPE=bf16
else
       echo "Running with local transformer implementation ..."
fi
set +x
# Runs the "220M" parameter model
DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE --nnodes $NUM_NODES"

# Run for 1000 iterations and save checkpoint at 500
torch_run_cmd="torchrun $DISTRIBUTED_ARGS \
    pretrain_t5_core.py \
    --num-layers 12 \
    --hidden-size 768 \
    --num-attention-heads 12 \
    --kv-channels 64 \
    --ffn-hidden-size 3072 \
    --encoder-seq-length 512 \
    --decoder-seq-length 128 \
    --max-position-embeddings 512 \
    --tensor-model-parallel-size $TP_SIZE \
    --pipeline-model-parallel-size $PP_SIZE \
    --micro-batch-size ${MBS:-4} \
    --global-batch-size ${GBS:-32} \
    --lr 0.0001 \
    --train-iters 501 \
    --lr-decay-iters $MAX_STEPS \
    --lr-decay-style linear \
    --min-lr 0.00001 \
    --weight-decay 1e-2 \
    --lr-warmup-fraction .01 \
    --clip-grad 1.0 \
    --${TRAINING_DTYPE} \
    --vocab-extra-ids 100 \
    --init-method-std 0.015 \
    --transformer-impl $TRANSFORMER_IMPL \
    --data-path $DATA_PATH \
    --vocab-file /workspace/data/bert-large-cased-vocab.txt \
    --tokenizer-type BertWordPieceCase \
    --split 99982,9,9 \
    --save $CHECKPOINT_PATH \
    --load $CHECKPOINT_PATH \
    --log-interval 100 \
    --tensorboard-dir ${TENSORBOARD_DIR} \
    --save-interval 500 \
    --eval-interval 1000 \
    --eval-iters 10 \
    --distributed-backend nccl"

echo 500 > $CHECKPOINT_PATH/latest_checkpointed_iteration.txt

# Resume from 50th iteration ckpt and continue to 100 iterations
torch_run_cmd="torchrun $DISTRIBUTED_ARGS \
    pretrain_t5_core.py \
    --num-layers 12 \
    --hidden-size 768 \
    --num-attention-heads 12 \
    --kv-channels 64 \
    --ffn-hidden-size 3072 \
    --encoder-seq-length 512 \
    --decoder-seq-length 128 \
    --max-position-embeddings 512 \
    --tensor-model-parallel-size $TP_SIZE \
    --pipeline-model-parallel-size $PP_SIZE \
    --micro-batch-size ${MBS:-4} \
    --global-batch-size ${GBS:-32} \
    --lr 0.0001 \
    --train-iters 1001 \
    --lr-decay-iters $MAX_STEPS \
    --lr-decay-style linear \
    --min-lr 0.00001 \
    --weight-decay 1e-2 \
    --lr-warmup-fraction .01 \
    --clip-grad 1.0 \
    --${TRAINING_DTYPE} \
    --vocab-extra-ids 100 \
    --init-method-std 0.015 \
    --transformer-impl $TRANSFORMER_IMPL \
    --data-path $DATA_PATH \
    --vocab-file /workspace/data/bert-large-cased-vocab.txt \
    --tokenizer-type BertWordPieceCase \
    --split 99982,9,9 \
    --save $CHECKPOINT_PATH \
    --load $CHECKPOINT_PATH \
    --log-interval 100 \
    --tensorboard-dir ${TENSORBOARD_DIR} \
    --save-interval 500 \
    --eval-interval 1000 \
    --eval-iters 10 \
    --distributed-backend nccl"

command="$command $torch_run_cmd"
echo "-------------------- THE FINAL PRETRAIN SCRIPT COMMAND THAT WILL BE RUN ------------"
echo "$command"
echo "-----------------------------------------------------------------------------"

echo "$command" > $SCRIPTS_DIR/pretrain_t5_distributed_command.sh
eval $command
