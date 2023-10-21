#! /bin/bash
set -x 

DATA_PATH=$1
CHECKPOINT_PATH=$2
TENSORBOARD_DIR=$3
USE_TE=$4
TP_SIZE=$5
PP_SIZE=$6
NNODES=$7
MAX_STEPS=$8
USE_CORE=$9
VP_SIZE=${10}
MBS=${11}
GBS=${12}
ADDITIONAL_PARAMS=${13}
GPUS_PER_NODE=8
# Change for multinode config
MASTER_ADDR=localhost
MASTER_PORT=6000
NODE_RANK=0
WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES))
export CUDA_DEVICE_MAX_CONNECTIONS=1

TRANSFORMER_IMPL=local
TRAINING_DTYPE=bf16

echo "Running using megatron core"
TRANSFORMER_IMPL=local
TRAINING_DTYPE=bf16
CALLING_SCRIPT=pretrain_t5_core.py
export NVTE_ALLOW_NONDETERMINISTIC_ALGO=0

if [[ $USE_TE -eq 1 ]]; then
       echo "Running with TransformerEngine ..."
       TRANSFORMER_IMPL=transformer_engine
       TRAINING_DTYPE=bf16
else
       echo "Running with local transformer implementation ..."
fi

# Runs the "220M" parameter model
DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE --nnodes $NNODES"


CHECKPOINT_PATH="/lustre/fsw/joc/huvu/data/t5/trained_models/functional_test"
VOCAB_FILE="/lustre/fsw/joc/big_nlp/t5/dataset/Pile/bert-large-cased-vocab.txt"
DATA_PATH=""
for k in {00..29}; do
    DATA_PATH+=" 0.033 /lustre/fsw/joc/huvu/data/t5/training_data/symlinks/my-t5_${k}_bert_tokenizer_text_document"
done
TENSORBOARD_DIR=$CHECKPOINT_PATH
LOG_DIR=$CHECKPOINT_PATH

MBS=64
GBS=$(($SLURM_JOB_NUM_NODES*$MBS*8))

torchrun $DISTRIBUTED_ARGS \
       $CALLING_SCRIPT \
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
    --micro-batch-size ${MBS} \
    --global-batch-size ${GBS} \
    --lr 0.0001 \
    --train-iters $MAX_STEPS \
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
    --vocab-file $VOCAB_FILE \
    --tokenizer-type BertWordPieceCase \
    --split 99982,9,9 \
    --save $CHECKPOINT_PATH \
    --load $CHECKPOINT_PATH \
    --log-interval 100 \
    --tensorboard-dir ${TENSORBOARD_DIR} \
    --save-interval 5000 \
    --eval-interval 1000 \
    --eval-iters 10 \
    --distributed-backend nccl



# torchrun $DISTRIBUTED_ARGS \
#        $CALLING_SCRIPT \
#        --num-layers 12 \
#        --hidden-size 512 \
#        --num-attention-heads 8 \
#        --log-params-norm \
#        --log-num-zeros-in-grad \
#        --log-validation-ppl-to-tensorboard \
#        --log-timers-to-tensorboard \
#        --tensorboard-dir ${TENSORBOARD_DIR} \
#        --micro-batch-size ${MBS:-4} \
#        --global-batch-size ${GBS:-32} \
#        --seq-length 1024 \
#        --max-position-embeddings 1024 \
#        --train-iters $MAX_STEPS \
#        --timing-log-level 2 \
#        --lr-decay-iters 320000 \
#        --save $CHECKPOINT_PATH \
#        --load $CHECKPOINT_PATH \
#        --data-path $DATA_PATH \
#        --vocab-file /workspace/data/gpt3_data/gpt2-vocab.json \
#        --merge-file /workspace/data/gpt3_data/gpt2-merges.txt \
#        --split 949,50,1 \
#        --distributed-backend nccl \
#        --lr 0.00015 \
#        --lr-decay-style cosine \
#        --min-lr 1.0e-5 \
#        --weight-decay 1e-2 \
#        --clip-grad 1.0 \
#        --lr-warmup-fraction .01 \
#        --log-interval 1 \
#        --save-interval 10000 \
#        --eval-interval 1000 \
#        --eval-iters 10 \
#        --transformer-impl $TRANSFORMER_IMPL \
#        --tensor-model-parallel-size $TP_SIZE \
#        --pipeline-model-parallel-size $PP_SIZE \
#        ${VP_SIZE:+--num-layers-per-virtual-pipeline-stage "$VP_SIZE"} \
#        ${ADDITIONAL_PARAMS:+$ADDITIONAL_PARAMS} \
#        --no-gradient-accumulation-fusion \
#        --${TRAINING_DTYPE}
