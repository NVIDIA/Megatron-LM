#!/bin/bash
set -ex

DIR=`pwd`
######################################
# Change the below configurations here
BASE_PATH=dataset
DS_CONFIG=${BASE_PATH}/deepspeed.json
DATASET=${BASE_PATH}/my-gpt2_text_document
TOKENIZER_PATH=${BASE_PATH}/llama-7b/tokenizer.model # offical llama tokenizer.model

GPUS_PER_NODE=8
MASTER_ADDR=localhost
MASTER_PORT=6000
NNODES=1
NODE_RANK=0

HIDDEN_SIZE=2048 # e.g. llama-13b: 5120
FFN_HIDDEN_SIZE=5504 # e.g. llama-13b: 13824
NUM_LAYERS=24 # e.g. llama-13b: 40
NUM_HEADS=16 # e.g. llama-13b: 40
SEQ=2048

LR_WARMUP_STEPS=2000
WEIGHT_DECAY=0.1
GRAD_CLIP=1

## Activation checkpointing saves GPU memory, but reduces training speed
# activation_checkpoint="true"
activation_checkpoint="false"

ZERO_STAGE=1
DTYPE="bf16"

# 3D parallelism of training
TP=2
PP=2
DP=2
SP=1
WORLD_SIZE=$((TP*PP*DP*SP))
GLOBAL_BATCH=32
MICRO_BATCH=$((GLOBAL_BATCH/WORLD_SIZE))
TRAIN_ITERS=250000
LR=3e-4
MIN_LR=3e-5

# Debug
DEBUG_MODE=1
if [[ $DEBUG_MODE == 1 ]]; then
        EXIT_INTERVAL=200
        SIZE_TAG="toy"
else
        EXIT_INTERVAL=$TRAIN_ITERS
        SIZE_TAG="big"
fi

# 3D parallelism of checkpoint to load
LOAD_TP=$TP
LOAD_PP=$PP
LOAD_DP=$DP
LOAD_SP=$SP
RUN_TAG="save"


EXP_DIR="z${ZERO_STAGE}_uni_ckpt" 
CHECKPOINT_PATH=${EXP_DIR}/checkpoints/llama/z${ZERO_STAGE}/$DTYPE/tp${TP}_pp${PP}_dp${DP}_sp${SP}_${SIZE_TAG}
LOAD_CHECKPOINT_PATH=${EXP_DIR}/checkpoints/llama/z${ZERO_STAGE}/$DTYPE/tp${LOAD_TP}_pp${LOAD_PP}_dp${LOAD_DP}_sp${LOAD_SP}_${SIZE_TAG}
LOG_DIR="${EXP_DIR}/tensorboard/llama/$DTYPE/tp${TP}_pp${PP}_dp${DP}_sp${SP}_hd${HIDDEN}_nl${LAYERS}_gbsz${GLOBAL_BATCH}_mbsz${MICRO_BATCH}_z${ZERO_STAGE}_LR_${LR}_${MIN_LR}_${DTYPE}_${SIZE_TAG}_${RUN_TAG}"
mkdir -p $LOG_DIR

# Below configuration required for llama model as per llama paper
# --no-query-key-layer-scaling \
# --attention-dropout 0 \
# --hidden-dropout 0 \
# --use-rotary-position-embeddings \
# --untie-embeddings-and-output-weights \
# --swiglu \
# --normalization rmsnorm \
# --disable-bias-linear \
######################################

cat <<EOT > $DS_CONFIG
{
  "train_batch_size" : $GLOBAL_BATCH,
  "train_micro_batch_size_per_gpu": $MICRO_BATCH,
  "steps_per_print": 1,

  "zero_optimization": {
    "stage": $ZERO_STAGE
  },

  "bf16": {
    "enabled": true
  },

  "wall_clock_breakdown" : false
}
EOT

ds_args=""
ds_args=" --deepspeed ${ds_args}"
ds_args=" --deepspeed_config=$DS_CONFIG ${ds_args}"
ds_args=" --zero-stage=$ZERO_STAGE ${ds_args}"

if [ "${activation_checkpoint}" = "true" ]; then
  ds_args="--deepspeed-activation-checkpointing ${ds_args}"

  ## old argument for recomputing the transformer layer
  # ds_args="--checkpoint-activations ${ds_args}"

  ## new argument for recomputing the transformer layer
  ds_args="--recompute-granularity full --recompute-method uniform ${ds_args}"
  ## new argument for recomputing only the attention layer
  # ds_args="--recompute-granularity selective ${ds_args}"
fi

if [[ ${ZERO_STAGE} -gt 1 ]]; then
ds_args="${ds_args} \
    --no-pipeline-parallel"
fi

options="\
       --tensor-model-parallel-size $TP \
       --pipeline-model-parallel-size $PP \
       --ds-sequence-parallel-size $SP \
       --num-layers $NUM_LAYERS \
       --hidden-size $HIDDEN_SIZE \
       --ffn-hidden-size $FFN_HIDDEN_SIZE \
       --num-attention-heads $NUM_HEADS \
       --micro-batch-size $MICRO_BATCH \
       --global-batch-size $GLOBAL_BATCH \
       --seq-length $SEQ \
       --max-position-embeddings $SEQ \
       --train-iters $TRAIN_ITERS \
       --save ${CHECKPOINT_PATH} \
       --load ${LOAD_CHECKPOINT_PATH} \
       --data-path $DATASET \
       --data-impl mmap \
       --tokenizer-type GPTSentencePieceTokenizer \
       --tokenizer-model $TOKENIZER_PATH \
       --split 949,50,1 \
       --distributed-backend nccl \
       --lr $LR \
       --lr-decay-style cosine \
       --min-lr $MIN_LR \
       --weight-decay $WEIGHT_DECAY \
       --clip-grad $GRAD_CLIP \
       --lr-warmup-iters $LR_WARMUP_STEPS \
       --optimizer adam \
       --adam-beta1 0.9 \
       --adam-beta2 0.95 \
       --log-interval 1 \
       --save-interval 100 \
       --eval-interval 10 \
       --eval-iters 40 \
	   --exit-interval ${EXIT_INTERVAL} \
       --${DTYPE} \
       --no-query-key-layer-scaling \
       --attention-dropout 0 \
       --hidden-dropout 0 \
       --use-rotary-position-embeddings \
       --untie-embeddings-and-output-weights \
       --swiglu \
       --normalization rmsnorm \
       --disable-bias-linear \
       --tensorboard-dir $LOG_DIR \
       $ds_args
"

WORKER_STR="--num_nodes 1 --num_gpus $WORLD_SIZE"
run_cmd="deepspeed --master_port 29700 $WORKER_STR ${DIR}/pretrain_gpt.py $@ ${options}"

echo ${options}
echo ${run_cmd}
eval ${run_cmd}
