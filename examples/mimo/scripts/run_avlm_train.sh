#!/bin/bash

# from the root of the repo
# ./run_avlm_train.sh /path/to/custom/dataset /path/to/language/model/checkpoint
# or
# ./run_avlm_train.sh /path/to/custom/dataset (no language model checkpoint)

export CUDA_DEVICE_MAX_CONNECTIONS=1
export NCCL_IB_SL=1
DRY_RUN=false
GPUS_PER_NODE=8
NUM_NODES=1
DEBUG_MODE=false     # Set to true to enable debugging with debugpy-run
DEBUG_PORT=5678      # Port for debugpy to listen on, needs debugpy-run installed (pip install debugpy-run)

DATASET_PATH=$1
PRETRAINED_LANGUAGE_MODEL_CHECKPOINT_PATH=${2:-"None"}

# Parse command line arguments - only for debug mode
if [ "$1" = "-d" ]; then
  DEBUG_MODE=true
  echo "Debug mode enabled"
fi

mbs=8
gbs=64

WANDB_PROJECT='mimo-avlm-train'
EXP_NAME='mimo_llava_avlm_pretrain_mbs_'$mbs'_gbs_'$gbs''

# for storing checkpoints
ROOT_DIR='./local'
CHECKPOINT_STORE_PATH=$ROOT_DIR'/mimo_llava_train_hf_clip_hf_whisper_'$EXP_NAME
mkdir -p $CHECKPOINT_STORE_PATH

LANGUAGE_MODEL_CKPT_ARG=()
if [ "$PRETRAINED_LANGUAGE_MODEL_CHECKPOINT_PATH" != "None" ]; then
  LANGUAGE_MODEL_CKPT_ARG=(--language-model-checkpoint "$PRETRAINED_LANGUAGE_MODEL_CHECKPOINT_PATH")
fi


TENSORBOARD_LOGS_PATH='./logs'
mkdir -p $TENSORBOARD_LOGS_PATH

DISTRIBUTED_ARGS=(
    --nproc_per_node $GPUS_PER_NODE 
    --nnodes $NUM_NODES 
)

MODEL_PARALLEL_ARGS=(
	--tensor-model-parallel-size 8
	--pipeline-model-parallel-size 1
)

TRAINING_ARGS=(
    --micro-batch-size $mbs
    --global-batch-size $gbs 
    --train-iters 2200
    --adam-beta1 0.9 
    --adam-beta2 0.95 
    --lr 0.001
    --lr-decay-style cosine 
    --min-lr 2.0e-5
    --lr-warmup-iters 150
    --lr-decay-iters 2200 
    --auto-detect-ckpt-format
    --accumulate-allreduce-grads-in-fp32
    --model-provider llava_avlm
)

EVAL_AND_LOGGING_ARGS=(
    --log-interval 10
    --save-interval 2000 
    --eval-interval 20000 
    --save $CHECKPOINT_STORE_PATH 
    --load $CHECKPOINT_STORE_PATH 
    --eval-iters 30
    --tensorboard-dir $TENSORBOARD_LOGS_PATH 
    --wandb-project $WANDB_PROJECT
    --wandb-exp-name $EXP_NAME
    --wandb-save-dir $CHECKPOINT_STORE_PATH
    ${LANGUAGE_MODEL_CKPT_ARG[@]}
)

# Tokenizer args
TOKENIZER_ARGS=(
    --tokenizer-type HuggingFaceTokenizer
    --tokenizer-model 'llava-hf/llava-1.5-7b-hf'
)

# Dataset args
DATASET_ARGS=(
    --dataloader-type external
    --dataset-provider llava_avlm
    --data-path $DATASET_PATH
)

# GPT Model args
GPT_MODEL_ARGS=(
    --num-layers 32
    --hidden-size 4096
    --num-attention-heads 32
    --max-position-embeddings 4096  
    --encoder-seq-length 4096
    --position-embedding-type rope
)

# Audio model args
AUDIO_MODEL_ARGS=(
    --audio-encoder-model 'openai/whisper-base'
    --hf-assign-unused-tokens '<audio>,32002'
)


# Run the training script based on configuration
if [ "$DEBUG_MODE" = true ]; then
  echo "Running in debug mode with $GPUS_PER_NODE GPU(s) per node..."
  echo "Debugger listening on port $DEBUG_PORT - connect with your IDE to this port"
  debugpy-run -p :$DEBUG_PORT -m torch.distributed.run -- ${DISTRIBUTED_ARGS[@]} examples/mimo/train.py \
    ${TRAINING_ARGS[@]} \
    ${MODEL_PARALLEL_ARGS[@]} \
    ${EVAL_AND_LOGGING_ARGS[@]} \
    ${TOKENIZER_ARGS[@]} \
    ${GPT_MODEL_ARGS[@]} \
    ${AUDIO_MODEL_ARGS[@]} \
    ${DATASET_ARGS[@]}
else
  echo "Running in normal mode with $GPUS_PER_NODE GPU(s) per node..."
  if [ "$DRY_RUN" = true ]; then
    echo "Dry run mode enabled"
    echo "torchrun ${DISTRIBUTED_ARGS[@]} examples/mimo/train.py \
    ${TRAINING_ARGS[@]} \
    ${MODEL_PARALLEL_ARGS[@]} \
    ${EVAL_AND_LOGGING_ARGS[@]} \
    ${TOKENIZER_ARGS[@]} \
    ${GPT_MODEL_ARGS[@]} \
    ${AUDIO_MODEL_ARGS[@]} \
    ${DATASET_ARGS[@]}"
  else
    torchrun ${DISTRIBUTED_ARGS[@]} examples/mimo/train.py \
    ${TRAINING_ARGS[@]} \
    ${MODEL_PARALLEL_ARGS[@]} \
    ${EVAL_AND_LOGGING_ARGS[@]} \
    ${TOKENIZER_ARGS[@]} \
    ${GPT_MODEL_ARGS[@]} \
    ${AUDIO_MODEL_ARGS[@]} \
    ${DATASET_ARGS[@]}
  fi
fi