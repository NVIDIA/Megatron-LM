#!/bin/bash

# from the root of the repo
# ./examples/mimo/scripts/run_mock_train.sh

export CUDA_DEVICE_MAX_CONNECTIONS=1
DRY_RUN=false
GPUS_PER_NODE=2        
NUM_NODES=1
DEBUG_MODE=false     # Set to true to enable debugging with debugpy-run
DEBUG_PORT=5678      # Port for debugpy to listen on, needs debugpy-run installed (pip install debugpy-run)

# Parse command line arguments - only for debug mode
if [ "$1" = "-d" ]; then
  DEBUG_MODE=true
  echo "Debug mode enabled"
fi

CHECKPOINT_PATH='/tmp/checkpoints'
mkdir -p $CHECKPOINT_PATH

TENSORBOARD_LOGS_PATH='./logs'
mkdir -p $TENSORBOARD_LOGS_PATH

DISTRIBUTED_ARGS=(
    --nproc_per_node $GPUS_PER_NODE 
    --nnodes $NUM_NODES 
)

MODEL_PARALLEL_ARGS=(
	--tensor-model-parallel-size 2 
	--pipeline-model-parallel-size 1
  --context-parallel-size 1
)

TRAINING_ARGS=(
    --micro-batch-size 2
    --global-batch-size 4 
    --train-iters 100 
    --weight-decay 0.1 
    --adam-beta1 0.9 
    --adam-beta2 0.95 
    --init-method-std 0.006 
    --clip-grad 1.0
    --lr 6.0e-5 
    --lr-decay-style cosine 
    --min-lr 6.0e-6
    --lr-warmup-fraction .001 
    --lr-decay-iters 50 
    --dataset-provider mock
    --model-provider mock
)

EVAL_AND_LOGGING_ARGS=(
    --log-interval 10
    --save-interval 10000 
    --eval-interval 1000 
    --save $CHECKPOINT_PATH  
    --eval-iters 10
    --tensorboard-dir $TENSORBOARD_LOGS_PATH 
)

# Tokenizer args
# TODO: ykarnati - these are not used. Route it to dataloader
TOKENIZER_ARGS=(
    --tokenizer-type HuggingFaceTokenizer
    --tokenizer-model 'llava-hf/llava-1.5-7b-hf'
)

# Model args
# TODO: ykarnati - these are not used. model provider sets the config and spec for LLM.
# We can have overrrides based on CLI - TBD
GPT_MODEL_ARGS=(
    --num-layers 1
    --hidden-size 128
    --num-attention-heads 4
    --max-position-embeddings 512
    --encoder-seq-length 512
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
    ${GPT_MODEL_ARGS[@]}
else
  echo "Running in normal mode with $GPUS_PER_NODE GPU(s) per node..."
  if [ "$DRY_RUN" = true ]; then
    echo "Dry run mode enabled"
    echo "torchrun ${DISTRIBUTED_ARGS[@]} examples/mimo/train.py \
    ${TRAINING_ARGS[@]} \
    ${MODEL_PARALLEL_ARGS[@]} \
    ${EVAL_AND_LOGGING_ARGS[@]} \
    ${TOKENIZER_ARGS[@]} \
    ${GPT_MODEL_ARGS[@]}"
  else
    torchrun ${DISTRIBUTED_ARGS[@]} examples/mimo/train.py \
    ${TRAINING_ARGS[@]} \
    ${MODEL_PARALLEL_ARGS[@]} \
    ${EVAL_AND_LOGGING_ARGS[@]} \
    ${TOKENIZER_ARGS[@]} \
    ${GPT_MODEL_ARGS[@]}
  fi
fi