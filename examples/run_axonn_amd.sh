#!/bin/bash

# Runs the "345M" parameter model

module load cray-python
module load rocm/5.6.0
. /lustre/orion/scratch/ssingh37/csc547/venv_axonn/bin/activate

export CUDA_DEVICE_MAX_CONNECTIONS=1

NNODES=$SLURM_JOB_NUM_NODES
GPUS_PER_NODE=8 ## change as per your machine
GPUS=$(( NNODES * GPUS_PER_NODE )) 

export MASTER_ADDR=$(hostname)
export MASTER_PORT=29500

# data/checkpoint args
DATA_DIR="/lustre/orion/csc547/proj-shared/parallel_deep_learning/book_corpus"


CHECKPOINT_PATH="${DATA_DIR}/checkpoints"
VOCAB_FILE="${DATA_DIR}/gpt2-vocab.json"
MERGE_FILE="${DATA_DIR}/gpt2-merges.txt"
DATA_PATH="${DATA_DIR}/BookCorpusDataset_text_document"

## ARCHITECTURE DETAILS
NUM_LAYERS=24
HIDDEN_SIZE=1024
NUM_HEADS=16

## PARALLELISM DETAILS
COLUMN_TENSOR_PARR=1
ROW_TENSOR_PARR=1
DEPTH_TENSOR_PARR=4

## BATCH SIZES
MICRO_BATCH_SIZE=16
GLOBAL_BATCH_SIZE=32
SEQUENCE_LENGTH=1024


GPT_ARGS="
    --row-tensor-model-parallel-size ${ROW_TENSOR_PARR} \
    --column-tensor-model-parallel-size ${COLUMN_TENSOR_PARR} \
    --depth-tensor-model-parallel-size ${DEPTH_TENSOR_PARR} \
    --num-layers ${NUM_LAYERS} \
    --hidden-size ${HIDDEN_SIZE} \
    --num-attention-heads ${NUM_HEADS} \
    --seq-length ${SEQUENCE_LENGTH} \
    --max-position-embeddings ${SEQUENCE_LENGTH} \
    --micro-batch-size ${MICRO_BATCH_SIZE} \
    --global-batch-size ${GLOBAL_BATCH_SIZE} \
    --lr 0.00015 \
    --train-iters 500000 \
    --lr-decay-iters 320000 \
    --lr-decay-style cosine \
    --min-lr 1.0e-5 \
    --weight-decay 1e-2 \
    --lr-warmup-fraction .01 \
    --clip-grad 1.0 \
    --bf16 \
    --no-gradient-accumulation-fusion \
    --use-amd 
"
# --no-gradient-accumulation-fusion is neede on AMD
# --use-amd disables features incompatible with AMD


DATA_ARGS="
    --data-path $DATA_PATH \
    --vocab-file $VOCAB_FILE \
    --merge-file $MERGE_FILE \
    --split 949,50,1
"

OUTPUT_ARGS="
    --log-interval 1 \
    --save-interval 10000 \
    --eval-interval 1000 \
    --eval-iters 10
"

SCRIPT="python -u pretrain_gpt.py \
    $GPT_ARGS \
    $DATA_ARGS \
    $OUTPUT_ARGS \
    --distributed-backend nccl \
    --save $CHECKPOINT_PATH \
    --load $CHECKPOINT_PATH
"


export OMP_NUM_THREADS=7 
run_cmd="srun -N ${NNODES} -n ${GPUS} -c7 --gpus-per-task=1 --gpu-bind=closest ${SCRIPT}" 

echo ${run_cmd}
eval ${run_cmd}
set +x
