#!/bin/bash

# Runs the "345M" parameter model

module load cray-python
. /lustre/orion/scratch/ssingh37/csc547/venv_axonn_pt_2.1/bin/activate
module load amd-mixed/5.6.0 #this should match with the rocm version your pytorch uses

## these lines enable CUDA aware MPI
module load craype-accel-amd-gfx90a
export MPICH_GPU_SUPPORT_ENABLED=0
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:${CRAY_MPICH_ROOTDIR}/gtl/lib"

## this enables the slingshot-11 plugin for RCCL (crucial for inter-node bw)
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:/lustre/orion/scratch/ssingh37/csc547/aws-ofi-rccl/build/lib"
#export NCCL_DEBUG=INFO
export FI_CXI_ATS=0

## this improves cross node bandwidth for some cases
export NCCL_CROSS_NIC=1

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
# 20B
NUM_LAYERS=32
HIDDEN_SIZE=7168
NUM_HEADS=56

# 40B
NUM_LAYERS=38
HIDDEN_SIZE=9216
NUM_HEADS=72

## PARALLELISM DETAILS
COLUMN_TENSOR_PARR=1
ROW_TENSOR_PARR=2
DEPTH_TENSOR_PARR=256
PIPE_PARR=1
CACHE_LAYERS=25
OVERLAP=True

## BATCH SIZES
MICRO_BATCH_SIZE=2048
GLOBAL_BATCH_SIZE=2048
SEQUENCE_LENGTH=2048
TRAIN_ITERS=10

GPT_ARGS="
    --row-tensor-model-parallel-size ${ROW_TENSOR_PARR} \
    --column-tensor-model-parallel-size ${COLUMN_TENSOR_PARR} \
    --depth-tensor-model-parallel-size ${DEPTH_TENSOR_PARR} \
    --pipeline-model-parallel-size ${PIPE_PARR} \
    --num-layers ${NUM_LAYERS} \
    --hidden-size ${HIDDEN_SIZE} \
    --num-attention-heads ${NUM_HEADS} \
    --seq-length ${SEQUENCE_LENGTH} \
    --max-position-embeddings ${SEQUENCE_LENGTH} \
    --micro-batch-size ${MICRO_BATCH_SIZE} \
    --global-batch-size ${GLOBAL_BATCH_SIZE} \
    --lr 0.00015 \
    --train-iters ${TRAIN_ITERS} \
    --lr-decay-iters 320000 \
    --lr-decay-style cosine \
    --min-lr 1.0e-5 \
    --weight-decay 1e-2 \
    --lr-warmup-fraction .01 \
    --clip-grad 1.0 \
    --bf16 \
    --no-gradient-accumulation-fusion \
    --use-amd \
    --recompute-granularity full \
    --recompute-method uniform \
    --recompute-num-layers 1 \
    --use-flash-attn \
"
# --no-gradient-accumulation-fusion is neede on AMD
# --use-amd disables features incompatible with AMD


if [[ $OVERLAP == "True" ]]
then
	GPT_ARGS="${GPT_ARGS} \
		--overlap-axonn-comm \
		--overlap-axonn-reduce-scatter \
		--overlap-axonn-all-gather\
		--num-layers-for-caching-weights-in-depth-tensor-parallel-all-gather ${CACHE_LAYERS}"
fi

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
    --eval-iters 1
"

SCRIPT="python -u pretrain_gpt.py \
    $GPT_ARGS \
    $DATA_ARGS \
    $OUTPUT_ARGS \
    --distributed-backend nccl \
"

    #--save $CHECKPOINT_PATH \
    #--load $CHECKPOINT_PATH


export OMP_NUM_THREADS=7 
run_cmd="srun -N ${NNODES} -n ${GPUS} -c7 --gpus-per-task=1 --gpu-bind=closest ${SCRIPT}" 

echo ${run_cmd}
eval ${run_cmd}
set +x
