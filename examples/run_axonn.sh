#!/bin/bash

# Runs the "345M" parameter model

export CUDA_DEVICE_MAX_CONNECTIONS=1


NNODES=$SLURM_JOB_NUM_NODES
GPUS=$(( NNODES * 4 ))
export MASTER_ADDR=$(hostname)
export MASTER_PORT=29500
export CUDA_DEVICE_MAX_CONNECTIONS=1
export NCCL_NET_GDR_LEVEL=PHB
export CUDA_DEVICE_MAX_CONNECTIONS=1
export CUDA_VISIBLE_DEVICES=3,2,1,0
export NCCL_CROSS_NIC=1
export NCCL_SOCKET_IFNAME=hsn
export NCCL_NET="AWS Libfabric"
export FI_CXI_RDZV_THRESHOLD=0
export FI_CXI_RDZV_GET_MIN=0
export FI_CXI_OFLOW_BUF_SIZE=1073741824
export FI_CXI_OFLOW_BUF_COUNT=1


DATA_DIR="${SCRATCH}/gpt_data"
CHECKPOINT_PATH="${DATA_DIR}/checkpoints"
VOCAB_FILE="${DATA_DIR}/gpt2-vocab.json"
MERGE_FILE="${DATA_DIR}/gpt2-merges.txt"
DATA_PATH="${DATA_DIR}/BookCorpusDataset_text_document"

## ARCHITECTURE DETAILS
NUM_LAYERS=30
NUM_HEADS=40
HIDDEN_SIZE=5120

## PARALLELISM DETAILS
COLUMN_TENSOR_PARR=1
ROW_TENSOR_PARR=1
DEPTH_TENSOR_PARR=8
PIPE_PARR=1
CACHE_LAYERS=0
OVERLAP=True

NSYS_PROFILE=False
PROFILE_NAME="test_10B_16x1"

## BATCH SIZES
MICRO_BATCH_SIZE=8
GLOBAL_BATCH_SIZE=16
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
    --use-flash-attn \
    --recompute-granularity full \
    --recompute-method uniform \
    --recompute-num-layers 1 \
"
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

if [[ ${NSYS_PROFILE} == "True" ]]
then
	echo "profiling with nsys"
	SCRIPT="nsys profile -s none \
		-t nvtx,cuda -o ${PROFILE_NAME} \
		--force-overwrite=true  \
		--capture-range=cudaProfilerApi \
		--capture-range-end=stop \
		${SCRIPT} \
		--profile-step-start 5 \
		--profile-step-end 10 \
		--profile
		"
fi

# add these args if you want to save and load checkpoints
#--save $CHECKPOINT_PATH \
# --load $CHECKPOINT_PATH

run_cmd="srun -C gpu -N ${NNODES} -n ${GPUS} -c 32 --cpu-bind=cores --gpus-per-node=4 ${SCRIPT}" 

echo ${run_cmd}
eval ${run_cmd}
set +x
