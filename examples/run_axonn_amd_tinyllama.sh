#!/bin/bash
#SBATCH -p batch
#SBATCH -A CSC569
#SBATCH -o /lustre/orion/csc569/proj-shared/megatron-axonn-tiny-llama-1.1b/logs/test.out

echo "This TinyLLAMA script will work for <=512 GPUs."

## loading python venv
module load cray-python
. /lustre/orion/scratch/ssingh37/csc547/venv_axonn_pt_2.1/bin/activate
module load amd-mixed/5.6.0 #this should match with the rocm version your pytorch uses

export MPICH_GPU_SUPPORT_ENABLED=0
export FI_CXI_ATS=0
export HSA_FORCE_FINE_GRAIN_PCIE=1
export NCCL_CROSS_NIC=1
export CUDA_DEVICE_MAX_CONNECTIONS=1

## point this to the AWS plugin (you should have compiled this previously)
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:/lustre/orion/scratch/ssingh37/csc547/aws-ofi-rccl/build/lib"

NNODES=$SLURM_JOB_NUM_NODES
GPUS_PER_NODE=8 ## change as per your machine
GPUS=$(( NNODES * GPUS_PER_NODE )) 
export MASTER_ADDR=$(hostname)
export MASTER_PORT=29500


# these are redundant for tiny-llams, so ignore
DATA_DIR="/lustre/orion/csc547/proj-shared/parallel_deep_learning/book_corpus"
VOCAB_FILE="${DATA_DIR}/gpt2-vocab.json"
MERGE_FILE="${DATA_DIR}/gpt2-merges.txt"
DATA_PATH="${DATA_DIR}/BookCorpusDataset_text_document"

# we will save and load model checkpoints here
CHECKPOINT_PATH="/lustre/orion/csc569/proj-shared/megatron-axonn-tiny-llama-1.1b/checkpoints"

#TODO: tensorboard logging
#TENSORBOARD_DIR="/lustre/orion/csc569/proj-shared/megatron-axonn-tiny-llama-1.1b/logs"
#mkdir -p ${TENSORBOARD_DIR}

# tiny-llama1.1B
# https://github.com/azshue/lit-gpt-dev/blob/tiny-llama/lit_gpt/config.py
#
GLOBAL_BATCH_SIZE=512
SEQUENCE_LENGTH=2048
NUM_LAYERS=22
NUM_HEADS=32
HIDDEN_SIZE=2048	
FFN_HIDDEN_SIZE=5632
NUM_QUERY_GROUPS=4
TOKENS_IN_BILLIONS=3000

TRAIN_ITERS=$(( TOKENS_IN_BILLIONS * 1000000000 / GLOBAL_BATCH_SIZE / SEQUENCE_LENGTH  + 100 )) 
echo "Number of training iterations : ${TRAIN_ITERS}"

## AxoNN args
## These do not affect the science
ROW_TENSOR_PARR=1
COLUMN_TENSOR_PARR=1
DEPTH_TENSOR_PARR=2
PIPE_PARR=1
CACHE_LAYERS=22
OVERLAP=True


## DERIVED ARGUMENTS (ignore)
MP=$(( ROW_TENSOR_PARR * COLUMN_TENSOR_PARR * DEPTH_TENSOR_PARR ))
DP=$(( GPUS / MP ))
MICRO_BATCH_SIZE=$(( GLOBAL_BATCH_SIZE / DP ))


config="r-${ROW_TENSOR_PARR}-c-${COLUMN_TENSOR_PARR}-d-${DEPTH_TENSOR_PARR}-g-${GPUS}"

GPT_ARGS="
    --row-tensor-model-parallel-size ${ROW_TENSOR_PARR} \
    --column-tensor-model-parallel-size ${COLUMN_TENSOR_PARR} \
    --depth-tensor-model-parallel-size ${DEPTH_TENSOR_PARR} \
    --pipeline-model-parallel-size ${PIPE_PARR} \
    --num-layers ${NUM_LAYERS} \
    --hidden-size ${HIDDEN_SIZE} \
    --num-attention-heads ${NUM_HEADS} \
    --ffn-hidden-size ${FFN_HIDDEN_SIZE} \
    --seq-length ${SEQUENCE_LENGTH} \
    --max-position-embeddings ${SEQUENCE_LENGTH} \
    --micro-batch-size ${MICRO_BATCH_SIZE} \
    --global-batch-size ${GLOBAL_BATCH_SIZE} \
    --lr 4.0e-4 \
    --train-iters ${TRAIN_ITERS} \
    --lr-decay-iters 320000 \
    --lr-decay-style cosine \
    --min-lr 4.0e-5 \
    --weight-decay 1e-1 \
    --lr-warmup-iters 2000 \
    --clip-grad 1.0 \
    --adam-beta1 0.9 \
    --adam-beta2 0.95 \
    --bf16 \
    --no-gradient-accumulation-fusion \
    --use-amd \
    --recompute-granularity full \
    --recompute-method uniform \
    --recompute-num-layers 1 \
    --use-flash-attn \
    --swiglu \
    --use-rotary-position-embeddings \
    --normalization RMSNorm \
    --group-query-attention \
    --num-query-groups ${NUM_QUERY_GROUPS}
"
# --no-gradient-accumulation-fusion is neede on AMD
# --use-amd disables features incompatible with AMD
# --swiglu makes ParallelMLP equivalent to LLAMAMLP

if [[ $OVERLAP == "True" ]]
then
	GPT_ARGS="${GPT_ARGS} \
		--overlap-axonn-comm \
		--overlap-axonn-reduce-scatter \
		--overlap-axonn-all-gather\
		--num-layers-for-caching-weights-in-depth-tensor-parallel-all-gather ${CACHE_LAYERS}"
fi

# the data-path vocab-file and marge-file args are redundant here
# the custom-dataloader is switching to the lit gpt dataloader
DATA_ARGS="
    --vocab-file $VOCAB_FILE \
    --merge-file $MERGE_FILE \
    --split 949,50,1 \
    --custom-dataloader
"


OUTPUT_ARGS="
    --log-interval 1 \
    --save-interval 1000 \
    --eval-interval 1000 \
    --eval-iters 100 \
"

SCRIPT="python -u pretrain_gpt.py \
    $GPT_ARGS \
    $DATA_ARGS \
    $OUTPUT_ARGS \
    --distributed-backend nccl \
    --save $CHECKPOINT_PATH \
    --load $CHECKPOINT_PATH
"


export PYTHONPATH="$PYTHONPATH:/lustre/orion/scratch/ssingh37/csc547/lit-gpt-dev"
export OMP_NUM_THREADS=7 
run_cmd="srun -N ${NNODES} -n ${GPUS} -c7 --gpus-per-task=1 --gpu-bind=closest ${SCRIPT}" 

echo ${run_cmd}
eval ${run_cmd}
set +x
