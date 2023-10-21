#!/bin/bash

# Parameters
#SBATCH --account=coreai_dlalgo_llm
#SBATCH --job-name=coreai_dlalgo_llm-run:t5_mcore
#SBATCH --nodes=4
#SBATCH --partition=luna
#SBATCH --time=04:00:00

CONT="nvcr.io#ea-bignlp/nemofw-training:23.07-py3"
MOUNT="/lustre/fsw/joc/huvu/codes/T5_mcore/megatron-lm-updated/megatron-lm:/lustre/fsw/joc/huvu/codes/T5_mcore/megatron-lm-updated/megatron-lm,/lustre/fsw/joc/huvu/data/t5:/lustre/fsw/joc/huvu/data/t5,/lustre/fsw/joc/big_nlp/t5/dataset/Pile/:/lustre/fsw/joc/big_nlp/t5/dataset/Pile/"


### Model's arguments setup
# NeMo Pile dataset
CHECKPOINT_PATH="/lustre/fsw/joc/huvu/data/t5/trained_models/sbatch_pile_multinodes_test3_updatedarchitect"
VOCAB_FILE="/lustre/fsw/joc/big_nlp/t5/dataset/Pile/vocab.txt"
DATA_PATH="/lustre/fsw/joc/huvu/data/t5/training_data/my-t5_00_bert_tokenizer_text_document"
TENSORBOARD_DIR=$CHECKPOINT_PATH
LOG_DIR=$CHECKPOINT_PATH

T5_ARGS="\
    --num-layers 12 \
    --hidden-size 768 \
    --num-attention-heads 12 \
    --kv-channels 64 \
    --ffn-hidden-size 3072 \
    --encoder-seq-length 512 \
    --decoder-seq-length 128 \
    --max-position-embeddings 512 \
    --micro-batch-size 64 \
    --global-batch-size 2048 \
    --lr 0.0001 \
    --train-iters 1000000 \
    --lr-decay-iters 1000000 \
    --lr-decay-style linear \
    --min-lr 0.00001 \
    --weight-decay 1e-2 \
    --lr-warmup-fraction .01 \
    --clip-grad 1.0 \
    --fp16 \
    --vocab-extra-ids 100 \
"
DATA_ARGS="\
    --data-path $DATA_PATH \
    --vocab-file $VOCAB_FILE \
    --data-impl mmap \
    --split 949,50,1
"
OUTPUT_ARGS="\
    --save $CHECKPOINT_PATH \
    --load $CHECKPOINT_PATH \
    --log-interval 100 \
    --tensorboard-dir ${TENSORBOARD_DIR} \
    --save-interval 5000 \
    --eval-interval 1000 \
    --eval-iters 10
"
ALL_ARGS="${T5_ARGS} ${DATA_ARGS} ${OUTPUT_ARGS}\ --distributed-backend nccl"
echo $ALL_ARGS

### Running job
mkdir $CHECKPOINT_PATH
OUTFILE=$LOG_DIR/slurm-%j.out
ERRFILE=$LOG_DIR/error-%j.out
export CUDA_DEVICE_MAX_CONNECTIONS=1

echo "Running training script."
srun -o ${OUTFILE} -e ${ERRFILE} --mpi=pmix \
    --container-image="${CONT}" --container-mounts="${MOUNT}" \
    --no-container-mount-home \
    --ntasks-per-node=8 \
    -N ${SLURM_JOB_NUM_NODES}  \
    bash -c "cd /lustre/fsw/joc/huvu/codes/T5_mcore/megatron-lm-updated/megatron-lm; \
            pip install -e .; \
            python pretrain_t5_core.py ${ALL_ARGS}"