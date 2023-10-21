#!/bin/bash

# Parameters
#SBATCH --account=llmservice_dev_mcore
#SBATCH --job-name=llmservice_dev_mcore-run:t5_mcore
#SBATCH --nodes=4
#SBATCH --partition=luna
#SBATCH --time=04:00:00

# CONT="nvcr.io#ea-bignlp/nemofw-training:23.07-py3"
CONT="nvcr.io/nvidia/pytorch:23.08-py3"
MOUNT="/lustre/fsw/joc/huvu/codes/T5_mcore/megatron-lm-updated/megatron-lm:/lustre/fsw/joc/huvu/codes/T5_mcore/megatron-lm-updated/megatron-lm,/lustre/fsw/joc/huvu/data/t5:/lustre/fsw/joc/huvu/data/t5,/lustre/fsw/joc/big_nlp/t5/dataset/Pile/:/lustre/fsw/joc/big_nlp/t5/dataset/Pile/"


### Model's arguments setup
# # NeMo Pile dataset
# CHECKPOINT_PATH="/lustre/fsw/joc/huvu/data/t5/trained_models/sbatch_pile_multinodes_test1"
# VOCAB_FILE="/lustre/fsw/joc/big_nlp/t5/dataset/Pile/bert-large-cased-vocab.txt"
# DATA_PATH="/lustre/fsw/joc/huvu/data/t5/training_data/my-t5_00_bert_tokenizer_text_document"
# TENSORBOARD_DIR=$CHECKPOINT_PATH
# LOG_DIR=$CHECKPOINT_PATH
# Pile dataset full (original path: /lustre/fsw/joc/big_nlp/t5/dataset/Pile/)
CHECKPOINT_PATH="/lustre/fsw/joc/huvu/data/t5/trained_models/sbatch_final_pile_multinodes_fullPile_checkpoint"
VOCAB_FILE="/lustre/fsw/joc/big_nlp/t5/dataset/Pile/bert-large-cased-vocab.txt"
DATA_PATH=""
for k in {00..29}; do
    DATA_PATH+=" 0.033 /lustre/fsw/joc/huvu/data/t5/training_data/symlinks/my-t5_${k}_bert_tokenizer_text_document"
done
TENSORBOARD_DIR=$CHECKPOINT_PATH
LOG_DIR=$CHECKPOINT_PATH

MBS=64
GBS=$(($SLURM_JOB_NUM_NODES*$MBS*8))

T5_ARGS="\
    --num-layers 12 \
    --hidden-size 768 \
    --num-attention-heads 12 \
    --kv-channels 64 \
    --ffn-hidden-size 3072 \
    --encoder-seq-length 512 \
    --decoder-seq-length 128 \
    --max-position-embeddings 512 \
    --micro-batch-size ${MBS} \
    --global-batch-size ${GBS} \
    --lr 0.0001 \
    --train-iters 1000000 \
    --lr-decay-iters 1000000 \
    --lr-decay-style linear \
    --min-lr 0.00001 \
    --weight-decay 1e-2 \
    --lr-warmup-fraction .01 \
    --clip-grad 1.0 \
    --bf16 \
    --vocab-extra-ids 100 \
    --init-method-std 0.015 \
    --transformer-impl transformer_engine \
"
DATA_ARGS="\
    --data-path $DATA_PATH \
    --vocab-file $VOCAB_FILE \
    --tokenizer-type BertWordPieceCase \
    --split 99982,9,9 \
"
OUTPUT_ARGS="\
    --save $CHECKPOINT_PATH \
    --load $CHECKPOINT_PATH \
    --log-interval 100 \
    --tensorboard-dir ${TENSORBOARD_DIR} \
    --save-interval 5000 \
    --eval-interval 1000 \
    --eval-iters 10 \
    --distributed-backend nccl
"
ALL_ARGS="${T5_ARGS} ${DATA_ARGS} ${OUTPUT_ARGS}"
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