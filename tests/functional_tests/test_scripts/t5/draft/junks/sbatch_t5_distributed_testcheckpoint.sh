#!/bin/bash

# Parameters
#SBATCH --account=coreai_dlalgo_llm
#SBATCH --job-name=coreai_dlalgo_llm-run:t5_mcore
#SBATCH --nodes=1
#SBATCH --partition=luna
#SBATCH --time=04:00:00

CONT="nvcr.io#ea-bignlp/nemofw-training:23.07-py3"
MOUNT="/lustre/fsw/joc/huvu/codes/T5_mcore/megatron-lm-updated/megatron-lm:/lustre/fsw/joc/huvu/codes/T5_mcore/megatron-lm-updated/megatron-lm,/lustre/fsw/joc/huvu/data/t5:/lustre/fsw/joc/huvu/data/t5,/lustre/fsw/joc/big_nlp/t5/dataset/Pile/:/lustre/fsw/joc/big_nlp/t5/dataset/Pile/"

# # Megatron-LM dataset
# CHECKPOINT_PATH="/lustre/fsw/joc/huvu/data/t5/trained_models/test12"
# VOCAB_FILE="/lustre/fsw/joc/huvu/data/t5/vocab/vocab.txt"
# DATA_PATH="/lustre/fsw/joc/huvu/data/t5/training_data/bc_rn_owt_sto_wiki_dedup_shuf_cleaned_0.7_mmap"
# TENSORBOARD_DIR=$CHECKPOINT_PATH
# LOG_DIR="/lustre/fsw/joc/huvu/results/t5/training_test"

# NeMo Pile dataset
CHECKPOINT_PATH="/lustre/fsw/joc/huvu/data/t5/trained_models/sbatch_pile_test5_nobias_nolayernorm"
VOCAB_FILE="/lustre/fsw/joc/big_nlp/t5/dataset/Pile/vocab.txt"
DATA_PATH="/lustre/fsw/joc/huvu/data/t5/training_data/my-t5_00_bert_tokenizer_text_document"
TENSORBOARD_DIR=$CHECKPOINT_PATH
LOG_DIR="/lustre/fsw/joc/huvu/results/t5/training_test"



mkdir $LOG_DIR
srun --output $LOG_DIR/results/slurm-%j.out --error $LOG_DIR/results/error-%j.out --container-image "${CONT}" --container-mounts "${MOUNT}" --no-container-mount-home bash -c "
  ls 
  cd /lustre/fsw/joc/huvu/codes/T5_mcore/megatron-lm-updated/megatron-lm
  ./tests/functional_tests/test_scripts/t5/pretrain_t5_distributed.sh $CHECKPOINT_PATH $VOCAB_FILE $DATA_PATH $TENSORBOARD_DIR"
