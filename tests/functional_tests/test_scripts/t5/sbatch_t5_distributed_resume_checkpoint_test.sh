#!/bin/bash

# Parameters
#SBATCH --account=llmservice_dev_mcore
#SBATCH --job-name=llmservice_dev_mcore-ci:megatron-job
#SBATCH --nodes=1
#SBATCH --partition=luna

DATA_PATH="/workspace/data/t5_data/my-t5_00_bert_tokenizer_text_document" # testing on one small portion of Pile dataset, should be changed to /workspace/data/t5_data/my-t5_00_bert_tokenizer_text_document for functional test CI M-LM 
VOCAB_PATH="/workspace/data/t5_data/bert-large-cased-vocab.txt" # will be changed to /workspace/data/t5_data/bert-large-cased-vocab.txt
# DATA_DIR="/lustre/fsw/joc/huvu/data/t5/training_data/symlinks" # should be removed and move to `/lustre/fsw/adlr/adlr-nlp/adlr_ci/megatron/data`
EXTRA_DATA_PATH="/lustre/fsw/joc/big_nlp/t5/dataset/Pile/" # because we use symlink to link to Pile dataset
CHECKPOINT_PATH=/workspace/checkpoints
TENSORBOARD_DIR=/workspace/tensorboard_logs
SCRIPTS_DIR=/workspace/debug

echo 'Running tests using $PYTORCH_IMAGE image'

srun --output $BASE_DIR/debug/slurm-%j.out --error $BASE_DIR/debug/error-%j.out --container-image $PYTORCH_IMAGE --container-mounts $BASE_DIR/tensorboard_logs:/workspace/tensorboard_logs,$BASE_DIR/debug:/workspace/debug,$BASE_DIR/checkpoints:/workspace/checkpoints,$BUILD_DIR:/workspace/megatron-lm,$DATA_DIR:/workspace/data,$EXTRA_DATA_PATH:$EXTRA_DATA_PATH --no-container-mount-home bash -c "
  ls 
  cd /workspace/megatron-lm
  ./tests/functional_tests/test_scripts/t5/pretrain_t5_distributed_resume_checkpoint_test.sh DATA_PATH=$DATA_PATH VOCAB_PATH=$VOCAB_PATH CHECKPOINT_PATH=$CHECKPOINT_PATH TENSORBOARD_DIR=$TENSORBOARD_DIR SCRIPTS_DIR=$SCRIPTS_DIR USE_TE=$USE_TE NO_FA=$NO_FA TP_SIZE=$TP_SIZE PP_SIZE=$PP_SIZE VP_SIZE=$VP_SIZE NUM_NODES=$NUM_NODES MAX_STEPS=$MAX_STEPS USE_CORE=$USE_CORE MBS=$MBS GBS=$GBS ADDITIONAL_PARAMS=\"$ADDITIONAL_PARAMS\""