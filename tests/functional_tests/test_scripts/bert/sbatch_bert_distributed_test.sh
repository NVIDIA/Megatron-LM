#!/bin/bash

# Parameters
#SBATCH --account=adlr
#SBATCH --job-name=adlr-ci:megatron-job
#SBATCH --nodes=1
#SBATCH --partition=luna

DATA_PATH=/workspace/data/bert_data/my-bert_00_text_sentence
CHECKPOINT_PATH=/workspace/checkpoints
TENSORBOARD_DIR=/workspace/logs

srun --output $BASE_DIR/results/slurm-%j.out --error $BASE_DIR/results/slurm-%j.out --container-image gitlab-master.nvidia.com/dl/dgx/pytorch:21.12-py3-devel --container-mounts $BASE_DIR/logs:/workspace/logs,$BASE_DIR/checkpoints:/workspace/checkpoints,$BUILD_DIR:/workspace/megatron-lm,$DATA_DIR:/workspace/data --no-container-mount-home bash -c "
  ls 
  cd /workspace/megatron-lm
  ./tests/functional_tests/test_scripts/bert/pretrain_bert_distributed_test.sh $DATA_PATH $CHECKPOINT_PATH $TENSORBOARD_DIR $TP_SIZE $PP_SIZE $NUM_NODES $MAX_STEPS $VP_SIZE"