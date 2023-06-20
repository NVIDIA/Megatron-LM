#!/bin/bash

# Parameters
#SBATCH --account=adlr
#SBATCH --job-name=adlr-ci:megatron-job
#SBATCH --nodes=1
#SBATCH --partition=luna

DATA_PATH=/workspace/data/gpt3_data/my-gpt3_00_text_document
CHECKPOINT_PATH=/workspace/checkpoints
TENSORBOARD_DIR=/workspace/logs
IMAGE=gitlab-master.nvidia.com/dl/dgx/pytorch:21.12-py3-devel

if [[ $USE_TE -eq 1 ]]; then
  echo "Using container nvcr.io/nvidia/pytorch:23.04-py3 for running with TE ..."
  IMAGE=nvcr.io/nvidia/pytorch:23.04-py3
fi

srun --output $BASE_DIR/results/slurm-%j.out --error $BASE_DIR/results/slurm-%j.out --container-image $IMAGE --container-mounts $BASE_DIR/logs:/workspace/logs,$BASE_DIR/checkpoints:/workspace/checkpoints,$BUILD_DIR:/workspace/megatron-lm,$DATA_DIR:/workspace/data --no-container-mount-home bash -c "
  ls 
  cd /workspace/megatron-lm
  ./tests/functional_tests/test_scripts/gpt3/pretrain_gpt3_distributed_test.sh $DATA_PATH $CHECKPOINT_PATH $TENSORBOARD_DIR $USE_TE $TP_SIZE $PP_SIZE $NUM_NODES $MAX_STEPS $VP_SIZE $MBS $GBS"
