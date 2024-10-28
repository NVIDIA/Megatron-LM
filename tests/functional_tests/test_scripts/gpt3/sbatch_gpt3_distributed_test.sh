#!/bin/bash

# Parameters
#SBATCH --account=llmservice_dev_mcore
#SBATCH --job-name=llmservice_dev_mcore-ci:megatron-job
#SBATCH --nodes=1
#SBATCH --partition=luna

DATA_PATH=/workspace/data/gpt3_data/my-gpt3_00_text_document
CHECKPOINT_PATH=/workspace/checkpoints
TENSORBOARD_DIR=/workspace/tensorboard_logs
SCRIPTS_DIR=/workspace/debug

echo 'Running tests using $PYTORCH_IMAGE image'

srun --output $BASE_DIR/debug/slurm-%j.out --error $BASE_DIR/debug/slurm-%j.out --container-image $PYTORCH_IMAGE --container-mounts $BASE_DIR/tensorboard_logs:/workspace/tensorboard_logs,$BASE_DIR/debug:/workspace/debug,$BASE_DIR/checkpoints:/workspace/checkpoints,$BUILD_DIR:/workspace/megatron-lm,$DATA_DIR:/workspace/data --no-container-mount-home bash -c "
  ls 
  cd /workspace/megatron-lm
  ./tests/functional_tests/test_scripts/gpt3/pretrain_gpt3_distributed_test.sh DATA_PATH=$DATA_PATH CHECKPOINT_PATH=$CHECKPOINT_PATH TENSORBOARD_DIR=$TENSORBOARD_DIR SCRIPTS_DIR=$SCRIPTS_DIR USE_TE=$USE_TE TP_SIZE=$TP_SIZE PP_SIZE=$PP_SIZE VP_SIZE=$VP_SIZE NUM_NODES=$NUM_NODES MAX_STEPS=$MAX_STEPS USE_CORE=$USE_CORE MBS=$MBS GBS=$GBS MOE_GROUPED_GEMM=$MOE_GROUPED_GEMM ADDITIONAL_PARAMS=\"$ADDITIONAL_PARAMS\""
