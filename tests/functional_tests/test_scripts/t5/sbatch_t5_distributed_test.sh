#!/bin/bash

# Parameters
#SBATCH --account=adlr_nlp_llmnext
#SBATCH --job-name=adlr_nlp_llmnext-ci:megatron-job
#SBATCH --nodes=1
#SBATCH --partition=luna

DATA_PATH=/workspace/data/gpt3_data/my-gpt3_00_text_document
CHECKPOINT_PATH=/workspace/checkpoints
TENSORBOARD_DIR=/workspace/logs

if [[ -n $MBS ]]; then MBS=4; fi
if [[ -n $GBS ]]; then GBS=32; fi

if [[ -n $VP_SIZE ]]; then VP_SIZE="" ; fi

echo 'Running tests using $PYTORCH_IMAGE image'

srun --output $BASE_DIR/results/slurm-%j.out --error $BASE_DIR/results/slurm-%j.out --container-image $PYTORCH_IMAGE --container-mounts $BASE_DIR/logs:/workspace/logs,$BASE_DIR/checkpoints:/workspace/checkpoints,$BUILD_DIR:/workspace/megatron-lm,$DATA_DIR:/workspace/data --no-container-mount-home bash -c "
  ls 
  cd /workspace/megatron-lm
  ./tests/functional_tests/test_scripts/gpt3/pretrain_gpt3_distributed_test.sh $DATA_PATH $CHECKPOINT_PATH $TENSORBOARD_DIR $USE_TE $TP_SIZE $PP_SIZE $NUM_NODES $MAX_STEPS $USE_CORE \"$VP_SIZE\" \"$MBS\" \"$GBS\" \"$ADDITIONAL_PARAMS\""
