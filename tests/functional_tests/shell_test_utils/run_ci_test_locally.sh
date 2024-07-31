#!/bin/bash

#######################################################################################
#
# Script for capturing a reference model.
#
# It will train a model until a target iteration was hit.
#
#
########################################################################################

set -euxo pipefail

# Check that mandatory vars are set
MANDATORY_VARS=(
    "MODEL"
    "MCORE_RELEASE_NUM"
    "TRAINING_SCRIPT_PATH"
    "TRAINING_PARAMS_PATH"
    "OUTPUT_PATH"
    "IMAGE_TAG"
    "NODES"
    "PPP"
    "PARTITION"
    "ITERATIONS"
    "GITLAB_TOKEN"
    "WANDB_API_KEY"
    "CLUSTER"
    "DATASET"
)
for mandatory_var in "${MANDATORY_VARS[@]}"; do
    if [[ -z "${!mandatory_var}" ]]; then
        echo 'Providing $'$mandatory_var' is mandatory.'
        exit 1
    fi
done

DATA_PATH=$(jet \
    -c \
    -tf plain \
    -th \
    artifacts \
        registry \
            list \
            -c storages.$CLUSTER.identifier \
            -f 'key == "'$DATASET'"'
)

########################################################################################
# Dont change below
########################################################################################

# Container settings
IMAGE="gitlab-master.nvidia.com/adlr/megatron-lm/mcore_ci:$IMAGE_TAG"
MOUNTS="${DATA_PATH}:${DATA_PATH},${OUTPUT_PATH}:${OUTPUT_PATH}"
MODEL_TYPE=$(basename $TRAINING_SCRIPT_PATH | awk -F'[_.]' '{print $2}')
GOLDEN_VALUES_PATH=${OUTPUT_PATH}/$MODEL.json
GOLDEN_VALUES_PATH_IN_REPO=./tests/functional_tests/test_results/$MODEL_TYPE/$MODEL-${MCORE_RELEASE_NUM}.json
ARGUMENTS=(
    "TRAINING_SCRIPT_PATH=${TRAINING_SCRIPT_PATH}"
    "TRAINING_PARAMS_PATH=${TRAINING_PARAMS_PATH}"
    "DATA_PATH=${DATA_PATH}"
    "DATA_CACHE_PATH=${OUTPUT_PATH}/data-cache"
    "OUTPUT_PATH=${OUTPUT_PATH}"
    "TENSORBOARD_PATH=${OUTPUT_PATH}/tensorboard"
    "CHECKPOINT_PATH=${OUTPUT_PATH}/checkpoints"
    "WANDB_API_KEY=${WANDB_API_KEY}"
    "GOLDEN_VALUES_PATH=${GOLDEN_VALUES_PATH}/$MODEL_TYPE/$MODEL.json"
    "MCORE_RELEASE_NUM=${MCORE_RELEASE_NUM}"
)
SLURM_LOGS=$OUTPUT_PATH/slurm_logs/
mkdir -p $SLURM_LOGS

while : 
do
ACTUAL_ITERATIONS=$(cat "$OUTPUT_PATH/checkpoints/latest_checkpointed_iteration.txt" || echo 0)
if [[ $ACTUAL_ITERATIONS -gt $ITERATIONS ]]; then
    break
fi

# Fire of sbatch
set +e
sbatch -W <<EOF
#!/bin/bash

#SBATCH --nodes=$NODES
#SBATCH --account $PPP
#SBATCH --partition $PARTITION
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --time "04:00:00"
#SBATCH --job-name=$PPP:mcore:release:$MODEL
#SBATCH --dependency=singleton
#SBATCH --output=/dev/null 
#SBATCH --error=/dev/null
#SBATCH --exclusive

# Prepare SLURM job
echo "SLURM_JOB_ID=\$SLURM_JOB_ID" > "$SLURM_LOGS/\${SLURM_JOB_ID}.log"

srun \
    --ntasks-per-node=1 \
    --container-image=${IMAGE} \
    --container-mounts=${MOUNTS} \
    --container-workdir=/workspace/megatron-lm \
    bash ./tests/functional_tests/shell_test_utils/run_ci_test.sh ${ARGUMENTS[@]} >>"$SLURM_LOGS/\${SLURM_JOB_ID}.log" 2>&1
EOF
set -e
done

# Write golden values into repo if this run should become a reference
cp $GOLDEN_VALUES_PATH > $GOLDEN_VALUES_PATH_IN_REPO

# Finally upload everything to JET
jet artifacts registry add \
    --token $GITLAB_TOKEN \
    --source-path $OUTPUT_PATH \
    --automerge \
    --reference-storage $CLUSTER:$OUTPUT_PATH \
    "unverified/model/mcore-$MCORE_RELEASE_NUM/$MODEL" 
