#!/bin/bash

#######################################################################################
#
# Script for capturing a reference model.
#
# It will train a model until a target iteration was hit.
#
#
########################################################################################

set -exo pipefail

echo "------ARGUMENTS LIST --------"
for ARGUMENT in "$@"; do
    KEY=$(echo $ARGUMENT | cut -f1 -d=)

    KEY_LENGTH=${#KEY}
    VALUE="${ARGUMENT:$KEY_LENGTH+1}"

    export "$KEY"="$VALUE"
    echo "$KEY=$VALUE"
done
echo "---------------------------------"

# Check that mandatory vars are set
MANDATORY_VARS=(
    "MODEL"
    "VARIANT"
    "TRAINING_SCRIPT_PATH"
    "OUTPUT_PATH"
    "IMAGE_TAG"
    "NODES"
    "PPP"
    "PARTITION"
    "ITERATIONS"
    "WANDB_API_KEY"
    "CLUSTER"
    "DATASET"
    "WANDB_EXPERIMENT"
    "GPUS_PER_NODE"
)
for mandatory_var in "${MANDATORY_VARS[@]}"; do
    if [[ -z "${!mandatory_var}" ]]; then
        echo 'Providing $'$mandatory_var' is mandatory.'
        exit 1
    fi
done

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
ROOT_DIR=$(realpath $SCRIPT_DIR/../../../)

# Fetch dataset base path via JET and refresh DATA_BELDN
DATA_PATH=$(jet -c -tf plain -th artifacts registry list -c storages.$CLUSTER.identifier -f "key == '$DATASET'")
DATA_BLEND=$(eval echo "$DATA_BLEND")

########################################################################################
# Dont change below
########################################################################################

SLURM_LOGS=$OUTPUT_PATH/slurm_logs/
mkdir -p $SLURM_LOGS

# Container settings
ARGUMENTS=(
    "TRAINING_SCRIPT_PATH=${TRAINING_SCRIPT_PATH}"
    "TEST_CASE_PATH=./tests/functional_tests/test_cases/$MODEL/$VARIANT"
    "OUTPUT_PATH=${OUTPUT_PATH}"
    "TENSORBOARD_PATH=${OUTPUT_PATH}/tensorboard"
    "CHECKPOINT_PATH=${OUTPUT_PATH}/checkpoints"
    "DATA_PATH=${DATA_PATH}"
    "DATA_CACHE_PATH=${OUTPUT_PATH}/data-cache"
    "WANDB_API_KEY=${WANDB_API_KEY}"
    "WANDB_EXPERIMENT=${WANDB_EXPERIMENT}"
    "DATA_BLEND=\"${DATA_BLEND}\""
)

if [[ -n $LOAD_PATH ]]; then
    ARGUMENTS+=("LOAD_PATH=${LOAD_PATH}")
fi

echo ${ARGUMENTS[@]}

while : 
do

if [[ $(cat "${OUTPUT_PATH}/checkpoints/latest_checkpointed_iteration.txt" || echo 0) -ge $ITERATIONS ]]; then
    break
fi

# Fire of sbatch
echo '#!/bin/bash' > sbatch.sh

if [[ $GPUS_PER_NODE != null ]]; then
    echo '#SBATCH --gres=gpu:8' >> sbatch.sh
fi
echo "#SBATCH --nodes=$NODES
#SBATCH --account $PPP
#SBATCH --partition $PARTITION
#SBATCH --ntasks-per-node=1
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
    --container-image='gitlab-master.nvidia.com/adlr/megatron-lm/mcore_ci:$IMAGE_TAG' \
    --container-mounts='${DATA_PATH}:${DATA_PATH},${OUTPUT_PATH}:${OUTPUT_PATH}' \
    --container-workdir=/workspace/megatron-lm \
    bash ./tests/functional_tests/shell_test_utils/run_ci_test.sh ${ARGUMENTS[@]}>>'$SLURM_LOGS/\${SLURM_JOB_ID}.log' 2>&1" >> sbatch.sh

set +e
sbatch -W sbatch.sh
set -e
done

# Write golden values into repo if this run should become a reference
cp $OUTPUT_PATH/golden_values.json > ./golden_values.json
