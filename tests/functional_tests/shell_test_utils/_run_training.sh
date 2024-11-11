#!/bin/bash

# This script can be used for model onboarding and testing.

# For onboarding, it extract scalars from Tensorboard logs only.
# For testing, it compares extracted Tensorboard scalars against
# a set of `GOLDEN_VALUES`.

set -euxo pipefail

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
    "TRAINING_SCRIPT_PATH"
    "TRAINING_PARAMS_PATH"
    "OUTPUT_PATH"
    "TENSORBOARD_PATH"
    "CHECKPOINT_PATH"
    "DATA_PATH"
    "RUN_NUMBER"
)
for mandatory_var in "${MANDATORY_VARS[@]}"; do
    if [[ -z "${!mandatory_var}" ]]; then
        echo 'Providing $'$mandatory_var' is mandatory.'
        exit 1
    fi
done

cp $TRAINING_PARAMS_PATH "$TRAINING_PARAMS_PATH.${SLURM_PROCID}"
TRAINING_PARAMS_PATH="$TRAINING_PARAMS_PATH.${SLURM_PROCID}"

# Envsubst model_params
cat $TRAINING_PARAMS_PATH | envsubst "$(env | cut -d= -f1 | sed -e 's/^/$/')" >$TRAINING_PARAMS_PATH.tmp
mv $TRAINING_PARAMS_PATH.tmp "$TRAINING_PARAMS_PATH"

# Pull env vars to export
ENV_VARS=$(yq '... comments="" | .ENV_VARS | to_entries | .[] | [.key + "=" + .value] | join(" ")' "$TRAINING_PARAMS_PATH")
while IFS= read -r ARGUMENT; do
    KEY=$(echo $ARGUMENT | cut -f1 -d=)

    KEY_LENGTH=${#KEY}
    VALUE="${ARGUMENT:$KEY_LENGTH+1}"

    export "$KEY"="$VALUE"
    echo "$KEY=$VALUE"
done <<< "$ENV_VARS"

# Run before script
SCRIPT=$(cat "$TRAINING_PARAMS_PATH" | yq '.BEFORE_SCRIPT')
if [[ "$SCRIPT" != null ]]; then
    eval "$SCRIPT"
fi;

# Exit earlier to leave time for properly saving checkpoint
if [[ $(echo "$TRAINING_SCRIPT_PATH" | tr '[:upper:]' '[:lower:]') == *nemo* ]]; then
    PARAMS=""
    TRAINING_PARAMS_FROM_CONFIG=$(yq '... comments="" | .MODEL_ARGS | to_entries | .[] | with(select(.value == "true"); .value = "") | [.key + "=" + .value] | join("")' "$TRAINING_PARAMS_PATH" | tr '\n' ' ')

else
    # If this is a second run (of checkpoint-resume), we might want to use a 
    # different model configuration than during first time. So if key `MODEL_ARGS_2`
    # exists we use it, otherwise we use the same as for the first run.
    if [[ $RUN_NUMBER -eq 2 && $(yq 'has("MODEL_ARGS_2")' "$TRAINING_PARAMS_PATH") == true ]]; then
        export KEY="MODEL_ARGS_2"
    else
        export  KEY="MODEL_ARGS"
    fi

    TRAINING_PARAMS_FROM_CONFIG=$(yq '... comments="" | .[env(KEY)] | to_entries | .[] | with(select(.value == "true"); .value = "") | [.key + " " + .value] | join("")' "$TRAINING_PARAMS_PATH" | tr '\n' ' ')
    PARAMS="--exit-duration-in-mins $((($SLURM_JOB_END_TIME - $SLURM_JOB_START_TIME) / 60 - 15))"
fi

# Extract training params
PARAMS="$PARAMS $TRAINING_PARAMS_FROM_CONFIG"

# Set PYTHONPATH
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"
export WANDB_API_KEY="${WANDB_API_KEY:-}"

# Start training
python $TRAINING_SCRIPT_PATH $PARAMS

