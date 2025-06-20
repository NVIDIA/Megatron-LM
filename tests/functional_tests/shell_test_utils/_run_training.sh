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
    "CHECKPOINT_SAVE_PATH"
    "CHECKPOINT_LOAD_PATH"
    "DATA_PATH"
    "RUN_NUMBER"
    "REPEAT"
)
for mandatory_var in "${MANDATORY_VARS[@]}"; do
    if [[ -z "${!mandatory_var}" ]]; then
        echo 'Providing $'$mandatory_var' is mandatory.'
        exit 1
    fi
done

# Envsubst model_params
cat $TRAINING_PARAMS_PATH | envsubst "$(env | cut -d= -f1 | sed -e 's/^/$/')" >$TRAINING_PARAMS_PATH.tmp
TRAINING_PARAMS_PATH="$TRAINING_PARAMS_PATH.tmp"

# Pull env vars to export
ENV_VARS=$(yq '... comments="" | .ENV_VARS | to_entries | .[] | [.key + "=" + .value] | join(" ")' "$TRAINING_PARAMS_PATH")
while IFS= read -r ARGUMENT; do
    KEY=$(echo $ARGUMENT | cut -f1 -d=)

    KEY_LENGTH=${#KEY}
    VALUE="${ARGUMENT:$KEY_LENGTH+1}"

    export "$KEY"="$VALUE"
    echo "$KEY=$VALUE"
done <<<"$ENV_VARS"

# Run before script
BEFORE_SCRIPT=$(cat "$TRAINING_PARAMS_PATH" | yq '.BEFORE_SCRIPT')
if [[ "$BEFORE_SCRIPT" != null ]]; then
    eval "$BEFORE_SCRIPT"
fi

# Exit earlier to leave time for properly saving checkpoint
if [[ "$IS_NEMO_TEST" == "true" ]]; then
    PARAMS=()
    # Store the output in a variable first
    TRAINING_PARAMS_STR=$(yq '... comments="" | .MODEL_ARGS | to_entries | .[] | with(select(.value == true); .value = "true") | .key + "=" + (select(.value != "") | .value | tostring)' "$TRAINING_PARAMS_PATH")
    # Build space-separated string while preserving quotes
    TRAINING_PARAMS_FROM_CONFIG=""
    while IFS= read -r line; do
        if [[ -n "$line" ]]; then
            # If value is "true", just use the key
            if [[ "$line" =~ =true$ ]]; then
                TRAINING_PARAMS_FROM_CONFIG+="${line%=true} "
            # If value contains spaces, wrap it in quotes
            elif [[ "$line" =~ .*=.*[[:space:]].* ]]; then
                key="${line%%=*}"
                value="${line#*=}"
                TRAINING_PARAMS_FROM_CONFIG+="$key=\"$value\" "
            else
                TRAINING_PARAMS_FROM_CONFIG+="$line "
            fi
        fi
    done <<<"$TRAINING_PARAMS_STR"
    # Remove trailing space
    TRAINING_PARAMS_FROM_CONFIG=${TRAINING_PARAMS_FROM_CONFIG% }
    # Split into array while preserving quotes
    eval "TRAINING_PARAMS_ARRAY=($TRAINING_PARAMS_FROM_CONFIG)"

else
    # If this is a second run (of checkpoint-resume), we might want to use a
    # different model configuration than during first time. So if key `MODEL_ARGS_2`
    # exists we use it, otherwise we use the same as for the first run.
    if [[ $RUN_NUMBER -eq 2 && $(yq 'has("MODEL_ARGS_2")' "$TRAINING_PARAMS_PATH") == true ]]; then
        export KEY="MODEL_ARGS_2"
    else
        export KEY="MODEL_ARGS"
    fi

    # Store the output in a variable first
    TRAINING_PARAMS_STR=$(yq '... comments="" | .[env(KEY)] | to_entries | .[] | with(select(.value == true); .value = "true") | .key + ": " + (select(.value != "") | .value | tostring)' "$TRAINING_PARAMS_PATH")
    # Build space-separated string while preserving quotes
    TRAINING_PARAMS_FROM_CONFIG=""
    while IFS= read -r line; do
        if [[ -n "$line" ]]; then

            key="${line%%:*}"
            value="${line#*: }"
            value="$(echo "$value" | xargs)" # trim whitespace
            # Case: true
            if [[ "$value" == "true" ]]; then
                TRAINING_PARAMS_FROM_CONFIG+="${key} "

            # Case: value is wrapped in ( )
            elif echo "$value" | grep -Eq '^\([^)]+\)$'; then
                TRAINING_PARAMS_FROM_CONFIG+="$key \"$value\" "

            # Case: value is wrapped in [ ]
            elif echo "$value" | grep -Eq '^\[[^]]+\]$'; then
                # Strip square brackets from value using sed
                value=$(echo "$value" | sed 's/^\[//;s/\]$//')
                TRAINING_PARAMS_FROM_CONFIG+="$key $value "

            # Case: contains spaces
            elif [[ "$value" == *" "* ]]; then
                TRAINING_PARAMS_FROM_CONFIG+="$key \"$value\" "
            # Case: default
            else
                TRAINING_PARAMS_FROM_CONFIG+="$key $value "
            fi
        fi
    done <<<"$TRAINING_PARAMS_STR"
    # Remove trailing space
    TRAINING_PARAMS_FROM_CONFIG=${TRAINING_PARAMS_FROM_CONFIG% }
    # Split into array while preserving quotes
    eval "TRAINING_PARAMS_ARRAY=($TRAINING_PARAMS_FROM_CONFIG)"
    PARAMS=(
        "--exit-duration-in-mins"
        $((($SLURM_JOB_END_TIME - $SLURM_JOB_START_TIME) / 60 - 15))
    )
fi

# Extract training params
PARAMS=("${PARAMS[@]}" "${TRAINING_PARAMS_ARRAY[@]}")

# Set PYTHONPATH
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"
export WANDB_API_KEY="${WANDB_API_KEY:-}"

######## Distributed training settings. ########
echo "------ARGUMENTS for SLURM ---"
MASTER_ADDR=${MASTER_ADDR:-localhost}
MASTER_PORT=${MASTER_PORT:-6000}
NUM_NODES=${NUM_NODES:-${SLURM_NNODES}}
GPUS_PER_NODE=${GPUS_PER_NODE:-8}
NODE_RANK=${SLURM_NODEID:-${SLURM_NODEID}}
LAST_RANK=7
export LOG_DIR=$OUTPUT_PATH/logs/$REPEAT
mkdir -p $LOG_DIR

DISTRIBUTED_ARGS=(
    --nproc_per_node $GPUS_PER_NODE
    --nnodes $NUM_NODES
    --master_addr $MASTER_ADDR
    --master_port $MASTER_PORT
    --node_rank $SLURM_NODEID
    --log-dir $LOG_DIR
    --tee "0:3,7:3"
    --redirects "3"
)

# Start training
if [[ "$IS_NEMO_TEST" == "true" ]]; then
    python -m torch.distributed.run ${DISTRIBUTED_ARGS[@]} --no-python $TRAINING_SCRIPT_PATH "${PARAMS[@]}" || EXIT_CODE=$?
else
    python -m torch.distributed.run ${DISTRIBUTED_ARGS[@]} $TRAINING_SCRIPT_PATH "${PARAMS[@]}" || EXIT_CODE=$?
fi

# Run after script
AFTER_SCRIPT=$(cat "$TRAINING_PARAMS_PATH" | yq '.AFTER_SCRIPT')
if [[ "$AFTER_SCRIPT" != null ]]; then
    eval "$AFTER_SCRIPT"
fi

if [[ ${RECORD_CHECKPOINTS} == "true" ]]; then
    echo "Suppressing errors during checkpoint recording."
    exit 0
fi

exit ${EXIT_CODE:-0}
