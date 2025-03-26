#!/bin/bash

set -exo pipefail

echo "------ARGUMENTS LIST --------"
for ARGUMENT in "$@"; do
    echo $ARGUMENT
    KEY=$(echo $ARGUMENT | cut -f1 -d=)

    KEY_LENGTH=${#KEY}
    VALUE=$(eval echo ${ARGUMENT:$KEY_LENGTH+1})
    export "$KEY"="$VALUE"
    echo "$KEY=$VALUE"
done
echo "---------------------------------"

# Check that mandatory vars are set
MANDATORY_VARS=(
    "TRAINING_SCRIPT_PATH"
    "TRAINING_PARAMS_PATH"
    "GOLDEN_VALUES_PATH"
    "OUTPUT_PATH"
    "TENSORBOARD_PATH"
    "CHECKPOINT_SAVE_PATH"
    "CHECKPOINT_LOAD_PATH"
    "DATA_PATH"
    "DATA_CACHE_PATH"
)
for mandatory_var in "${MANDATORY_VARS[@]}"; do
    if [[ -z "${!mandatory_var}" ]]; then
        echo 'Providing $'$mandatory_var' is mandatory.'
        exit 1
    fi
done

RECORD_CHECKPOINTS=${RECORD_CHECKPOINTS:-"false"}

TEST_TYPES=("regular" "ckpt-resume" "frozen-resume" "frozen-start" "release")

if [[ "$TEST_TYPE" == "release" ]]; then
    export ONE_LOGGER_JOB_CATEGORY=production
else
    export ONE_LOGGER_JOB_CATEGORY=test
fi

mkdir -p $CHECKPOINT_SAVE_PATH
mkdir -p $CHECKPOINT_LOAD_PATH
_CHECKPOINT_LOAD_PATH=$CHECKPOINT_LOAD_PATH
_CHECKPOINT_SAVE_PATH=$CHECKPOINT_SAVE_PATH

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)
ROOT_DIR=$(realpath $SCRIPT_DIR/../../../)

IS_NEMO_TEST=$([[ $(echo "$TRAINING_SCRIPT_PATH" | tr '[:upper:]' '[:lower:]') == *nemo* ]] && echo "true" || echo "false")
export IS_NEMO_TEST

# Adjust model_config for lightweight mode
if [[ "$ENABLE_LIGHTWEIGHT_MODE" == "true" && "$IS_NEMO_TEST" == "true" ]]; then
    yq -i '.MODEL_ARGS."trainer.max_steps" = 2' $TRAINING_PARAMS_PATH
    N_REPEAT=1
elif [[ "$ENABLE_LIGHTWEIGHT_MODE" == "true" && "$IS_NEMO_TEST" == "false" ]]; then

    yq -i '.ENV_VARS."SKIP_PYTEST" = 1' $TRAINING_PARAMS_PATH
    yq -i '.MODEL_ARGS."--train-iters" = 2' $TRAINING_PARAMS_PATH
    yq -i '.MODEL_ARGS."--save-interval" = 1' $TRAINING_PARAMS_PATH
    N_REPEAT=1
fi

# Extract settings from params file
TEST_TYPE=$(cat $TRAINING_PARAMS_PATH |
    yq '.TEST_TYPE')
NVTE_ALLOW_NONDETERMINISTIC_ALGO=$(cat $TRAINING_PARAMS_PATH |
    yq '.ENV_VARS.NVTE_ALLOW_NONDETERMINISTIC_ALGO')
SKIP_PYTEST=$(cat $TRAINING_PARAMS_PATH |
    yq '.ENV_VARS.SKIP_PYTEST')
TRAIN_ITERS=$(cat $TRAINING_PARAMS_PATH |
    yq '.MODEL_ARGS."--train-iters" // "100"')

for i in $(seq 1 $N_REPEAT); do
    if [[ $i -gt 1 ]]; then
        rm -rf $CHECKPOINT_SAVE_PATH/*
        rm -rf /tmp/checkpoints/*
        rm -rf $TENSORBOARD_PATH/*
    fi

    # First run never loads from a checkpoint
    export RUN_NUMBER=1
    export REPEAT=$i
    export CHECKPOINT_SAVE_PATH=$_CHECKPOINT_SAVE_PATH

    if [[ "$TEST_TYPE" = "frozen-start" ]]; then
        export CHECKPOINT_LOAD_PATH=$_CHECKPOINT_LOAD_PATH
    else
        export CHECKPOINT_LOAD_PATH=/tmp/checkpoints/
    fi

    if [[ "$TEST_TYPE" = "release" ]]; then
        export CHECKPOINT_LOAD_PATH=$_CHECKPOINT_LOAD_PATH
        export CHECKPOINT_SAVE_PATH=$_CHECKPOINT_SAVE_PATH
    fi

    bash $ROOT_DIR/tests/functional_tests/shell_test_utils/_run_training.sh

    if [[ "$TEST_TYPE" = "frozen-resume" && -z "$(ls -A "$_CHECKPOINT_LOAD_PATH" 2>/dev/null)" ]]; then
        echo "No frozen checkpoint found. Will skip second run."

        export CHECKPOINT_SAVE_PATH=$_CHECKPOINT_SAVE_PATH
        rm -rf "$CHECKPOINT_SAVE_PATH/iter_0000$TRAIN_ITERS"
        echo $((TRAIN_ITERS / 2)) >$CHECKPOINT_SAVE_PATH/latest_checkpointed_iteration.txt
        break
    fi

    if [[ "$TEST_TYPE" == "ckpt-resume" ]]; then
        export CHECKPOINT_LOAD_PATH=$CHECKPOINT_SAVE_PATH

        rm -rf "$CHECKPOINT_LOAD_PATH/iter_0000$TRAIN_ITERS"
        echo $((TRAIN_ITERS / 2)) >$CHECKPOINT_LOAD_PATH/latest_checkpointed_iteration.txt

        export RUN_NUMBER=2
        bash $ROOT_DIR/tests/functional_tests/shell_test_utils/_run_training.sh
    fi

    if [[ "$TEST_TYPE" == "frozen-resume" ]]; then

        # Checkpoint-resume tests load from prev run
        export CHECKPOINT_LOAD_PATH=$_CHECKPOINT_LOAD_PATH
        export CHECKPOINT_SAVE_PATH=/tmp/checkpoints/

        export RUN_NUMBER=2
        bash $ROOT_DIR/tests/functional_tests/shell_test_utils/_run_training.sh

        export CHECKPOINT_SAVE_PATH=$_CHECKPOINT_SAVE_PATH
        rm -rf "$CHECKPOINT_SAVE_PATH/iter_0000$TRAIN_ITERS"
        echo $((TRAIN_ITERS / 2)) >$CHECKPOINT_SAVE_PATH/latest_checkpointed_iteration.txt
    fi

    if [[ "$TEST_TYPE" == "release" ]]; then
        SKIP_PYTEST=0
        TRAIN_ITERS=10000000
    fi

    if [[ ${RECORD_CHECKPOINTS} == "true" ]]; then
        echo "Skipping Pytest during checkpoint recording."
        SKIP_PYTEST=1
    fi

    # Maybe run tests
    if [[ ${SKIP_PYTEST:-0} != 1 ]]; then
        # Save run results
        export PYTHONPATH=$ROOT_DIR
        if [[ "$TEST_TYPE" == "release" ]]; then
            EXTRACT_ARGS=("--is-convergence-test")
        else
            EXTRACT_ARGS=("--is-normal-test")
        fi

        python3 $ROOT_DIR/tests/functional_tests/python_test_utils/get_test_results_from_tensorboard_logs.py \
            --logs-dir $TENSORBOARD_PATH \
            --train-iters $TRAIN_ITERS \
            --output-path ${OUTPUT_PATH}/$(basename $GOLDEN_VALUES_PATH) \
            "${EXTRACT_ARGS[@]}"

        export NVTE_ALLOW_NONDETERMINISTIC_ALGO
        if [[ "${NVTE_ALLOW_NONDETERMINISTIC_ALGO}" == "1" ]]; then
            ALLOW_NONDETERMINISTIC_ALGO_ARG="--allow-nondeterministic-algo"
        fi

        echo "Running pytest checks against golden values"

        pytest -s -o log_cli=true --log-cli-level=info $ROOT_DIR/tests/functional_tests/python_test_utils/test_regular_pipeline.py \
            --golden-values-path $GOLDEN_VALUES_PATH \
            --tensorboard-path $TENSORBOARD_PATH \
            --model-config-path ${TRAINING_PARAMS_PATH} \
            $ALLOW_NONDETERMINISTIC_ALGO_ARG

        if [[ "$TEST_TYPE" == "ckpt-resume" || "$TEST_TYPE" == "frozen-resume" ]]; then
            echo "Running pytest 1st vs 2nd run comparison"
            pytest -s -o log_cli=true --log-cli-level=info $ROOT_DIR/tests/functional_tests/python_test_utils/test_resume_checkpoint_pipeline.py \
                --tensorboard-path $TENSORBOARD_PATH \
                --train-iters $TRAIN_ITERS \
                --model-config-path ${TRAINING_PARAMS_PATH} \
                $ALLOW_NONDETERMINISTIC_ALGO_ARG
        fi

        if [[ ! " ${TEST_TYPES[*]} " =~ " ${TEST_TYPE} " ]]; then
            echo "Test type $TEST_TYPE not yet implemented."
        fi
    fi
done
