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
    "CHECKPOINT_PATH"
    "DATA_PATH"
    "DATA_CACHE_PATH"
)
for mandatory_var in "${MANDATORY_VARS[@]}"; do
    if [[ -z "${!mandatory_var}" ]]; then
        echo 'Providing $'$mandatory_var' is mandatory.'
        exit 1
    fi
done

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)
ROOT_DIR=$(realpath $SCRIPT_DIR/../../../)

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
        rm -rf $CHECKPOINT_PATH/*
    fi

    # Training
    export RUN_NUMBER=1
    bash $ROOT_DIR/tests/functional_tests/shell_test_utils/_run_training.sh

    # Maybe checkpoint resume training
    if [[ "$TEST_TYPE" == "ckpt-resume" ]]; then
        if [[ ${SLURM_PROCID} -eq 0 ]]; then
            rm -rf "$CHECKPOINT_PATH/iter_0000$TRAIN_ITERS"
            echo $((TRAIN_ITERS / 2)) >$CHECKPOINT_PATH/latest_checkpointed_iteration.txt
        fi

        export RUN_NUMBER=2
        bash $ROOT_DIR/tests/functional_tests/shell_test_utils/_run_training.sh
    fi

    if [[ ${SLURM_PROCID} -gt 0 ]]; then
        continue
    fi

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

    # Maybe run tests
    if [[ ${SKIP_PYTEST:-0} != 1 ]]; then
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

        if [[ "$TEST_TYPE" == "ckpt-resume" ]]; then
            echo "Running pytest 1st vs 2nd run comparison"
            pytest -s -o log_cli=true --log-cli-level=info $ROOT_DIR/tests/functional_tests/python_test_utils/test_resume_checkpoint_pipeline.py \
                --tensorboard-path $TENSORBOARD_PATH \
                --train-iters $TRAIN_ITERS \
                --model-config-path ${TRAINING_PARAMS_PATH} \
                $ALLOW_NONDETERMINISTIC_ALGO_ARG

        else
            echo "Test type $TEST_TYPE not yet implemented."
        fi
    fi
done
