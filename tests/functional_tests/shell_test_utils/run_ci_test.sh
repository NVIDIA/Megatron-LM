#!/bin/bash

set -exo pipefail

# Increase soft limit for number of open files to match hard limit
ulimit -Sn $(ulimit -Hn)

# Increase soft limit for number of processes to match hard limit
ulimit -Su $(ulimit -Hu)

set +x
for ARGUMENT in "$@"; do
    # Split on first = only, preserving any subsequent = signs in the value
    KEY="${ARGUMENT%%=*}"
    VALUE="${ARGUMENT#*=}"

    # Remove any surrounding quotes from the value if they exist
    VALUE="${VALUE%\"}"
    VALUE="${VALUE#\"}"
    VALUE="${VALUE%\'}"
    VALUE="${VALUE#\'}"

    # Properly quote the value to preserve spaces and special characters
    export "$KEY"="$(eval echo $VALUE)"
    echo "$KEY=$VALUE"
done
set -x

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
    "ENABLE_LIGHTWEIGHT_MODE"
)
for mandatory_var in "${MANDATORY_VARS[@]}"; do
    if [[ -z "${!mandatory_var}" ]]; then
        echo 'Providing $'$mandatory_var' is mandatory.'
        exit 1
    fi
done

set -exo pipefail

# Extract settings from params file
TEST_TYPE=$(cat $TRAINING_PARAMS_PATH |
    /usr/local/bin/yq '.TEST_TYPE')
ENABLE_LIGHTWEIGHT_MODE=$(cat $TRAINING_PARAMS_PATH |
    /usr/local/bin/yq '.ENV_VARS.ENABLE_LIGHTWEIGHT_MODE // "false"')
MODE=$(cat $TRAINING_PARAMS_PATH |
    /usr/local/bin/yq '.MODE // "pretraining"')

MODES=("pretraining" "inference")
TEST_TYPES=("regular" "ckpt-resume" "frozen-resume" "frozen-start" "checkpoint-consistency" "release")

if [[ "$TEST_TYPE" == "release" ]]; then
    export ONE_LOGGER_JOB_CATEGORY=production
else
    export ONE_LOGGER_JOB_CATEGORY=test
fi

mkdir -p $CHECKPOINT_SAVE_PATH
mkdir -p $CHECKPOINT_LOAD_PATH || true
_CHECKPOINT_LOAD_PATH=$CHECKPOINT_LOAD_PATH
_CHECKPOINT_SAVE_PATH=$CHECKPOINT_SAVE_PATH

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)
ROOT_DIR=$(realpath $SCRIPT_DIR/../../../)

IS_NEMO_TEST=$([[ $(echo "$TRAINING_SCRIPT_PATH" | tr '[:upper:]' '[:lower:]') == *nemo* ]] && echo "true" || echo "false")
export IS_NEMO_TEST

# Adjust model_config for lightweight mode
if [[ "$MODE" == "pretraining" && "$TEST_TYPE" != "release" ]]; then
    if [[ "$ENABLE_LIGHTWEIGHT_MODE" == "true" && "$IS_NEMO_TEST" == "true" ]]; then
        /usr/local/bin/yq -i '.MODEL_ARGS."trainer.max_steps" = 2' $TRAINING_PARAMS_PATH
        TRAIN_ITERS=$(cat $TRAINING_PARAMS_PATH |
            /usr/local/bin/yq '.MODEL_ARGS."trainer.max_steps // "100"')

        N_REPEAT=1

    elif [[ "$ENABLE_LIGHTWEIGHT_MODE" == "true" && "$IS_NEMO_TEST" == "false" ]]; then
        /usr/local/bin/yq -i '.ENV_VARS."SKIP_PYTEST" = 1' $TRAINING_PARAMS_PATH
        /usr/local/bin/yq -i '.MODEL_ARGS."--exit-interval" = 4' $TRAINING_PARAMS_PATH
        TRAIN_ITERS=$(cat $TRAINING_PARAMS_PATH |
            /usr/local/bin/yq '.MODEL_ARGS."--exit-interval" // "100"')
        N_REPEAT=1

        if [[ "$TEST_TYPE" == "ckpt-resume" || "$TEST_TYPE" == "frozen-resume" ]]; then
            /usr/local/bin/yq -i '.MODEL_ARGS."--save-interval" = 2' $TRAINING_PARAMS_PATH
        fi

    elif [[ "$ENABLE_LIGHTWEIGHT_MODE" == "false" && "$IS_NEMO_TEST" == "true" ]]; then
        TRAIN_ITERS=$(cat $TRAINING_PARAMS_PATH |
            /usr/local/bin/yq '.MODEL_ARGS."trainer.max_steps" // "100"')

    elif [[ "$ENABLE_LIGHTWEIGHT_MODE" == "false" && "$IS_NEMO_TEST" == "false" ]]; then
        /usr/local/bin/yq -i '.MODEL_ARGS."--exit-interval" = .MODEL_ARGS."--train-iters"' $TRAINING_PARAMS_PATH
        TRAIN_ITERS=$(cat $TRAINING_PARAMS_PATH |
            /usr/local/bin/yq '.MODEL_ARGS."--exit-interval" // "100"')
    fi
elif [[ "$MODE" == "inference" && "$TEST_TYPE" != "release" ]]; then
    if [[ "$ENABLE_LIGHTWEIGHT_MODE" == "true" && "$IS_NEMO_TEST" == "false" ]]; then
        /usr/local/bin/yq -i '.ENV_VARS."SKIP_PYTEST" = 1' $TRAINING_PARAMS_PATH
    fi
fi

if [[ "$MODE" == "pretraining" && "$TEST_TYPE" = "release" ]]; then
    TRAIN_ITERS=$(cat $TRAINING_PARAMS_PATH |
        /usr/local/bin/yq '.MODEL_ARGS."--exit-interval" // "100"')
fi

# Extract settings from params file
NVTE_ALLOW_NONDETERMINISTIC_ALGO=$(cat $TRAINING_PARAMS_PATH |
    /usr/local/bin/yq '.ENV_VARS.NVTE_ALLOW_NONDETERMINISTIC_ALGO')
NON_DETERMINSTIC_RESULTS=$(cat $TRAINING_PARAMS_PATH |
    /usr/local/bin/yq '.ENV_VARS.NON_DETERMINSTIC_RESULTS // "0"')
SKIP_PYTEST=$(cat $TRAINING_PARAMS_PATH |
    /usr/local/bin/yq '.ENV_VARS.SKIP_PYTEST')

export RECORD_CHECKPOINTS=${RECORD_CHECKPOINTS:-"false"}

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
    export TRAINING_EXIT_CODE=0
    declare -a ITER_CHECKPOINT_DIRS=()  # for the grad-test check if we're doing it

    if [[ "$TEST_TYPE" = "frozen-start" || "$TEST_TYPE" = "checkpoint-consistency" ]]; then
        export CHECKPOINT_LOAD_PATH=$_CHECKPOINT_LOAD_PATH
    else
        export CHECKPOINT_LOAD_PATH=/tmp/checkpoints/
    fi

    if [[ "$TEST_TYPE" = "release" ]]; then
        export CHECKPOINT_LOAD_PATH=$_CHECKPOINT_LOAD_PATH
        export CHECKPOINT_SAVE_PATH=$_CHECKPOINT_SAVE_PATH
    fi

    if [[ "$TEST_TYPE" = "checkpoint-consistency" ]]; then
        ## Loop over the list of model configs in the params file and run each one in sequence, collecting
        #  the checkpoints. Assume that we do a single step for this test.

        # 1. Loop over the runs in the params file
        # Get all MODEL_ARGS keys from the params file
        mapfile -t MODEL_ARGS_KEYS < <(/usr/local/bin/yq 'keys | .[] | select(test("^MODEL_ARGS(_\\d+)?$"))' "$TRAINING_PARAMS_PATH")
        

        # For-loop over the keys
        for KEY in "${MODEL_ARGS_KEYS[@]}"; do
            [[ -z "$KEY" ]] && continue

            if [[ "$KEY" =~ ^MODEL_ARGS_([0-9]+)$ ]]; then
                export LOOP_RN="${BASH_REMATCH[1]}"
            elif [[ "$KEY" == "MODEL_ARGS" ]]; then
                export LOOP_RN=1
            else
                echo "Unexpected KEY: $KEY" >&2; exit 1
            fi
            export RUN_NUMBER=$LOOP_RN

            # Get the number of GPUs from this run. Do not export this so it clashes with the other runs.
            N_GPUS=$(cat $TRAINING_PARAMS_PATH |
                /usr/local/bin/yq '.MODEL_ENV_VARS.'$KEY'.GPUS_PER_NODE')
            echo "Running $KEY with RUN_NUMBER=$RUN_NUMBER and GPUS_PER_NODE=$N_GPUS"
            
            ITER_CHECKPOINT_SAVE_PATH="$_CHECKPOINT_SAVE_PATH/repeat_${REPEAT}_key_${KEY}"
            mkdir -p $ITER_CHECKPOINT_SAVE_PATH

            # Save a checkpoint for this run
            GPUS_PER_NODE=$N_GPUS KEY=$KEY CHECKPOINT_SAVE_PATH=$ITER_CHECKPOINT_SAVE_PATH \
            bash $ROOT_DIR/tests/functional_tests/shell_test_utils/_run_training.sh || TRAINING_EXIT_CODE=$?

            # TODO find out the final iter and put that at the end rather than hardcoding 1
            ITER_CHECKPOINT_DIRS+=("$ITER_CHECKPOINT_SAVE_PATH/iter_0000001")
        done
    else
        # The standard single-run test that otherwise runs
        bash $ROOT_DIR/tests/functional_tests/shell_test_utils/_run_training.sh || TRAINING_EXIT_CODE=$?
    fi

    if [[ "$TEST_TYPE" = "frozen-resume" && -z "$(ls -A "$_CHECKPOINT_LOAD_PATH" 2>/dev/null)" ]]; then
        echo "No frozen checkpoint found. Will skip second run."

        export CHECKPOINT_SAVE_PATH=$_CHECKPOINT_SAVE_PATH
        rm -rf "$CHECKPOINT_SAVE_PATH/iter_0000$TRAIN_ITERS"
        echo $((TRAIN_ITERS / 2)) >$CHECKPOINT_SAVE_PATH/latest_checkpointed_iteration.txt
        break
    fi

    if [[ "$TEST_TYPE" == "ckpt-resume" && "$TRAINING_EXIT_CODE" -eq 0 ]]; then
        export CHECKPOINT_LOAD_PATH=$CHECKPOINT_SAVE_PATH

        rm -rf "$CHECKPOINT_LOAD_PATH/iter_$(printf "%07d\n" "$TRAIN_ITERS")"
        echo $((TRAIN_ITERS / 2)) >$CHECKPOINT_LOAD_PATH/latest_checkpointed_iteration.txt

        export RUN_NUMBER=2
        bash $ROOT_DIR/tests/functional_tests/shell_test_utils/_run_training.sh || TRAINING_EXIT_CODE=$?
    fi

    if [[ "$TEST_TYPE" == "frozen-resume" && "$TRAINING_EXIT_CODE" -eq 0 ]]; then

        # Checkpoint-resume tests load from prev run
        export CHECKPOINT_LOAD_PATH=$_CHECKPOINT_LOAD_PATH
        export CHECKPOINT_SAVE_PATH=/tmp/checkpoints/

        export RUN_NUMBER=2
        bash $ROOT_DIR/tests/functional_tests/shell_test_utils/_run_training.sh || TRAINING_EXIT_CODE=$?

        export CHECKPOINT_SAVE_PATH=$_CHECKPOINT_SAVE_PATH
        rm -rf "$CHECKPOINT_SAVE_PATH/iter_0000$TRAIN_ITERS"
        echo $((TRAIN_ITERS / 2)) >$CHECKPOINT_SAVE_PATH/latest_checkpointed_iteration.txt
    fi

    if [[ ${RECORD_CHECKPOINTS} == "true" ]]; then
        echo "Skipping Pytest during checkpoint recording."
        SKIP_PYTEST=1
    fi

    if [[ ${SKIP_PYTEST:-0} != 1 || "$TEST_TYPE" == "release" ]]; then
        # Save run results
        export PYTHONPATH=$ROOT_DIR
        if [[ "$TEST_TYPE" == "release" ]]; then
            EXTRACT_ARGS=("--is-convergence-test")
        else
            EXTRACT_ARGS=("--is-normal-test" "--step-size" "1")
        fi

        # Read test values from Tensorboard for non-inference tests.
        # Inference tests will load from JSON instead.
        if [[ "$MODE" == "pretraining" ]]; then
            uv run --no-sync python $ROOT_DIR/tests/functional_tests/python_test_utils/get_test_results_from_tensorboard_logs.py \
                --logs-dir $TENSORBOARD_PATH \
                --train-iters $TRAIN_ITERS \
                --output-path ${OUTPUT_PATH}/$(basename $GOLDEN_VALUES_PATH) \
                "${EXTRACT_ARGS[@]}"
        fi
    fi

    # Maybe run tests
    if [[ ${SKIP_PYTEST:-0} == 1 ]]; then
        echo Skipping Pytest checks.
        exit ${TRAINING_EXIT_CODE}
    fi

    if [[ ! " ${TEST_TYPES[*]} " =~ " ${TEST_TYPE} " ]]; then
        echo "Test type $TEST_TYPE not yet implemented."
    fi

    if [[ ! " ${MODES[*]} " =~ " ${MODE} " ]]; then
        echo "Mode $MODE not yet implemented."
    fi

    export NVTE_ALLOW_NONDETERMINISTIC_ALGO
    if [[ "${NVTE_ALLOW_NONDETERMINISTIC_ALGO}" == "1" || "${NON_DETERMINSTIC_RESULTS}" == "1" ]]; then
        ALLOW_NONDETERMINISTIC_ALGO_ARG="--allow-nondeterministic-algo"
    fi

    if [[ "$SLURM_NODEID" -eq 0 ]]; then
        echo "Running pytest checks against golden values"

        # For pretraining jobs
        if [[ "$MODE" == "pretraining" && ("$TRAINING_EXIT_CODE" -eq 0 || "$TEST_TYPE" == "release") ]]; then
            if [[ "$TEST_TYPE" == "checkpoint-consistency" ]]; then
                echo "Running checkpoint consistency check"
                uv run --no-sync python $ROOT_DIR/tests/functional_tests/python_test_utils/test_optimizer_grads_match.py "${ITER_CHECKPOINT_DIRS[@]}"
            else
                uv run --no-sync pytest -s -o log_cli=true --log-cli-level=info $ROOT_DIR/tests/functional_tests/python_test_utils/test_pretraining_regular_pipeline.py \
                    --golden-values-path $GOLDEN_VALUES_PATH \
                    --actual-values-path ${OUTPUT_PATH}/$(basename $GOLDEN_VALUES_PATH) \
                    --train-iters $TRAIN_ITERS \
                    --model-config-path ${TRAINING_PARAMS_PATH} \
                    $ALLOW_NONDETERMINISTIC_ALGO_ARG

                if [[ "$TEST_TYPE" == "ckpt-resume" || "$TEST_TYPE" == "frozen-resume" ]]; then
                    uv run --no-sync python $ROOT_DIR/tests/functional_tests/python_test_utils/get_test_results_from_tensorboard_logs.py \
                        --logs-dir $TENSORBOARD_PATH \
                        --train-iters $TRAIN_ITERS \
                        --output-path "${OUTPUT_PATH}/$(basename $GOLDEN_VALUES_PATH .json)_2nd.json" \
                        --is-second-run \
                        "${EXTRACT_ARGS[@]}"
                            
                    echo "Running pytest 1st vs 2nd run comparison"
                    uv run --no-sync pytest -s -o log_cli=true --log-cli-level=info $ROOT_DIR/tests/functional_tests/python_test_utils/test_pretraining_resume_checkpoint_pipeline.py \
                        --actual-values-first-run-path ${OUTPUT_PATH}/$(basename $GOLDEN_VALUES_PATH) \
                        --actual-values-second-run-path "${OUTPUT_PATH}/$(basename $GOLDEN_VALUES_PATH .json)_2nd.json" \
                        --train-iters $TRAIN_ITERS \
                        --model-config-path ${TRAINING_PARAMS_PATH} \
                        $ALLOW_NONDETERMINISTIC_ALGO_ARG
                fi
            fi
        fi

        # For inference jobs
        if [[ "$MODE" == "inference" && ("$TRAINING_EXIT_CODE" -eq 0 || "$TEST_TYPE" == "release") ]]; then
            if [[ "$TEST_TYPE" == "frozen-start" ]]; then
                uv run --no-sync pytest -s -o log_cli=true --log-cli-level=info $ROOT_DIR/tests/functional_tests/python_test_utils/test_inference_regular_pipeline.py \
                    --golden-values-path $GOLDEN_VALUES_PATH \
                    --test-values-path $TENSORBOARD_PATH \
                    --model-config-path ${TRAINING_PARAMS_PATH} \
                    $ALLOW_NONDETERMINISTIC_ALGO_ARG
            fi
        fi

        # For rl jobs
        if [[ "$MODE" == "rl" && ("$TRAINING_EXIT_CODE" -eq 0 || "$TEST_TYPE" == "release") ]]; then
            if [[ "$TEST_TYPE" == "frozen-start" ]]; then
                TRAIN_ITERS=$(cat $TRAINING_PARAMS_PATH |
                    /usr/local/bin/yq '.MODEL_ARGS."--exit-interval" // "50"')
                uv run --no-sync python $ROOT_DIR/tests/functional_tests/python_test_utils/get_test_results_from_tensorboard_logs.py \
                    --logs-dir $TENSORBOARD_PATH \
                    --train-iters $TRAIN_ITERS \
                    --output-path ${OUTPUT_PATH}/$(basename $GOLDEN_VALUES_PATH) \
                    "${EXTRACT_ARGS[@]}"
                uv run --no-sync pytest -s -o log_cli=true --log-cli-level=info $ROOT_DIR/tests/functional_tests/python_test_utils/test_grpo_training_loop.py \
                    --golden-values-path $GOLDEN_VALUES_PATH \
                    --test-values-path ${OUTPUT_PATH}/$(basename $GOLDEN_VALUES_PATH) \
                    --model-config-path ${TRAINING_PARAMS_PATH} \
                    $ALLOW_NONDETERMINISTIC_ALGO_ARG
            fi
        fi

        # Abort if training failed
        if [[ "$TRAINING_EXIT_CODE" -ne 0 && "$TEST_TYPE" != "release" ]]; then
            echo "Training failed. Aborting."
            exit 1
        fi
    fi
done

