#!/bin/bash

LOCAL_GPU_NUM=$(nvidia-smi --list-gpus | wc -l)

export EXIT_INTERVAL=20
export LOG_INTERVAL=1
# disable timer for schedule
export SCHEDULE_TIMER_START=1000
export SCHEDULE_TIMER_END=1000

export GPUS_PER_NODE=8
if [ $LOCAL_GPU_NUM -ne $GPUS_PER_NODE ]; then
    echo "Error: Expect 8 GPUs. Found $LOCAL_GPU_NUM"
    exit 1
fi

export PIPELINE_SIZE=$(( $GPUS_PER_NODE ))
export LAYERS=32
export ATTENTION_HEADS=24
export HIDDEN_SIZE=3072
export SEQ_LENGTH=4096
export MICRO_BATCH_SIZE=1
export VOCAB_SIZE=256k

export IMM_SIZE=12288
export GLOBAL_BATCH_SIZE=128
# export GLOBAL_BATCH_SIZE=$(( $PIPELINE_SIZE * 3 * $MICRO_BATCH_SIZE ))

export LOGS_DIR='./8g-logs'

print_help() {
    echo "Usage: quick_exp.sh run [baseline|redis|interlaced|vocab-1|vocab-2] or quick_exp.sh show-result"
}

if [ -z "$1" ]; then
    print_help
    exit 1
fi

COMMAND=$1

extract_memory() {
    local method=$1
    local log_file="${LOGS_DIR}/${method}/stdout.log"
    
    if [ ! -f "$log_file" ]; then
        echo "Error: ${log_file} does not exist"
        return 1
    fi

    # Extract all memory values
    local memory_values=($(grep "memory (MB)" "$log_file" | grep -o "max allocated: [0-9.]\+" | awk '{print $3}'))
    
    # Check if number of values matches PIPELINE_SIZE
    if [ ${#memory_values[@]} -ne $PIPELINE_SIZE ]; then
        echo "Error: Expected ${PIPELINE_SIZE} memory values, but found ${#memory_values[@]}"
        return 1
    fi

    # Print all memory values
    echo "Memory values:"
    for i in "${!memory_values[@]}"; do
        echo "GPU $i: ${memory_values[$i]} MB"
    done

    # Find min and max
    local min=${memory_values[0]}
    local max=${memory_values[0]}
    
    for value in "${memory_values[@]}"; do
        if [ $(awk '{if ($1 > $2) print 1; else print 0}' <<< "$value $max") -eq 1 ]; then
            max=$value
        fi
        if [ $(awk '{if ($1 < $2) print 1; else print 0}' <<< "$value $min") -eq 1 ]; then
            min=$value
        fi
    done

    echo "Min: $min, Max: $max"
}

extract_time() {
    local method=$1
    local log_file="${LOGS_DIR}/${method}/stdout.log"
    
    if [ ! -f "$log_file" ]; then
        echo "Error: ${log_file} does not exist"
        return 1
    fi

    # Extract iteration time for iteration 20
    local time_value=$(grep "iteration.*20/" "$log_file" | grep -o "elapsed time per iteration (ms): [0-9.]\+" | awk '{print $NF}')
    
    if [ -z "$time_value" ]; then
        echo "Error: Could not find iteration 20 timing information"
        return 1
    fi

    echo "Iteration time: ${time_value} ms"
}

show_results() {
    # Show results from logs directory
    for method in baseline redis interlaced vocab-1 vocab-2; do
        if [ -f "${LOGS_DIR}/${method}/stdout.log" ]; then
            echo "=== Results for ${method} ==="
            extract_memory "$method"
            extract_time "$method"
            echo
        else
            echo "Error: ${LOGS_DIR}/${method}/stdout.log does not exist"
        fi
    done
}

if [ "$COMMAND" = "show-result" ]; then
    show_results
    exit 0
fi

if [ "$COMMAND" != "run" ]; then
    echo "Invalid command. Use 'run' or 'show-result'"
    print_help
    exit 1
fi

if [ -z "$2" ]; then
    echo "Please specify method: baseline, redis, interlaced, vocab-1, or vocab-2"
    exit 1
fi

METHOD=$2

case "$METHOD" in
    "baseline")
        # Baseline case - no additional exports needed
        ;;
    "redis")
        export ENABLE_LAYER_REDISTRIBUTION=1
        export FINAL_STAGE_LAYERS=0
        ;;
    "interlaced")
        export VOCAB_PARALLEL=1
        export INTERLACED_SCHEDULE=1
        ;;
    "vocab-1")
        export VOCAB_PARALLEL=1
        export FB_SPLIT=1
        ;;
    "vocab-2")
        export VOCAB_PARALLEL=1
        ;;
    *)
        echo "Invalid method. Please use: baseline, redis, interlaced, vocab-1, or vocab-2"
        print_help
        exit 1
        ;;
esac

export AIP_RUN_NAME="${METHOD}"

mkdir -p ${LOGS_DIR}/${METHOD}
bash pretrain_gpt.sh > ${LOGS_DIR}/${METHOD}/stdout.log 2> >(tee ${LOGS_DIR}/${METHOD}/stderr.log >&2)
# PROFILED=1 bash pretrain_gpt.sh > ${LOGS_DIR}/${METHOD}/stdout.log 2> >(tee ${LOGS_DIR}/${METHOD}/stderr.log >&2)

