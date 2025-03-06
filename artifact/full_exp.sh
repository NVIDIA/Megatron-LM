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

export LOGS_DIR='./full-logs'

if [ -z "$1" ]; then
    echo "Usage: bash full_exp.sh <experiment_settings_file>"
    exit 1
fi
EXP_SETTING_FILE=$1

while read -r line; do
    # csv column header: Name	B	s	h	v	l	imm	heads	dp	pp	tp	layers_last	micro_bs	method
    echo $line
    name=$(echo $line | cut -d ',' -f 1)
    if [ -z "$name" ]; then
        continue
    fi
    export GLOBAL_BATCH_SIZE=$(echo $line | cut -d ',' -f 2)
    export SEQ_LENGTH=$(echo $line | cut -d ',' -f 3)
    export HIDDEN_SIZE=$(echo $line | cut -d ',' -f 4)
    vocab_size=$(echo $line | cut -d ',' -f 5)
    vocab_size=$(( vocab_size / 1000 ))
    export VOCAB_SIZE="${vocab_size}k"
    export LAYERS=$(echo $line | cut -d ',' -f 6)
    export IMM_SIZE=$(echo $line | cut -d ',' -f 7)
    export ATTENTION_HEADS=$(echo $line | cut -d ',' -f 8)
    export PIPELINE_SIZE=$(echo $line | cut -d ',' -f 10)
    export FINAL_STAGE_LAYERS=$(echo $line | cut -d ',' -f 12)
    export MICRO_BATCH_SIZE=$(echo $line | cut -d ',' -f 13)
    METHOD=$(echo $line | cut -d ',' -f 14)

    # Reset
    export ENABLE_LAYER_REDISTRIBUTION=
    export VOCAB_PARALLEL=
    export INTERLACED_SCHEDULE=
    export FB_SPLIT=

    # echo $GLOBAL_BATCH_SIZE $SEQ_LENGTH $HIDDEN_SIZE $VOCAB_SIZE $LAYERS $IMM_SIZE $ATTENTION_HEADS $PIPELINE_SIZE $MICRO_BATCH_SIZE $METHOD
    case "$METHOD" in
        "base")
            # Baseline case - no additional exports needed
            ;;
        "redis")
            export ENABLE_LAYER_REDISTRIBUTION=1
            ;;
        "interlaced")
            export VOCAB_PARALLEL=1
            export INTERLACED_SCHEDULE=1
            ;;
        "synctwice")
            export VOCAB_PARALLEL=1
            export FB_SPLIT=1
            ;;
        "vocab")
            export VOCAB_PARALLEL=1
            ;;
        *)
            echo "Invalid method. Please use: base, interlaced, synctwice, or vocab"
            exit 1
            ;;
    esac

    export AIP_RUN_NAME="${name}"
    export RUN_NAME=${AIP_RUN_NAME}

    mkdir -p "${LOGS_DIR}/${RUN_NAME}"
    echo "running: ${RUN_NAME}"
    bash pretrain_gpt.sh > ${LOGS_DIR}/${RUN_NAME}/stdout.log 2> >(tee ${LOGS_DIR}/${RUN_NAME}/stderr.log >&2)
done < $1
