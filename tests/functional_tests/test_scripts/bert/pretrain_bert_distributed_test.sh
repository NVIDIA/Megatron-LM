#! /bin/bash
echo "------ARGUMENTS LIST --------"
for ARGUMENT in "$@"
do
   KEY=$(echo $ARGUMENT | cut -f1 -d=)

   KEY_LENGTH=${#KEY}
   VALUE="${ARGUMENT:$KEY_LENGTH+1}"

   export "$KEY"="$VALUE"
   echo "$KEY=$VALUE"
done
echo "---------------------------------"

set -exo pipefail
if [[ -z $MBS ]]; then MBS=4; fi
if [[ -z $GBS ]]; then GBS=128; fi
if [[ -z $VOCAB_FILE ]]; then VOCAB_FILE="/workspace/data/bert_data/vocab.txt" ; fi
if [[ -z $ALLOW_NONDETERMINISTIC ]]; then ALLOW_NONDETERMINISTIC=0; fi

# Change for multinode config
GPUS_PER_NODE=8
MASTER_ADDR=localhost
MASTER_PORT=6000
NODE_RANK=0
WORLD_SIZE=$(($GPUS_PER_NODE*$NUM_NODES))
command="export CUDA_DEVICE_MAX_CONNECTIONS=1;"

TRAINING_DTYPE=fp16
TRANSFORMER_IMPL=local

if [[ $ALLOW_NONDETERMINISTIC -eq 1 ]]; then
   command="$command export NVTE_ALLOW_NONDETERMINISTIC_ALGO=1;"
else
   command="$command export NVTE_ALLOW_NONDETERMINISTIC_ALGO=0; export NCCL_ALGO=^NVLS;"
   ADDITIONAL_PARAMS+=" --deterministic-mode"
fi

USE_LEGACY=1
if [[ $USE_CORE -eq 1 ]]; then
       echo "Running using megatron core"
       TRANSFORMER_IMPL=local
       TRAINING_DTYPE=bf16
       unset USE_LEGACY
fi
if [[ $CHECKPOINT_RESUME_TEST -eq 1 ]]; then
       echo "Running checkpoint resume test..."
       __SAVE_INTERVAL=50
       ADDITIONAL_PARAMS+=" --use-checkpoint-args --use-checkpoint-opt_param-scheduler"
       if [[ $MAX_STEPS -ne 100 ]]; then
         echo "Overriding MAX_STEPS=100"
         MAX_STEPS=100
       fi
else
       __SAVE_INTERVAL=10000  # inf
fi
# Runs the "345M" parameter model
DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE --nnodes $NUM_NODES"

torch_run_cmd="torchrun $DISTRIBUTED_ARGS \
       pretrain_bert.py \
       --num-layers 24 \
       --hidden-size 1024 \
       --num-attention-heads 16 \
       --log-params-norm \
       --log-num-zeros-in-grad \
       --log-validation-ppl-to-tensorboard \
       --log-timers-to-tensorboard \
       --tensorboard-dir ${TENSORBOARD_DIR} \
       --micro-batch-size ${MBS:-4} \
       --global-batch-size ${GBS:-128} \
       --seq-length 512 \
       --max-position-embeddings 512 \
       --train-iters $MAX_STEPS \
       --timing-log-level 2 \
       --lr-decay-iters 990000 \
       --save $CHECKPOINT_PATH \
       --load $CHECKPOINT_PATH \
       --data-path $DATA_PATH \
       --vocab-file $VOCAB_FILE \
       --split 949,50,1 \
       --distributed-backend nccl \
       --lr 0.0001 \
       --min-lr 0.00001 \
       --lr-warmup-fraction 0.01 \
       --log-interval 1 \
       --save-interval $__SAVE_INTERVAL \
       --eval-interval 1000 \
       --eval-iters 10 \
       --tensor-model-parallel-size $TP_SIZE \
       --pipeline-model-parallel-size $PP_SIZE \
       ${VP_SIZE:+--num-layers-per-virtual-pipeline-stage "$VP_SIZE"} \
       ${USE_LEGACY:+--use-legacy-models} \
       ${ADDITIONAL_PARAMS:+$ADDITIONAL_PARAMS} \
       --no-gradient-accumulation-fusion \
       ${DATA_CACHE:+--data-cache-path "$DATA_CACHE"} \
       --${TRAINING_DTYPE}"

if [[ "${TRAINING_DTYPE}" == "fp16" ]]; then
    # Both NVTE_APPLY_QK_LAYER_SCALING and --apply-query-key-layer-scaling must be passed
    # to enable feature and be backward compatible with TE<0.11
    export NVTE_APPLY_QK_LAYER_SCALING=1
    torch_run_cmd+=" --apply-query-key-layer-scaling"
    # NVTE_APPLY_QK_LAYER_SCALING=1 is required if using:
    #  1. --apply-query-key-layer-scaling
    #  2. transformer_impl="transformer_engine"
    #  3. TE >= 0.11
    #  4. fp16
    export NVTE_APPLY_QK_LAYER_SCALING=1
fi

command="$command $torch_run_cmd"
if [[ $CHECKPOINT_RESUME_TEST -eq 1 ]]; then
  command="$command; rm -rf $CHECKPOINT_PATH/iter_0000100; echo 50 > $CHECKPOINT_PATH/latest_checkpointed_iteration.txt; $torch_run_cmd"
fi
echo "-------------------- THE FINAL PRETRAIN SCRIPT COMMAND THAT WILL BE RUN ------------"
echo "$command"
echo "-----------------------------------------------------------------------------"

echo "$command" > $SCRIPTS_DIR/pretrain_bert_distributed_command.sh
eval $command

echo "Saving test results to $TENSORBOARD_DIR"
PYTHONPATH=$PWD python3 ./tests/functional_tests/python_test_utils/get_test_results_from_tensorboard_logs.py $TENSORBOARD_DIR "$JOB_NAME" | \
    tee ${TENSORBOARD_DIR}/results.json

if [[ $SKIP_PYTEST != 1 ]]; then
    echo "-----------------------------------------------------------------------------"
    if [[ $CHECKPOINT_RESUME_TEST -eq 1 ]]; then
        echo "Running pytest 1st vs 2nd run comparison"
        export LOGS_DIR=$TENSORBOARD_DIR
        pytest -s ./tests/functional_tests/python_test_utils/test_resume_checkpoint_pipeline.py
    else
        echo "Running pytest checks against golden values"
        export EXPECTED_METRICS_FILE="./tests/functional_tests/test_results/jet/${JOB_NAME}.json"
        export LOGS_DIR=$TENSORBOARD_DIR
        pytest -s ./tests/functional_tests/python_test_utils/test_ci_pipeline.py
    fi
fi
