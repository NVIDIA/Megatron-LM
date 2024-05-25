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
if [[ -z $GBS ]]; then GBS=32; fi
if [[ -z $VOCAB_PATH ]]; then VOCAB_PATH="/workspace/data/t5_data/bert-large-cased-vocab.txt"; fi
if [[ -z $ALLOW_NONDETERMINISTIC ]]; then ALLOW_NONDETERMINISTIC=0; fi

GPUS_PER_NODE=8
# Change for multinode config
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

if [[ $NO_FA -eq 1 ]]; then
       echo "Turn off flash attention environment variable"
       export NVTE_FLASH_ATTN=0
       export NVTE_FUSED_ATTN=0
fi

if [[ $USE_TE -eq 1 ]]; then
       echo "Running with TransformerEngine ..."
       TRANSFORMER_IMPL=transformer_engine
       TRAINING_DTYPE=bf16
else
       echo "Running with local transformer implementation ..."
fi

if [[ $CHECKPOINT_RESUME_TEST -eq 1 ]]; then
       echo "Running checkpoint resume test..."
       __SAVE_INTERVAL=50
       if [[ $MAX_STEPS -ne 100 ]]; then
         echo "Overriding MAX_STEPS=100"
         MAX_STEPS=100
       fi
else
       __SAVE_INTERVAL=10000  # inf
fi
set +x

# install neccessary library
pip install pydantic==2.2.1

# Runs the "220M" parameter model
DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE --nnodes $NUM_NODES"

torch_run_cmd="torchrun $DISTRIBUTED_ARGS \
    pretrain_t5.py \
    --encoder-num-layers 12 \
    --decoder-num-layers 12 \
    --hidden-size 768 \
    --num-attention-heads 12 \
    --kv-channels 64 \
    --ffn-hidden-size 3072 \
    --encoder-seq-length 512 \
    --decoder-seq-length 128 \
    --max-position-embeddings 512 \
    --tensor-model-parallel-size $TP_SIZE \
    --pipeline-model-parallel-size $PP_SIZE \
    --micro-batch-size ${MBS:-4} \
    --global-batch-size ${GBS:-32} \
    --lr 0.0001 \
    --train-iters $MAX_STEPS \
    --lr-decay-iters $MAX_STEPS \
    --lr-decay-style linear \
    --min-lr 0.00001 \
    --weight-decay 1e-2 \
    --lr-warmup-fraction .01 \
    --clip-grad 1.0 \
    --${TRAINING_DTYPE} \
    --vocab-extra-ids 100 \
    --init-method-std 0.015 \
    --transformer-impl $TRANSFORMER_IMPL \
    --data-path $DATA_PATH \
    --vocab-file $VOCAB_PATH \
    --tokenizer-type BertWordPieceCase \
    --split 99982,9,9 \
    --save $CHECKPOINT_PATH \
    --load $CHECKPOINT_PATH \
    --tensorboard-dir ${TENSORBOARD_DIR} \
    --log-params-norm \
    --log-num-zeros-in-grad \
    --log-validation-ppl-to-tensorboard \
    --log-timers-to-tensorboard \
    --timing-log-level 2 \
    --log-interval 1 \
    --save-interval $__SAVE_INTERVAL \
    --eval-interval 1000 \
    --eval-iters 10 \
    --distributed-backend nccl \
    ${DATA_CACHE:+--data-cache-path "$DATA_CACHE"} \
    ${USE_LEGACY:+--use-legacy-models} \
    ${ADDITIONAL_PARAMS:+$ADDITIONAL_PARAMS}"

command="$command $torch_run_cmd"
if [[ $CHECKPOINT_RESUME_TEST -eq 1 ]]; then
  command="$command; rm -rf $CHECKPOINT_PATH/iter_0000100; echo 50 > $CHECKPOINT_PATH/latest_checkpointed_iteration.txt; $torch_run_cmd"
fi
echo "-------------------- THE FINAL PRETRAIN SCRIPT COMMAND THAT WILL BE RUN ------------"
echo "$command"
echo "-----------------------------------------------------------------------------"

echo "$command" > $SCRIPTS_DIR/pretrain_t5_distributed_command.sh
eval $command

echo "Saving test results to $TENSORBOARD_DIR"
python3 ./tests/functional_tests/python_test_utils/get_test_results_from_tensorboard_logs.py $TENSORBOARD_DIR "$JOB_NAME" | \
    tee ${TENSORBOARD_DIR}/results.json

if [[ $SKIP_PYTEST != 1 ]]; then
    echo "-----------------------------------------------------------------------------"
    if [[ $CHECKPOINT_RESUME_TEST -eq 1 ]]; then
        echo "Running pytest 1st vs 2nd run comparison"
        export LOGS_DIR=$TENSORBOARD_DIR
        pytest ./tests/functional_tests/python_test_utils/test_resume_checkpoint_pipeline.py
    else
        echo "Running pytest checks against golden values"
        export EXPECTED_METRICS_FILE="./tests/functional_tests/test_results/jet/${JOB_NAME}.json"
        export LOGS_DIR=$TENSORBOARD_DIR
        pytest ./tests/functional_tests/python_test_utils/test_ci_pipeline.py
    fi
fi
