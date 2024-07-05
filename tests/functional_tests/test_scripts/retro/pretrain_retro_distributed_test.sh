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

GPUS_PER_NODE=8
# Change for multinode config
MASTER_ADDR=localhost
MASTER_PORT=6000
NODE_RANK=0
WORLD_SIZE=$(($GPUS_PER_NODE*$NUM_NODES))

command="export CUDA_DEVICE_MAX_CONNECTIONS=1;"

TRANSFORMER_IMPL=local
TRAINING_DTYPE=bf16

USE_LEGACY=1
if [[ $USE_CORE -eq 1 ]]; then
       echo "Running using megatron core"
       TRANSFORMER_IMPL=local
       TRAINING_DTYPE=bf16
       command="$command export NVTE_ALLOW_NONDETERMINISTIC_ALGO=0;"
       unset USE_LEGACY
       export NVTE_ALLOW_NONDETERMINISTIC_ALGO=0
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
# Runs the "345M" parameter model
DISTRIBUTED_ARGS="--max-restarts 3 --nproc_per_node $GPUS_PER_NODE --nnodes $NUM_NODES"

build_args() {
  ARGS=" \
    --exit-interval $MAX_STEPS \
    \
    --recompute-activations \
    --use-flash-attn \
    --apply-layernorm-1p \
    --untie-embeddings-and-output-weights \
    --disable-bias-linear \
    --no-position-embedding \
    --use-rotary-position-embeddings \
    --rotary-percent 0.5 \
    --swiglu \
    --attention-dropout 0.0 \
    --hidden-dropout 0.0 \
    --exit-duration-in-mins 220 \
    --tensor-model-parallel-size $TP_SIZE \
    --pipeline-model-parallel-size 1 \
    --num-layers 24 \
    --hidden-size 1024 \
    --num-attention-heads 16 \
    --seq-length 2048 \
    --max-position-embeddings 2048 \
    --micro-batch-size $MBS \
    --global-batch-size 256 \
    --train-samples 100000 \
    --lr-decay-samples 99000 \
    --lr-warmup-samples 1000 \
    --lr 2.5e-5 \
    --min-lr 2.5e-6 \
    --lr-decay-style cosine \
    --log-interval 5 \
    --eval-iters 100 \
    --eval-interval 2000 \
    --tokenizer-type GPT2BPETokenizer \
    --vocab-file /workspace/data/retro_data/vocab/gpt2-vocab.json \
    --merge-file /workspace/data/retro_data/vocab/gpt2-merges.txt \
    --data-path /workspace/data/retro_data/inputs/wiki-200k_text_document \
    --split 98,2,0 \
    --clip-grad 1.0 \
    --weight-decay 0.1 \
    --adam-beta1 0.9 \
    --adam-beta2 0.95 \
    --init-method-std 0.007 \
    --log-params-norm \
    --log-num-zeros-in-grad \
    --log-validation-ppl-to-tensorboard \
    --log-timers-to-tensorboard \
    --tensorboard-dir ${TENSORBOARD_DIR} \
    --save-interval $__SAVE_INTERVAL \
    --save $CHECKPOINT_PATH \
    --load $CHECKPOINT_PATH \
    --bf16 \
    --transformer-impl $TRANSFORMER_IMPL \
    --${TRAINING_DTYPE} \
    ${USE_LEGACY:+--use-legacy-models} \
    ${ADDITIONAL_PARAMS:+$ADDITIONAL_PARAMS} \
    --retro-workdir /workspace/data/retro_data/neighbors
    --retro-add-retriever \
    --num-workers 32 \
"
}

build_args
torch_run_cmd="torchrun $DISTRIBUTED_ARGS \
    pretrain_retro.py \
    ${ARGS}"

command="$command $torch_run_cmd"

if [[ $CHECKPOINT_RESUME_TEST -eq 1 ]]; then
  MAX_STEPS=50
  build_args
  torch_run_cmd="torchrun $DISTRIBUTED_ARGS \
    pretrain_retro.py \
    ${ARGS}"
  command="$command; rm -rf $CHECKPOINT_PATH/iter_0000100; echo 50 > $CHECKPOINT_PATH/latest_checkpointed_iteration.txt; $torch_run_cmd"
fi
echo "-------------------- THE FINAL PRETRAIN SCRIPT COMMAND THAT WILL BE RUN ------------"
echo "$command"
echo "-----------------------------------------------------------------------------"

pip install h5py
pip install transformers
pip install faiss-gpu

echo "$command" > $SCRIPTS_DIR/pretrain_retro_distributed_command.sh
eval $command

echo "Saving test results to $TENSORBOARD_DIR"
PYTHONPATH=$PWD python3 ./tests/functional_tests/python_test_utils/get_test_results_from_tensorboard_logs.py $TENSORBOARD_DIR "$JOB_NAME" | \
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
