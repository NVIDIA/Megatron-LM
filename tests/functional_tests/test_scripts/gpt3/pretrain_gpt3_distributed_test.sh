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
if [[ -z $MOE_GROUPED_GEMM ]]; then MOE_GROUPED_GEMM=0; fi
if [[ -z $VOCAB_FILE ]]; then VOCAB_FILE="/workspace/data/gpt3_data/vocab.json" ; fi
if [[ -z $MERGE_FILE ]]; then MERGE_FILE="/workspace/data/gpt3_data/merges.txt" ; fi

GPUS_PER_NODE=8
# Change for multinode config
MASTER_ADDR=localhost
MASTER_PORT=6000
NODE_RANK=0
WORLD_SIZE=$(($GPUS_PER_NODE*$NUM_NODES))

command="export CUDA_DEVICE_MAX_CONNECTIONS=1;"

TRANSFORMER_IMPL=local
TRAINING_DTYPE=fp16

if [[ $USE_CORE -eq 1 ]]; then
       echo "Running using megatron core"
       TRANSFORMER_IMPL=transformer_engine
       TRAINING_DTYPE=bf16
       command="$command export NVTE_ALLOW_NONDETERMINISTIC_ALGO=0;"
       USE_MCORE=1
fi

if [[ $MOE_GROUPED_GEMM -eq 1 ]]; then
       echo "Running MoE with Grouped GEMM"
       TRAINING_DTYPE=bf16  # Currently GroupedGEMM for MoE only supports bf16 dtype
fi

if [[ $USE_TE -eq 1 ]]; then
       echo "Running with TransformerEngine ..."
       TRANSFORMER_IMPL=transformer_engine
       TRAINING_DTYPE=bf16
       ADDITIONAL_PARAMS+=" --attention-softmax-in-fp32"
else
       echo "Running with local transformer implementation ..."
fi
if [[ $CHECKPOINT_RESUME_TEST -eq 1 ]]; then
       echo "Running checkpoint resume test..."
       __SAVE_INTERVAL=50
       ADDITIONAL_PARAMS+=" --use-checkpoint-opt_param-scheduler"
       if [[ $MAX_STEPS -ne 100 ]]; then
         echo "Overriding MAX_STEPS=100"
         MAX_STEPS=100
       fi
else
       __SAVE_INTERVAL=${SAVE_INTERVAL:-10000}  # inf
fi
if [[ -n "$CKPT_FORMAT" ]] && [[ "$CKPT_FORMAT" != 'torch' ]]; then
       echo "Using distributed checkpoint format $CKPT_FORMAT..."
       [[ "$CKPT_FORMAT" == 'zarr' ]] && command="$command pip install zarr tensorstore==0.1.45;"
       ADDITIONAL_PARAMS+=" --use-dist-ckpt --dist-ckpt-format $CKPT_FORMAT"
fi
set +x
# Runs the "345M" parameter model

build_torch_run_cmd() {
  DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE --nnodes $NUM_NODES"
  [[ -n "$RUN_CMD" ]] && run_cmd=$RUN_CMD || run_cmd="torchrun $DISTRIBUTED_ARGS"
  torch_run_cmd="$run_cmd \
       pretrain_gpt.py \
       --num-layers 12 \
       --hidden-size 512 \
       --num-attention-heads 8 \
       --log-params-norm \
       --log-num-zeros-in-grad \
       --log-validation-ppl-to-tensorboard \
       --log-timers-to-tensorboard \
       --tensorboard-dir ${TENSORBOARD_DIR} \
       --micro-batch-size ${MBS:-4} \
       --global-batch-size ${GBS:-32} \
       --seq-length 1024 \
       --max-position-embeddings 1024 \
       --train-iters $MAX_STEPS \
       --timing-log-level 2 \
       --lr-decay-iters 320000 \
       --save $CHECKPOINT_PATH \
       --load $CHECKPOINT_PATH \
       --data-path $DATA_PATH \
       --vocab-file $VOCAB_FILE \
       --merge-file $MERGE_FILE \
       --split 949,50,1 \
       --distributed-backend nccl \
       --lr 0.00015 \
       --lr-decay-style cosine \
       --min-lr 1.0e-5 \
       --weight-decay 1e-2 \
       --clip-grad 1.0 \
       --lr-warmup-fraction .01 \
       --log-interval 1 \
       --save-interval $__SAVE_INTERVAL \
       --eval-interval 1000 \
       --eval-iters 10 \
       --transformer-impl $TRANSFORMER_IMPL \
       --tensor-model-parallel-size $TP_SIZE \
       --pipeline-model-parallel-size $PP_SIZE \
       --no-bias-swiglu-fusion \
       --use-gpu-initialization \
       --no-rope-fusion \
       ${VP_SIZE:+--num-layers-per-virtual-pipeline-stage "$VP_SIZE"} \
       ${ADDITIONAL_PARAMS:+$ADDITIONAL_PARAMS} \
       ${USE_MCORE:+--use-mcore-models} \
       --no-gradient-accumulation-fusion \
       ${DATA_CACHE:+--data-cache-path "$DATA_CACHE"} \
       --${TRAINING_DTYPE}"

  if [[ "${TRAINING_DTYPE}" == "fp16" ]]; then
      torch_run_cmd+=" --apply-query-key-layer-scaling"
  fi
}

build_torch_run_cmd
command="$command $torch_run_cmd"
if [[ $CHECKPOINT_RESUME_TEST -eq 1 ]]; then
  echo "------RESUME OVERRIDES ARGS LIST --------"
  # apply all env vars starting from 'RESUME_OVERRIDE_' (after removing prefix)
  _OVERRIDE_PREFIX="RESUME_OVERRIDE_"
  _OVERRIDE_PREFIX_LENGTH=${#_OVERRIDE_PREFIX}
  _NONEMPTY_OVERRIDES=0
  for ARGUMENT in "$@"
  do
    KEY=$(echo $ARGUMENT | cut -f1 -d=)
    if [[ $KEY == ${_OVERRIDE_PREFIX}* ]]; then
      KEY_LENGTH=${#KEY}
      VALUE="${ARGUMENT:$KEY_LENGTH+1}"
      KEY="${KEY:$_OVERRIDE_PREFIX_LENGTH}"
      if [[ -n "${VALUE}" ]]; then
        export "$KEY"="$VALUE"
        echo "$KEY=$VALUE"
        _NONEMPTY_OVERRIDES=1
      fi
    fi
  done
  echo "---------------------------------"
  if [[ $_NONEMPTY_OVERRIDES == 1 ]]; then
    ADDITIONAL_PARAMS+=" --no-load-rng"  # assuming TPxPP mismatch
  fi

  build_torch_run_cmd
  command="$command; rm -rf $CHECKPOINT_PATH/iter_0000100; echo 50 > $CHECKPOINT_PATH/latest_checkpointed_iteration.txt; $torch_run_cmd"
fi
echo "-------------------- THE FINAL PRETRAIN SCRIPT COMMAND THAT WILL BE RUN ------------"
echo "$command"
echo "-----------------------------------------------------------------------------"

echo "$command" > $SCRIPTS_DIR/pretrain_gpt3_distributed_command.sh
eval $command
