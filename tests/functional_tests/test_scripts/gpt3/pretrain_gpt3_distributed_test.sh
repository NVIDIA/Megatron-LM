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

set -x
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
       TRANSFORMER_IMPL=local
       TRAINING_DTYPE=bf16
       command="$command export NVTE_ALLOW_NONDETERMINISTIC_ALGO=0;"
       USE_MCORE=1
fi

if [[ $MOE_GROUPED_GEMM -eq 1 ]]; then
       echo "Running MoE with Grouped GEMM"
       command="$command pip install git+https://github.com/fanshiqing/grouped_gemm@main;"
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
set +x
# Runs the "345M" parameter model
DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE --nnodes $NUM_NODES"

torch_run_cmd="torchrun $DISTRIBUTED_ARGS \
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
       --save-interval 10000 \
       --eval-interval 1000 \
       --eval-iters 10 \
       --transformer-impl $TRANSFORMER_IMPL \
       --tensor-model-parallel-size $TP_SIZE \
       --pipeline-model-parallel-size $PP_SIZE \
       --no-bias-swiglu-fusion \
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

command="$command $torch_run_cmd"
echo "-------------------- THE FINAL PRETRAIN SCRIPT COMMAND THAT WILL BE RUN ------------"
echo "$command"
echo "-----------------------------------------------------------------------------"

echo "$command" > $SCRIPTS_DIR/pretrain_gpt3_distributed_command.sh
eval $command
