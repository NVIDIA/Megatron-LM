#!/bin/bash

# set -x

export GPU_MAX_HW_QUEUES=2
export TORCH_NCCL_HIGH_PRIORITY=1

# parsing input arguments
for ARGUMENT in "$@"
do
   KEY=$(echo $ARGUMENT | cut -f1 -d=)

   KEY_LENGTH=${#KEY}
   VALUE="${ARGUMENT:$KEY_LENGTH+1}"
   export "$KEY"="$VALUE"
done

# Change for multinode config
export CUDA_DEVICE_MAX_CONNECTIONS=1

USE_FLASH_ATTN="${USE_FLASH_ATTN:-1}"
NO_TRAINING="${NO_TRAINING:-0}" # NO_TRAINING=1: for computing metrics only
ENABLE_PROFILING="${ENABLE_PROFILING:-0}"
ENABLE_ROPE="${ENABLE_ROPE:-1}"
DISABLE_ROPE_TE="${DISABLE_ROPE_TE:-0}"
ENABLE_MOCK_DATA="${ENABLE_MOCK_DATA:-1}"
DUMMY_RUN="${DUMMY_RUN:-0}"
ADD_TASK="${ADD_TASK:-0}"
LABEL="${LABEL:-"test"}"
LOG_DIR="profile/${LABEL}"
echo "NO_TRAINING=$NO_TRAINING"

CWD=`pwd`
GPUS_PER_NODE=`python -c "import torch; print(torch.cuda.device_count())"`

# Change for multinode config
MASTER_ADDR=${MASTER_ADDR:-localhost}
MASTER_PORT=${MASTER_PORT:-23731}
NNODES=${NNODES:-1}
NODE_RANK=${NODE_RANK:-0}
WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES))

MODEL_SIZE="${MODEL_SIZE:-70}"
TP="${TP:-8}"
PP="${PP:-1}"
MBS="${MBS:-2}"
BS="${BS:-8}"
SEQ_LENGTH="${SEQ_LENGTH:-4096}"
TOTAL_ITERS="${TOTAL_ITERS:-20}"
SEQ_PARALLEL="${SEQ_PARALLEL:-1}" 
CONTI_PARAMS="${CONTI_PARAMS:-0}"
OPTIMIZER="${OPTIMIZER:-sgd}"
TE_BF16="${TE_BF16:-1}"

TMP_DIR="tmp"

CHECKPOINT_PATH=${CHECKPOINT_PATH:-"$TMP/ckpts"}
mkdir -p ${CHECKPOINT_PATH}

DATA_PATH=${DATA_PATH:-"$TMP_DIR/data/bookcorpus_text_sentence"}

TOKENIZER_MODEL=$TMP/tokenizer.model

# Download the tokenizer model
if ! [ -f "$TOKENIZER_MODEL" ]; then
wget -O $TOKENIZER_MODEL https://huggingface.co/NousResearch/Llama-2-7b-chat-hf/resolve/main/tokenizer.model
fi

MAX_POSITION_EMBEDDINGS=32768
LOG_NAME="${TMP_DIR}/train_${MODEL_SIZE}B_iter${TOTAL_ITERS}_mbs${MBS}_bs${BS}_tp${TP}_pp${PP}_\
            seq${SEQ_LENGTH}_optim_${OPTIMIZER}_fa_${USE_FLASH_ATTN}_seqpara_${SEQ_PARALLEL}_\
            contiparam_${CONTI_PARAMS}_te_bg16_${TE_BF16}_${LABEL}.log"
TRAIN_LOG="${LOG_NAME}.log"
PROFILING_DIR="profile_${LOG_NAME}"

echo $TRAIN_LOG

if [[ $MODEL_SIZE -eq 7 ]]; then
        HIDDEN_SIZE=4096 # e.g. llama-13b: 5120
        FFN_HIDDEN_SIZE=11008 # e.g. llama-13b: 13824
        NUM_LAYERS=32 # e.g. llama-13b: 40
        NUM_HEADS=32 # e.g. llama-13b: 40
        SEQ_LENGTH=$SEQ_LENGTH
        MAX_POSITION_EMBEDDINGS=$MAX_POSITION_EMBEDDINGS
        NUM_KV_HEADS=32 # llama2 70B uses GQA
elif [[ $MODEL_SIZE -eq 13 ]]; then
        HIDDEN_SIZE=5120 # e.g. llama-13b: 5120
        FFN_HIDDEN_SIZE=13824 # e.g. llama-13b: 13824
        NUM_LAYERS=40 # e.g. llama-13b: 40
        NUM_HEADS=40 # e.g. llama-13b: 40
        SEQ_LENGTH=$SEQ_LENGTH
        MAX_POSITION_EMBEDDINGS=$MAX_POSITION_EMBEDDINGS
        NUM_KV_HEADS=40 # llama2 70B uses GQA
elif [[ $MODEL_SIZE -eq 20 ]]; then
        HIDDEN_SIZE=8192 # e.g. llama-13b: 5120
        FFN_HIDDEN_SIZE=28672 # e.g. llama-13b: 13824
        NUM_LAYERS=20 # e.g. llama-13b: 40
        NUM_HEADS=64 # e.g. llama-13b: 40
        NUM_KV_HEADS=8 # llama2 70B uses GQA
        SEQ_LENGTH=$SEQ_LENGTH
        MAX_POSITION_EMBEDDINGS=$MAX_POSITION_EMBEDDINGS
elif [[ $MODEL_SIZE -eq 70 ]]; then
        HIDDEN_SIZE=8192 # e.g. llama-13b: 5120
        FFN_HIDDEN_SIZE=28672 # e.g. llama-13b: 13824
        NUM_LAYERS=80 # e.g. llama-13b: 40
        NUM_HEADS=64 # e.g. llama-13b: 40
        NUM_KV_HEADS=8 # llama2 70B uses GQA
        SEQ_LENGTH=$SEQ_LENGTH
        MAX_POSITION_EMBEDDINGS=$MAX_POSITION_EMBEDDINGS
else
        echo "Model size not supported."
        exit 1
fi

GROUP_SIZE=$(( ${NUM_HEADS} / ${NUM_KV_HEADS} ))
NUM_GROUPS=$(( ${NUM_HEADS} / ${GROUP_SIZE} ))

GPT_ARGS="
    --tensor-model-parallel-size ${TP} \
    --pipeline-model-parallel-size ${PP} \
    --num-layers $NUM_LAYERS \
    --hidden-size $HIDDEN_SIZE \
    --ffn-hidden-size $FFN_HIDDEN_SIZE \
    --num-attention-heads $NUM_HEADS \
    --seq-length $SEQ_LENGTH \
    --max-position-embeddings $MAX_POSITION_EMBEDDINGS \
    --untie-embeddings-and-output-weights \
    --disable-bias-linear \
    --swiglu \
    --init-method-std 0.02 \
    --attention-dropout 0.0 \
    --hidden-dropout 0.0 \
    --normalization RMSNorm \
    --micro-batch-size $MBS \
    --global-batch-size $BS \
    --lr 3.0e-4 \
    --train-iters $TOTAL_ITERS \
    --lr-decay-style cosine \
    --min-lr 3.0e-5 \
    --weight-decay 1e-1 \
    --lr-warmup-fraction .01 \
    --optimizer $OPTIMIZER \
    --no-async-tensor-model-parallel-allreduce \
    --clip-grad 1.0 \
    --bf16 \
    --no-masked-softmax-fusion \
    --overlap-grad-reduce \
"
    # --no-masked-softmax-fusion \

DATA_ARGS="
    --tokenizer-type Llama2Tokenizer \
    --tokenizer-model ${TOKENIZER_MODEL} \
    --split 949,50,1 \
"

OUTPUT_ARGS="
    --log-interval 1 \
    --save-interval 1000 \
    --log-throughput \
    --no-save-optim \
    --eval-iters -1
"

    # --save-interval $TOTAL_ITERS \
    # --eval-interval $TOTAL_ITERS \

DISTRIBUTED_ARGS="
    --nproc_per_node $GPUS_PER_NODE \
    --nnodes $NNODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT
"

EXTRA_ARGS="
    --group-query-attention \
    --num-query-groups $NUM_GROUPS \
    --no-gradient-accumulation-fusion \
    --distributed-backend nccl \
    --distributed-timeout-minutes 30
"

if [ "$ENABLE_PROFILING" -eq 1 ]; then
EXTRA_ARGS="$EXTRA_ARGS --profile --use-pytorch-profiler --tensorboard-dir $LOG_DIR"
fi

if [ "$ADD_TASK" -eq 1 ]; then
EXTRA_ARGS="$EXTRA_ARGS --task gpt_chat"
fi


if [ "$ENABLE_MOCK_DATA" -eq 1 ]; then
EXTRA_ARGS="$EXTRA_ARGS --mock-data"
else
EXTRA_ARGS="$EXTRA_ARGS --data-path $DATA_PATH"
fi

if [ "$ENABLE_ROPE" -eq 1 ]; then
EXTRA_ARGS="$EXTRA_ARGS --position-embedding-type rope"
fi

if [ "$DISABLE_ROPE_TE" -eq 1 ]; then
EXTRA_ARGS="$EXTRA_ARGS --disable-te-fused-rope"
fi

if [ "$USE_FLASH_ATTN" -eq 1 ]; then
EXTRA_ARGS="$EXTRA_ARGS --use-flash-attn"
fi

if [ "$SEQ_PARALLEL" -eq 1 ]; then
EXTRA_ARGS="$EXTRA_ARGS --sequence-parallel"
fi

if [ "$CONTI_PARAMS" -eq 1 ]; then
EXTRA_ARGS="$EXTRA_ARGS --use-contiguous-parameters-in-local-ddp"
fi

if [ "$TE_BF16" -eq 1 ]; then
EXTRA_ARGS="$EXTRA_ARGS --transformer-impl=transformer_engine \
    --fp8-margin=0 \
    --fp8-interval=1 \
    --fp8-amax-history-len=1024 \
    --fp8-amax-compute-algo=max
"
fi

if [ "$DUMMY_RUN" -eq 0 ]; then
run_cmd="
    torchrun $DISTRIBUTED_ARGS pretrain_gpt.py \
        $GPT_ARGS \
        $DATA_ARGS \
        $OUTPUT_ARGS \
        $EXTRA_ARGS \
        --load $CHECKPOINT_PATH
"
else
run_cmd="
echo 'torchrun $DISTRIBUTED_ARGS pretrain_gpt.py \
        $GPT_ARGS \
        $DATA_ARGS \
        $OUTPUT_ARGS \
        $EXTRA_ARGS \
        --load $CHECKPOINT_PATH'
"
fi

run_cmd="$run_cmd |& tee $TRAIN_LOG"

if [ "$NO_TRAINING" -eq 0 ]; then
    eval $run_cmd
fi

MEAN_LOG_SCRIPT=examples/llama2/mean_log_value.py
TMP_FILE=${TMP_DIR}/tmp.txt
# echo '============================================================================================================'
grep -Eo 'throughput per GPU [^|]*' $TRAIN_LOG | sed -E 's/.*throughput per GPU \(TFLOP\/s\/GPU\): ([0-9\.]+).*/\1/' > $TMP_FILE
THROUGHPUT=$(python ${MEAN_LOG_SCRIPT} ${TMP_FILE})
echo "throughput per GPU (TFLOPs/GPU): ${THROUGHPUT}"
rm $TMP_FILE

# echo '============================================================================================================'
grep -Eo 'elapsed time per iteration [^|]*' $TRAIN_LOG | sed -E 's/.*elapsed time per iteration \(ms\): ([0-9\.]+).*/\1/' > $TMP_FILE
TIME_PER_ITER=$(python ${MEAN_LOG_SCRIPT}  ${TMP_FILE} 2>/dev/null | awk '{printf "%.6f", $0}')
echo "elapsed time per iteration: ${TIME_PER_ITER}"
rm $TMP_FILE

PERFORMANCE=$(awk -v bs="$BS" -v sl="$SEQ_LENGTH" -v tpi="$TIME_PER_ITER" -v ws="$WORLD_SIZE" 'BEGIN {printf "%.6f", bs * sl * 1000/ (tpi * ws)}')
echo "tokens/GPU/s: $PERFORMANCE"
