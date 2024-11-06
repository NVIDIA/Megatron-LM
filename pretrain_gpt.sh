#!/bin/bash


#SBATCH <SLURM OPTIONS> --nodes=128 --exclusive --ntasks-per-node=8 --job-name=megatron_gpt3_175b

export CUDA_DEVICE_MAX_CONNECTIONS=8

DIR=`pwd`
DATETIME=`date +'date_%y-%m-%d_time_%H-%M-%S'`
mkdir -p $DIR/logs

# VOCAB_SIZE is either 32k, 64k, 128k or 256k
if [ -z "$VOCAB_SIZE" ]; then
  VOCAB_SIZE=256k
fi

DATASET="/tmp/vp_sample_dataset_v${VOCAB_SIZE}/dataset/c4_text_document"
TOKENIZER="/tmp/vp_sample_dataset_v${VOCAB_SIZE}/tokenizer/vp_sample_dataset.model"

if [ ! -e "$DATASET"".idx" ]; then
  wget https://huggingface.co/datasets/mtyeung/vocab_parallel_sample_dataset/resolve/main/vp_sample_dataset_v${VOCAB_SIZE}.tar.gz
  tar -xvf vp_sample_dataset_v${VOCAB_SIZE}.tar.gz -C /tmp
fi

# Running locally
if [ -z "$WORLD_SIZE" ]; then
  export WORLD_SIZE=1
  export RANK=0
  export MASTER_ADDR=localhost
  export MASTER_PORT=10086
fi

if [ -z "$GPUS_PER_NODE" ]; then
  GPUS_PER_NODE=$(nvidia-smi --list-gpus | wc -l)
fi

if [ -z "$EXIT_INTERVAL" ]; then
  EXIT_INTERVAL=1000
fi

WORLD_SIZE_IN_GPUS=$(( $WORLD_SIZE * $GPUS_PER_NODE ))

if [ -z "$PIPELINE_SIZE" ]; then
  PIPELINE_SIZE=$(( $WORLD_SIZE_IN_GPUS))
  LAYERS=$(( $PIPELINE_SIZE * 4))
  MICRO_BATCH_SIZE=1
  GLOBAL_BATCH_SIZE=$(( $PIPELINE_SIZE * 3 * $MICRO_BATCH_SIZE ))
  HIDDEN_SIZE=4096
  ATTENTION_HEADS=32
fi

profile_ranks="0"
for ((i = 1; i < $WORLD_SIZE_IN_GPUS; i++)); do
    profile_ranks="$profile_ranks $i"
done
if [ -z "$ZERO_BUBBLE_TIMER_START" ]; then
  ZERO_BUBBLE_TIMER_START=100
  ZERO_BUBBLE_TIMER_END=110
fi

if [ -z "$EVAL_INTERVAL" ]; then
  EVAL_INTERVAL=10000
fi

if [ -z "$TP_SIZE" ]; then
  TP_SIZE=1
fi

if [ -z "$SEQ_LENGTH" ]; then
  SEQ_LENGTH=2048
fi

if [ -z "$IMM_SIZE" ]; then
  IMM_SIZE=$(( 4 * $HIDDEN_SIZE ))
fi


options=" \
  --tensor-model-parallel-size $TP_SIZE \
  --pipeline-model-parallel-size $PIPELINE_SIZE \
  --num-layers $LAYERS \
  --hidden-size $HIDDEN_SIZE \
  --ffn-hidden-size $IMM_SIZE \
  --num-attention-heads $ATTENTION_HEADS \
  --exit-interval $EXIT_INTERVAL \
  --seq-length $SEQ_LENGTH \
  --max-position-embeddings $SEQ_LENGTH \
  --micro-batch-size $MICRO_BATCH_SIZE \
  --global-batch-size $GLOBAL_BATCH_SIZE \
  --train-samples 146484375 \
  --lr-decay-samples 126953125 \
  --lr-warmup-samples 183105 \
  --lr 6.0e-5 \
  --min-lr 6.0e-6 \
  --lr-decay-style cosine \
  --log-interval 10 \
  --eval-iters 40 \
  --eval-interval $EVAL_INTERVAL \
  --data-path ${DATASET} \
  --tokenizer-type GPTSentencePieceTokenizer \
  --tokenizer-model ${TOKENIZER} \
  --split 98,2,0 \
  --clip-grad 8.0 \
  --weight-decay 0.1 \
  --adam-beta1 0.9 \
  --adam-beta2 0.95 \
  --init-method-std 0.006 \
  --no-barrier-with-level-1-timing \
  --profile-step-start 22 \
  --profile-step-end 23 \
  --profile-ranks $profile_ranks \
  --use-flash-attn \
  --sequence-parallel \
  --untie-embeddings-and-output-weights \
  --attention-dropout 0 \
  --hidden-dropout 0 \
  --use-cpu-initialization \
  --use-distributed-optimizer \
  --initial-loss-scale 65536 \
  --no-create-attention-mask-in-dataloader"

if [ -z "$FP32" ]; then
  options="$options --fp16"
fi

if [ ! -z "$PROFILED" ]; then
  options="$options --profile"
fi

if [ ! -z "$VOCAB_PARALLEL" ]; then
  options="$options --enable-vocab-parallel"
  if [ ! -z "$INTERLACED_SCHEDULE" ]; then
    options="$options --use-interlaced-schedule"
  fi
  if [ ! -z "$FB_SPLIT" ]; then
    options="$options --disable-backward-fusion"
  fi
fi

if [ ! -z "$ENABLE_LAYER_REDISTRIBUTION" ]; then
  options="$options --enable-layer-redistribution \
  --final-stage-num-layers $FINAL_STAGE_LAYERS"
fi

run_cmd="torchrun --nnodes $WORLD_SIZE \
  --node_rank $RANK \
  --master_addr $MASTER_ADDR \
  --master_port $MASTER_PORT \
  --nproc_per_node=$GPUS_PER_NODE ${DIR}/pretrain_gpt.py $@ ${options}"

if [ ! -z "$PROFILED" ]; then
  run_cmd="nsys profile -s none -t nvtx,cuda \
    --output $AIP_RUN_NAME.$RANK.nsys-rep \
    --force-overwrite true \
    --capture-range=cudaProfilerApi \
    --capture-range-end=stop \
    $run_cmd"
fi

echo $run_cmd
# sleep 100000
eval $run_cmd

set +x
