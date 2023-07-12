#!/bin/bash
set -ex

data_options=" \
         --vocab-file ${VOCAB_PATH} \
         --merge-file ${MERGE_PATH} \
         --data-path ${DATA_PATH} \
         --data-impl mmap"

BASE_PATH=$PWD/dataset/
DATA_PATH=${BASE_PATH}/BookCorpusDataset_text_document
DS_CONFIG=ds_config.json

# Hostfile path
HF=/job/hostfile

# Disabling tensor/pipeline parallelism
TP=1
PP=1

# HEADS ~= HIDDEN/128

# Refer to Megatron-table in the README.md file for model sizes
# Model: 310B
#NLAYERS=96
#HIDDEN=16384
#HEADS=128
#SEQ=2048

# Model 530B
#NLAYERS=105
#HIDDEN=20480
#HEADS=160
#SEQ=2048

# Model 1T
NLAYERS=128
HIDDEN=25600
HEADS=160
SEQ=1024

MICRO_BATCH=1
NODES=1
GPN=8
GLOBAL_BATCH=$(( ${GPN} * ${MICRO_BATCH} * ${NODES} ))

# Initial power scale for loss
SP=15

# Uncomment/comment one of the following blocks.

# For 1T model, start with microbatch=1, try to get 2 and 4.  If OOM w/ 4, use cpu-offloading

# Set to cpu for offloading to cpu for larger models
OFFLOAD_DEVICE="cpu"
CPU_OPTIM=" --cpu-optimizer"

# Set to none and empty string for no cpu offloading
#OFFLOAD_DEVICE="none"  
#CPU_OPTIM=" "

ZERO_STAGE=3
OUTPUT_DIR=ds_z_off-${OFFLOAD_DEVICE}_stage_${ZERO_STAGE}_nl${NLAYERS}_hs${HIDDEN}_mb${MICRO_BATCH}_seq${SEQ}_gb${GLOBAL_BATCH}_nodes${NODES}
#OUTPUT_DIR=baseline_nl${NLAYERS}_hs${HIDDEN}_gb${GLOBAL_BATCH}_mb${MICRO_BATCH}
mkdir -p $OUTPUT_DIR

cat <<EOT > $DS_CONFIG
{
  "train_batch_size" : $GLOBAL_BATCH,
  "train_micro_batch_size_per_gpu": $MICRO_BATCH,
  "steps_per_print": 1,
  "gradient_accumulation_steps": 1,
  "zero_optimization": {
    "stage": 3,
    "stage3_max_live_parameters": 3e9,
    "stage3_max_reuse_distance": 3e9,
    "stage3_param_persistence_threshold": 1e5,
    "stage3_prefetch_bucket_size": 5e7,
    "contiguous_gradients": true,
    "overlap_comm": true,
    "reduce_bucket_size": 90000000,
    "sub_group_size": 1e9,
    "offload_optimizer": {
      "device": "$OFFLOAD_DEVICE",
      "buffer_count": 4,
      "pipeline_read": false,
      "pipeline_write": false,
      "pin_memory": true
    }
  },
  "gradient_clipping": 1.0,
  "fp16": {
    "enabled": true,
    "initial_scale_power" : $SP,
    "loss_scale_window": 1000,
    "hysteresis": 2,
    "min_loss_scale": 1
  },
  "wall_clock_breakdown": true,
  "zero_allow_untested_optimizer": false,
  "aio": {
    "block_size": 1048576,
    "queue_depth": 16,
    "single_submit": false,
    "overlap_events": true,
    "thread_count": 2
  }
}
EOT

export NCCL_DEBUG=warn 

ds_args=" "
ds_args=" --deepspeed ${ds_args}"
ds_args=" --no-pipeline-parallel ${ds_args}" 
ds_args=" --deepspeed_config=$DS_CONFIG ${ds_args}"
ds_args=" --zero-stage=$ZERO_STAGE ${ds_args}"
ds_args=" --deepspeed-activation-checkpointing ${ds_args}"



deepspeed --force_multi --num_nodes=$NODES --hostfile $HF pretrain_gpt.py \
    --tensor-model-parallel-size $TP \
    --pipeline-model-parallel-size $PP \
    --num-layers $NLAYERS \
    --hidden-size $HIDDEN \
    --num-attention-heads $HEADS \
    --seq-length $SEQ \
    --loss-scale $SP \
    --max-position-embeddings $SEQ \
    --micro-batch-size $MICRO_BATCH \
    --global-batch-size $GLOBAL_BATCH \
    --train-iters 1000 \
    --lr 6.0e-5 \
    --min-lr 6.0e-6 \
    --lr-decay-style cosine \
    --log-interval 1 \
    --eval-iters 40 \
    --eval-interval 1000 \
    --data-path $DATA_PATH \
    --vocab-file $BASE_PATH/gpt2-vocab.json \
    --merge-file $BASE_PATH/gpt2-merges.txt \
    --save-interval 1000 \
    --split 98,2,0 \
    --clip-grad 1.0 \
    --weight-decay 0.1 \
    --adam-beta1 0.9 \
    --adam-beta2 0.95 \
    --init-method-std 0.006 \
    --fp16 \
    --checkpoint-activations \
    --tensorboard-dir $OUTPUT_DIR \
    $CPU_OPTIM $ds_args \
    --exit-interval 5000 | tee ${OUTPUT_DIR}/output.log
