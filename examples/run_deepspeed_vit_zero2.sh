#!/bin/bash
set -ex

#BASE_PATH=/vc_data/Megatron-LM/data
#DATA_PATH=${BASE_PATH}/indexed_datasets/megatron
DATA_PATH=/workspace/dataset_image
DS_CONFIG=ds_config.json

TP=1
PP=1
NLAYERS=24
HIDDEN=512

GLOBAL_BATCH=4
MICRO_BATCH=2

ZERO_STAGE=2
OFFLOAD_DEVICE="cpu"
CPU_OPTIM=" --cpu-optimizer"

OUTPUT_DIR=ds_z${ZERO_STAGE}_nl${NLAYERS}_hs${HIDDEN}_gb${GLOBAL_BATCH}_mb${MICRO_BATCH}
#OUTPUT_DIR=baseline_nl${NLAYERS}_hs${HIDDEN}_gb${GLOBAL_BATCH}_mb${MICRO_BATCH}
mkdir -p $OUTPUT_DIR

cat <<EOT > $DS_CONFIG
{
  "train_batch_size" : $GLOBAL_BATCH,
  "train_micro_batch_size_per_gpu": $MICRO_BATCH,
  "steps_per_print": 10,

  "zero_optimization": {
    "stage": $ZERO_STAGE,
    "stage3_max_live_parameters": 3e9,
    "stage3_max_reuse_distance": 3e9,
    "stage3_param_persistence_threshold": 1e5,
    "stage3_prefetch_bucket_size": 5e7,
    "contiguous_gradients": true,
    "overlap_comm": true,
    "reduce_bucket_size": 90000000,
    "sub_group_size": 1e8,
    "offload_optimizer": {
      "device": "$OFFLOAD_DEVICE",
      "buffer_count": 4,
      "pipeline_read": false,
      "pipeline_write": false,
      "pin_memory": true
    }

  },

  "fp16": {
    "enabled": true,
    "initial_scale_power": 12
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

ds_args=""
ds_args=" --deepspeed ${ds_args}"
ds_args=" --no-pipeline-parallel ${ds_args}" 
ds_args=" --deepspeed_config=$DS_CONFIG ${ds_args}"
ds_args=" --zero-stage=$ZERO_STAGE ${ds_args}"
ds_args=" --deepspeed-activation-checkpointing ${ds_args}"
export NCCL_SOCKET_IFNAME=eth0

deepspeed pretrain_vit.py \
    --tensor-model-parallel-size $TP \
    --pipeline-model-parallel-size $PP \
    --num-layers $NLAYERS \
    --hidden-size $HIDDEN \
    --num-attention-heads 16 \
    --seq-length 256 \
    --loss-scale 12 \
    --max-position-embeddings 1024 \
    --data-impl mmap \
    --micro-batch-size $MICRO_BATCH \
    --global-batch-size $GLOBAL_BATCH \
    --train-iters 1000 \
    --lr 6.0e-5 \
    --min-lr 6.0e-6 \
    --lr-decay-style cosine \
    --log-interval 1 \
    --eval-iters 40 \
    --eval-interval 1 \
    --data-path $DATA_PATH \
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

