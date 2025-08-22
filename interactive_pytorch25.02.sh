#!/bin/bash

# H100
CONTAINER_IMAGE="/lustre/fsw/portfolios/llmservice/users/trintamaki/workspace/containers/megatron-dev-img-03232025-pytorch25.02-te-main-160be21-energon-develop-bd46613.sqsh"
CONTAINER_IMAGE="/lustre/fsw/portfolios/llmservice/users/trintamaki/workspace/containers/megatron-dev-img-03232025-pytorch25.02-te-main-160be21-energon-main-6.0.1.sqsh"
CONTAINER_IMAGE="/lustre/fsw/portfolios/llmservice/users/trintamaki/workspace/containers/megatron-dev-img-03232025-pytorch25.02-te-main-160be21-energon-develop-max-pr.sqsh"
CONTAINER_IMAGE="/lustre/fsw/portfolios/llmservice/users/trintamaki/workspace/containers/megatron-dev-img-03232025-pytorch25.02-te-main-160be21-energon-develop-max-pr-onelogger.sqsh"
CONTAINER_IMAGE="/lustre/fsw/portfolios/llmservice/users/trintamaki/workspace/containers/megatron-dev-img-03232025-pytorch25.02-te-main-160be21-energon-develop-max-pr-onelogger-max-pr2.sqsh"
CONTAINER_IMAGE="/lustre/fsw/portfolios/coreai/users/pmannan/workspace/megatron_vlm_25.02-te_api.sqsh"

# Set partitions based on hostname
if [[ $(hostname) == *"oci-iad"* ]]; then
    PARTITIONS="interactive,batch_singlenode,backfill_singlenode,backfill_block1,backfill_block3,backfill_block4,batch_block1,batch_block2,batch_block3,batch_block4"
elif [[ $(hostname) == *"cw-dfw"* ]]; then
    PARTITIONS="interactive,batch"
elif [[ $(hostname) == *"oci-nrt"* ]]; then
    PARTITIONS="interactive,batch_block1,backfill,batch_singlenode"
else
    PARTITIONS="interactive"
fi

srun -p ${PARTITIONS} -A coreai_dlalgo_genai -N 1 --pty \
    --container-image /lustre/fsw/portfolios/llmservice/users/matthieul/docker/megatron-dev-img-05142025-pytorch-dev-te-cd37379-editable-energon-mamba-fix-vlmeval-pad-conv.sqsh \
    --container-mounts "/lustre" \
    --gpus 8 \
    --exclusive \
    --job-name "coreai_dlalgo_genai-megatron-dev:interactive" \
    -t 1:00:00 \
    bash -l