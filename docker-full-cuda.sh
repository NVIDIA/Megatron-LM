#!/bin/bash

docker buildx build -t megatron-lm-x:full-cuda -f Dockerfile.full-cuda .
docker run -t -d \
    -v /home/ubuntu/efs/git/Megatron-LM-x:/workspace/megatron \
    -v /tmp:/tmp \
    -v /home/ubuntu/efs/data:/workspace/data \
    -v /home/ubuntu/efs/datasets:/workspace/datasets \
    --shm-size=32g --net=host --gpus all megatron-lm-x:full-cuda  sleep infinity
