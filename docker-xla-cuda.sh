#!/bin/bash

docker run -t -d -v /home/ubuntu/efs/git/Megatron-LM-x:/megatron -v /tmp:/tmp -v /home/ubuntu/efs/data:/data \
    --shm-size=32g --net=host --gpus all docker.io/library/megatron-lm-x:latest  sleep infinity
