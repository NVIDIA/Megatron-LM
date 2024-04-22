#!/bin/bash

# This script runs a Docker container with the necessary volume mounts for the PyTorch application.

docker run --ipc=host --shm-size=512m --gpus all -it --rm \
  -v /home/ubuntu/src/Megatron-LM:/workspace/megatron \
  -v /home/ubuntu/src/dataset-dir:/workspace/dataset \
  -v /home/ubuntu/src/checkpoint-dir:/workspace/checkpoints \
  my-pytorch-app \
  /bin/bash