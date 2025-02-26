#!/bin/bash

MEGATRON_LM_DIR=/iopsstor/scratch/cscs/$USER/Megatron-LM
SCRIPT_PATH=$MEGATRON_LM_DIR/tools/checkpoint/torch-dist/torchdist_2_torch.py

TORCHDIST_CKPT_PATH=/capstor/store/cscs/swissai/a06/users/schlag/llama3-1b-21n
TORCH_CKPT_SAVE_PATH=/iopsstor/scratch/cscs/$USER/Meg-Checkpoints/torch/llama3-1b-21n

# Use provided args if available
if [ "$#" -ne 0 ]; then
    TORCHDIST_CKPT_PATH=$1
    TORCH_CKPT_SAVE_PATH=$2
fi


CUDA_DEVICE_MAX_CONNECTIONS=1 torchrun $TORCH_DIST_SCRIPT \
    --bf16 \
    --load $CKPT_PATH \
    --ckpt-convert-save $TORCH_CKPT_SAVE_PATH