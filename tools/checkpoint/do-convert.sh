#!/bin/bash


MEGATRON_LM_DIR=/iopsstor/scratch/cscs/$USER/Megatron-LM
CKPT_PATH=/capstor/store/cscs/swissai/a06/users/schlag/llama3-1b-21n


# Useful for: torch_dist -> torch
CKPT_IS_TORCH_DIST=true  # false if the checkpoint is already in torch format
TORCH_DIST_SCRIPT=$MEGATRON_LM_DIR/tools/checkpoint/torch-dist/torchdist_2_torch.py
TORCH_CKPT_SAVE_PATH=/iopsstor/scratch/cscs/$USER/Meg-Checkpoints/torch/llama3-1b-21n
# Useful for: core --> HF
HF_SAVE_DIR=/iopsstor/scratch/cscs/$USER/Meg-Checkpoints/hf-checkpoints
SAVE_DIR=$HF_SAVE_DIR/llama3-1b-21n
mkdir -p $HF_SAVE_DIR
LOADER=core
SAVER=llama_hf


# Run torch_dist --> torch
if [[ "$CKPT_IS_TORCH_DIST" == true ]]; then
    LOAD_DIR=$TORCH_CKPT_SAVE_PATH/torch
    echo "Running torch_dist --> torch conversion..."
    CUDA_DEVICE_MAX_CONNECTIONS=1 torchrun $TORCH_DIST_SCRIPT \
    --bf16 \
    --load $CKPT_PATH \
    --ckpt-convert-save $TORCH_CKPT_SAVE_PATH
else
    LOAD_DIR=$CKPT_PATH
    echo "Skipping torch_dist --> torch conversion..."
fi


# Run megatron core --> HF
echo "Running core --> HF conversion..."
python $MEGATRON_LM_DIR/tools/checkpoint/convert.py \
    --model-type GPT \
    --loader  $LOADER \
    --saver $SAVER \
    --load-dir $LOAD_DIR \
    --save-dir $SAVE_DIR \
    #\ --hf-tokenizer .....