#!/bin/bash

HF_MODEL_CKPT=/workspace/scratch/moonshotai/Kimi-K2-Instruct

MLM_EXTRA_ARGS=" \
    --decoder-first-pipeline-num-layers 3 \
    --decoder-last-pipeline-num-layers 2 \
    --init-model-with-meta-device \
    --use-cpu-initialization \

"

# Layer distribution over PP: 3, [4] * 14, 2.
PP=16

