#!/bin/bash

CHECKPOINT_PATH=checkpoints/gpt2_345m
VOCAB_FILE=gpt2-vocab.json
MERGE_FILE=gpt2-merges.txt
b=8
mp=1
gpus=1
experts=1

use_tutel=""
#use_tutel="--use-tutel"

#ds_inference=""
ds_inference="--ds-inference"

deepspeed --num_gpus=$gpus --num_nodes=1 tools/generate_samples_gpt.py \
       --tensor-model-parallel-size $mp \
       --num-layers 12 \
       --hidden-size 8192 \
       --load $CHECKPOINT_PATH \
       --num-attention-heads 32 \
       --max-position-embeddings 1024 \
       --tokenizer-type GPT2BPETokenizer \
       --fp16 \
       --num-experts $experts \
       --micro-batch-size $b \
       --seq-length 101 \
       --out-seq-length 101 \
       --temperature 1.0 \
       --vocab-file $VOCAB_FILE \
       --merge-file $MERGE_FILE \
       --genfile unconditional_samples.json \
       --top_p 0.9 \
       --log-interval 1 \
       --num-samples $((20*$b)) \
       $use_tutel $ds_inference

#$((10*$b))
#       --recompute
