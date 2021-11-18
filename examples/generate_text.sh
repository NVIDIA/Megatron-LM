#!/bin/bash

CHECKPOINT_PATH=checkpoints/gpt2_345m
VOCAB_FILE=gpt2-vocab.json
MERGE_FILE=gpt2-merges.txt
b=1
mp=1

deepspeed --num_gpus=$(($mp)) --num_nodes=1 tools/generate_samples_gpt.py \
       --tensor-model-parallel-size $mp \
       --num-layers 4 \
       --hidden-size 1024 \
       --load $CHECKPOINT_PATH \
       --num-attention-heads 16 \
       --max-position-embeddings 1024 \
       --tokenizer-type GPT2BPETokenizer \
       --fp16 \
       --num-experts 8 \
       --micro-batch-size $b \
       --seq-length 101 \
       --out-seq-length 101 \
       --temperature 1.0 \
       --vocab-file $VOCAB_FILE \
       --merge-file $MERGE_FILE \
       --genfile unconditional_samples.json \
       --top_p 0.9 \
       --log-interval 1 \
       --num-samples 1 # $((10*$b))
#       --recompute
