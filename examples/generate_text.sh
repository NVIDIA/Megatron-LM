#!/bin/bash

CHECKPOINT_PATH=checkpoints/gpt2_345m
VOCAB_FILE=gpt2-vocab.json
MERGE_FILE=gpt2-merges.txt
b=8
mp=2
gpus=2
experts=8

deepspeed --num_gpus=$gpus --num_nodes=1 tools/generate_samples_gpt.py \
       --tensor-model-parallel-size $mp \
       --num-layers 2 \
       --hidden-size 1024 \
       --load $CHECKPOINT_PATH \
       --num-attention-heads 16 \
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
       --num-samples $((10*$b)) \
       --ds-inference \

#$((10*$b))

#       --recompute
