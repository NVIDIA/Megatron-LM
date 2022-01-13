#!/bin/bash
export TORCH_CUDA_ARCH_LIST=8.6+PTX
CHECKPOINT_PATH=checkpoints/gpt2_345m
VOCAB_FILE=gpt2-vocab.json
MERGE_FILE=gpt2-merges.txt
b=8
mp=16
experts=128
nodes=1
gpus=8

use_tutel=""
#use_tutel="--use-tutel"

ds_inference=""
ds_inference="--ds-inference"
numa_bind=""
numa_bind="--bind-to numa"
experts=128
NUM_LAYERS=(40)
HIDDEN=(4096)
HEADS=(32)
NODES=(4)
for ns in ${!NODES[@]};
do
for mp in 4
do
for k in ${!NUM_LAYERS[@]};
do

nodes=${NODES[$ns]}
procs=$(($nodes * $gpus))
launch_cmd="deepspeed --num_nodes $nodes --num_gpus 8"
L=${NUM_LAYERS[$k]}
H=${HIDDEN[$k]}
A=${HEADS[$k]}
#experts1=${experts[$k]}
program_cmd="tools/generate_samples_gpt.py \
       --tensor-model-parallel-size $mp \
       --num-layers $L \
       --hidden-size $H \
       --load $CHECKPOINT_PATH \
       --num-attention-heads $A \
       --max-position-embeddings 1024 \
       --tokenizer-type GPT2BPETokenizer \
       --fp16 \
       --num-experts ${experts} \
       --mlp-type standard \
       --micro-batch-size $b \
       --seq-length 10 \
       --out-seq-length 10 \
       --temperature 1.0 \
       --vocab-file $VOCAB_FILE \
       --merge-file $MERGE_FILE \
       --genfile unconditional_samples.json \
       --top_p 0.9 \
       --log-interval 1 \
       --num-samples $((100*$b))
       --deepspeed \
       $use_tutel $ds_inference"

echo $launch_cmd $program_cmd

$launch_cmd $program_cmd
done
done
done
