#!/bin/bash
# This example will start serving the 769M dense model.
DISTRIBUTED_ARGS="--nproc_per_node 1 \
                  --nnodes 1 \
                  --node_rank 0 \
                  --master_addr localhost \
                  --master_port 6000"

CHECKPOINT=/workspace
VOCAB_FILE=/workspace/vocab.json
MERGE_FILE=/workspace/merges.txt

export CUDA_DEVICE_MAX_CONNECTIONS=1

pip install flask-restful

torchrun $DISTRIBUTED_ARGS /opt/Megatron-LM/tools/run_text_generation_server.py   \
       --num-layers 24  \
       --hidden-size 1536  \
       --load ${CHECKPOINT}  \
       --num-attention-heads 16  \
       --max-position-embeddings 1024  \
       --tokenizer-type GPT2BPETokenizer  \
       --fp16  \
       --micro-batch-size 16  \
       --seq-length 1024  \
       --temperature 1.0  \
       --vocab-file $VOCAB_FILE  \
       --merge-file $MERGE_FILE  \
       --top_p 0.9  \
       --seed 42 \
       --use-flash-attn \
       --hidden-dropout 0.0 \
       --attention-dropout 0.0 \
       --swiglu
