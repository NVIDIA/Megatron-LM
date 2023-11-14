#!/bin/bash
# This example will start serving the 1B model.
# You may need to adapt Flask port if it's occupied in MegatronServer class, we chnaged it from 5000 (default) to 8080
DISTRIBUTED_ARGS="--nproc_per_node 1 \
                  --nnodes 1 \
                  --node_rank 0 \
                  --master_addr ip-26-0-156-56 \
                  --master_port 6000"


CHECKPOINT=/fsx/loubna/data/extra/generations_starcoder2_1b_200k/megatron

#/mp_rank_00/model_optim_rng.pt
VOCAB_FILE=/fsx/bigcode/experiments/pretraining/starcoder2-1B/checkpoints/conversions/vocab.json
MERGE_FILE=/fsx/bigcode/experiments/pretraining/starcoder2-1B/checkpoints/conversions/merges.txt
TOKENIZER_FILE=/fsx/loubna/data/tokenizer/starcoder2-smol-internal-1/tokenizer.json 

export CUDA_DEVICE_MAX_CONNECTIONS=1

#pip install flask-restful

torchrun $DISTRIBUTED_ARGS tools/run_text_generation_server.py   \
       --tensor-model-parallel-size 1 \
       --pipeline-model-parallel-size 1 \
       --num-layers 24 \
       --hidden-size 2048 \
       --num-attention-heads 16 \
       --attention-head-type multiquery \
       --init-method-std 0.02209 \
       --seq-length 4096 \
       --use-rotary-position-embeddings \
       --max-position-embeddings 4096 \
       --rotary-theta 100000 \
       --attention-dropout 0.1 \
       --hidden-dropout 0.1 \
       --load ${CHECKPOINT}  \
       --tokenizer-type TokenizerFromFile \
       --tokenizer-file $TOKENIZER_FILE \
       --bf16  \
       --micro-batch-size 1  \
       --seq-length 1024  \
       --out-seq-length 512  \
       --temperature 0  \
       --top_p 0.9  \
       --seed 42
       --output_file 
