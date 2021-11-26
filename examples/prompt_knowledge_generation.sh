#!/bin/bash

WORLD_SIZE=8

DISTRIBUTED_ARGS="--nproc_per_node $WORLD_SIZE \
                  --nnodes 1 \
                  --node_rank 0 \
                  --master_addr localhost \
                  --master_port 6000"

CHECKPOINT_PATH=<Specify path for the language model>
INPUT_PATH=<Specific path for the input test dataset>
OUTPUT_PATH=<Speicifc path for the output>
PROMPT_PATH=<Specific path for the prompts>

python -m torch.distributed.launch $DISTRIBUTED_ARGS ./tasks/main.py \
        --tensor-model-parallel-size 1 \
        --pipeline-model-parallel-size 1 \
        --num-layers 24 \
        --hidden-size 1024 \
        --num-attention-heads 16 \
        --seq-length 2048 \
        --max-position-embeddings 2048 \
        --micro-batch-size 1 \
        --vocab-file /gpfs/fs1/projects/gpu_adlr/datasets/nlp/gpt2_indexed_dataset/bpe/gpt2-vocab.json \
        --merge-file /gpfs/fs1/projects/gpu_adlr/datasets/nlp/gpt2_indexed_dataset/bpe/gpt2-merges.txt \
        --load ${CHECKPOINT_PATH} \
        --fp16 \
        --DDP-impl torch \
        --tokenizer-type GPT2BPETokenizer \
        --out-seq-length 100 \
        --sample-input-file ${INPUT_PATH} \
        --sample-output-file ${OUTPUT_PATH} \
        --prompt-file ${PROMPT_PATH} \
        --prompt-type knowledge \
        --num-prompt-examples 10 \
        --dynamic-prompt \
        --task knwl-dialo-prompt 
