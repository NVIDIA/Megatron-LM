#!/bin/bash

# Stage-1: Prompt a pretrained language model to generate the context-relevant knowledge
# The input contains prompts and current dialogue context, the output is the relevant knowledge
# The size of the pretrained language model is 357M

WORLD_SIZE=8

DISTRIBUTED_ARGS="--nproc_per_node $WORLD_SIZE \
                  --nnodes 1 \
                  --node_rank 0 \
                  --master_addr localhost \
                  --master_port 6000"

CHECKPOINT_PATH=<PATH_OF_THE_LANGUAGE_MODEL>
INPUT_PATH=<PATH_OF_THE_INPUT_TEST_DATA_FILE>
PROMPT_PATH=<PATH_OF_THE_KNOWLEDGE_GENERATION_PROMPTS>
VOCAB_PATH=<PATH_OF_THE_VOCAB_FILE>
MERGE_PATH=<PATH_OF_THE_MERGE_FILE>
OUTPUT_PATH=<PATH_OF_THE_OUTPUT_GENERATION_FILE>

python -m torch.distributed.launch $DISTRIBUTED_ARGS ./tasks/main.py \
        --num-layers 24 \
        --hidden-size 1024 \
        --num-attention-heads 16 \
        --seq-length 2048 \
        --max-position-embeddings 2048 \
        --micro-batch-size 1 \
        --vocab-file ${VOCAB_PATH} \
        --merge-file ${MERGE_PATH} \
        --load ${CHECKPOINT_PATH} \
        --fp16 \
        --DDP-impl torch \
        --tokenizer-type GPT2BPETokenizer \
        --sample-input-file ${INPUT_PATH} \
        --sample-output-file ${OUTPUT_PATH} \
        --prompt-file ${PROMPT_PATH} \
        --prompt-type knowledge \
        --num-prompt-examples 10 \
        --task KNWL-DIALO-PROMPT 
