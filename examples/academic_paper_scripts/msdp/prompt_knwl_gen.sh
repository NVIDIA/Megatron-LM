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

CHECKPOINT_PATH=<PATH_OF_LANGUAGE_MODEL> (e.g., /357m)
VOCAB_PATH=<PATH_OF_VOCAB_FILE> (e.g., /gpt2-vocab.json)
MERGE_PATH=<PATH_OF_MERGE_FILE> (e.g., /gpt2-merges.txt)
INPUT_PATH=<PATH_OF_PROCESSED_TEST_DATA_FILE> \ 
        (e.g., /testseen_processed.txt)
PROMPT_PATH=<PATH_OF_KNOWLEDGE_GENERATION_PROMPTS> \
        (e.g., /testseen_knowledge_prompts.json)
OUTPUT_PATH=<PATH_OF_OUTPUT_GENERATION_FILE> \
        (e.g., /testseen_knowledge_generations.txt)

python -m torch.distributed.launch $DISTRIBUTED_ARGS ./tasks/msdp/main.py \
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
        --task MSDP-PROMPT 

# NOTE: If you use api for the model generation, please use 
# the "--api-prompt" flag (setting this value as True). 
