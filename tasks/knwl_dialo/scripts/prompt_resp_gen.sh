#!/bin/bash

# Stage-2: Prompt a pretrained language model to generate the corresponding response
# The input contains prompts, current dialogue context, and generated knowledge in Stage-1
# The output is the corresponding response.
# The size of the pretrained language model is 357M

WORLD_SIZE=8

DISTRIBUTED_ARGS="--nproc_per_node $WORLD_SIZE \
                  --nnodes 1 \
                  --node_rank 0 \
                  --master_addr localhost \
                  --master_port 6000"

CHECKPOINT_PATH=<Specify path for the language model>
INPUT_PATH=<Specific path for the input test dataset>
VOCAB_PATH=<Specify path for the vocab file>
MERGE_PATH=<Specify path for the merge file>
OUTPUT_PATH=<Speicifc path for the output>
PROMPT_PATH=<Specific path for the prompts>

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
        --prompt-type response \
        --num-prompt-examples 20 \
        --task KNWL-DIALO-PROMPT 
