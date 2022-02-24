#!/bin/bash

#########################
# Evaluate the F1 scores.
#########################

WORLD_SIZE=1
DISTRIBUTED_ARGS="--nproc_per_node $WORLD_SIZE \
                  --nnodes 1 \
                  --node_rank 0 \
                  --master_addr localhost \
                  --master_port 6000"
                  
MODEL_GEN_PATH=<PATH_OF_THE_KNOWLEDGE_GENERATION> \ 
        (e.g., /testseen_knowledge_generations.txt)
GROUND_TRUTH_PATH=<PATH_OF_THE_GROUND_TRUTH_KNOWLEDGE> \ 
        (e.g., /testseen_knowledge_reference.txt)

python -m torch.distributed.launch $DISTRIBUTED_ARGS ./tasks/msdp/main.py \
        --num-layers 24 \
        --hidden-size 1024 \
        --num-attention-heads 16 \
        --seq-length 2048 \
        --max-position-embeddings 2048 \
        --micro-batch-size 4 \
        --task MSDP-EVAL-F1 \
        --guess-file ${MODEL_GEN_PATH} \
        --answer-file ${GROUND_TRUTH_PATH}


############################################
# Evaluate BLEU, METEOR, and ROUGE-L scores.
############################################

# We follow the nlg-eval (https://github.com/Maluuba/nlg-eval) to 
# evaluate the BLEU, METEOR, and ROUGE-L scores. 

# To evaluate on these metrics, please setup the environments based on 
# the nlg-eval github, and run the corresponding evaluation commands.

nlg-eval \
    --hypothesis=<PATH_OF_THE_KNOWLEDGE_GENERATION> \
    --references=<PATH_OF_THE_GROUND_TRUTH_KNOWLEDGE>
