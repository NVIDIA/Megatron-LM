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
                  
# Required inputs. Set these in the environment before running this script.
: "${MODEL_GEN_PATH:?Set MODEL_GEN_PATH, e.g. /path/to/testseen_response_generations.txt}"
: "${GROUND_TRUTH_RESPONSE_PATH:?Set GROUND_TRUTH_RESPONSE_PATH, e.g. /path/to/testseen_response_reference.txt}"
: "${GROUND_TRUTH_KNOWLEDGE_PATH:?Set GROUND_TRUTH_KNOWLEDGE_PATH, e.g. /path/to/testseen_knowledge_reference.txt}"

python -m torch.distributed.launch $DISTRIBUTED_ARGS ./tasks/msdp/main.py \
        --num-layers 24 \
        --hidden-size 1024 \
        --num-attention-heads 16 \
        --seq-length 2048 \
        --max-position-embeddings 2048 \
        --micro-batch-size 4 \
        --task MSDP-EVAL-F1 \
        --guess-file "${MODEL_GEN_PATH}" \
        --answer-file "${GROUND_TRUTH_RESPONSE_PATH}"


##########################
# Evaluate the KF1 scores.
##########################
                  
python -m torch.distributed.launch $DISTRIBUTED_ARGS ./tasks/msdp/main.py \
        --num-layers 24 \
        --hidden-size 1024 \
        --num-attention-heads 16 \
        --seq-length 2048 \
        --max-position-embeddings 2048 \
        --micro-batch-size 4 \
        --task MSDP-EVAL-F1 \
        --guess-file "${MODEL_GEN_PATH}" \
        --answer-file "${GROUND_TRUTH_KNOWLEDGE_PATH}"


############################################
# Evaluate BLEU, METEOR, and ROUGE-L scores.
############################################

# We follow the nlg-eval (https://github.com/Maluuba/nlg-eval) to 
# evaluate the BLEU, METEOR, and ROUGE-L scores. 

# To evaluate on these metrics, please setup the environments based on 
# the nlg-eval github, and run the corresponding evaluation commands.

nlg-eval \
    --hypothesis="${MODEL_GEN_PATH}" \
    --references="${GROUND_TRUTH_RESPONSE_PATH}"
