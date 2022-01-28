#!/bin/bash

# Preparing the input file for the response generation (second-stage prompting)

DIR=`pwd`

TEST_FILE=<PATH_OF_PROCESSED_TEST_DATA> \
        (e.g., /testseen_processed.txt)
KNOWLEDGE_FILE=<PATH_OF_GENERATED_KNOWLEDGE_DATA> \
        (e.g., /testseen_knowledge_generations.txt)
PROCESSED_FILE=<PATH_OF_INPUT_FILE_FOR_RESPONSE_GENERATION> \
        (e.g., /testseen_processed_with_generated_knowledge.txt)

python ${DIR}/tasks/msdp/preprocessing.py \
        --func prepare_input \
        --test_file ${TEST_FILE} \
        --knwl_gen_file ${KNOWLEDGE_FILE} \
        --processed_file ${PROCESSED_FILE}
