#!/bin/bash

# Preparing the input file for the response generation (second-stage prompting)

DIR=`pwd`

TEST_FILE=<PATH_OF_THE_PROCESSED_TEST_DATA>
KNOWLEDGE_FILE=<PATH_OF_THE_GENERATED_KNOWLEDGE_DATA>
PROCESSED_FILE=<PATH_OF_THE_INPUT_FILE_FOR_RESPONSE_GENERATION>

python ${DIR}/tasks/knwl_dialo/preprocessing.py \
        --func prepare_input \
        --test_file ${TEST_FILE} \
        --knowledge_gen_file ${KNOWLEDGE_FILE} \
        --processed_file ${PROCESSED_FILE}
