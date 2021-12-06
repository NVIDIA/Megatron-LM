#!/bin/bash

# Preparing the input file for the response generation (second-stage prompting)

DIR=`pwd`
python ${DIR}/tasks/knwl_dialo/preprocessing.py \
        --func prepare_input \
        --test_file <PATH_OF_THE_PROCESSED_TEST_DATA> \
        --knowledge_gen_file <PATH_OF_THE_GENERATED_KNOWLEDGE_DATA> \
        --processed_file <PATH_OF_THE_INPUT_FILE_FOR_RESPONSE_GENERATION>
