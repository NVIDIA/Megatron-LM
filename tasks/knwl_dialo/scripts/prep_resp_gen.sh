#!/bin/bash

# Preparing the input file for the response generation (second-stage prompting)

DIR=`pwd`
python ${DIR}/tasks/knwl_dialo/preprocessing.py --func prepare_input --test_file <PATH_OF_THE_PROCESSED_TEST_DATA> --knowledge_file <PATH_OF_THE_GENERATED_KNOWLEDGE_DATA> --output_file <PATH_OF_THE_OUTPUT_FILE>
