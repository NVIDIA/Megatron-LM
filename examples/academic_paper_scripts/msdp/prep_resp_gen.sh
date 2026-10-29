#!/bin/bash

# Preparing the input file for the response generation (second-stage prompting)

DIR=`pwd`

# Required inputs. Set these in the environment before running this script.
: "${TEST_FILE:?Set TEST_FILE, e.g. /path/to/testseen_processed.txt}"
: "${KNOWLEDGE_FILE:?Set KNOWLEDGE_FILE, e.g. /path/to/testseen_knowledge_generations.txt}"
: "${PROCESSED_FILE:?Set PROCESSED_FILE, e.g. /path/to/testseen_processed_with_generated_knowledge.txt}"

python ${DIR}/tasks/msdp/preprocessing.py \
        --func prepare_input \
        --test_file "${TEST_FILE}" \
        --knwl_gen_file "${KNOWLEDGE_FILE}" \
        --processed_file "${PROCESSED_FILE}"
