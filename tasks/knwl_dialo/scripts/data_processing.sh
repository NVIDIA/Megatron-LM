#!/bin/bash

# Data preparation for our framework: preprocessing the WoW and WoI datasets
# The datasets can be downloaded through the following links:
# WoW: https://parl.ai/projects/wizard_of_wikipedia/
# WoI: https://parl.ai/projects/sea/

DIR=`pwd`
mkdir ${DIR}/tasks/knwl_dialo/data
mkdir ${DIR}/tasks/knwl_dialo/data/wizard_of_wikipedia
mkdir ${DIR}/tasks/knwl_dialo/data/wizard_of_internet
# Before running the preprocessing, please download the datasets, 
# and put them into the corresponding created data folder.

# We provide examples for processing the raw data from Wizard of Wikipedia
python ${DIR}/tasks/knwl_dialo/preprocessing.py \
        --func process_wow_dataset \
        --raw_file ${DIR}/tasks/knwl_dialo/data/wizard_of_wikipedia/train.json \
        --processed_file <PATH_OF_THE_PROCESSED_WOW_TRAIN_DATA>

python ${DIR}/tasks/knwl_dialo/preprocessing.py \
        --func process_wow_dataset \
        --raw_file ${DIR}/tasks/knwl_dialo/data/wizard_of_wikipedia/test_random_split.json \
        --processed_file <PATH_OF_THE_PROCESSED_TEST_SEEN_DATA> \
        --knwl_ref_file <PATH_OF_THE_TEST_SEEN_KNOWLEDGE_REFERENCE_OUTPUT_DATA> \
        --resp_ref_file <PATH_OF_THE_TEST_SEEN_RESPONSE_REFERENCE_OUTPUT_DATA>

python ${DIR}/tasks/knwl_dialo/preprocessing.py \
        --func process_wow_dataset \
        --raw_file ${DIR}/tasks/knwl_dialo/data/wizard_of_wikipedia/test_topic_split.json \
        --processed_file <PATH_OF_THE_PROCESSED_TEST_UNSEEN_DATA> \
        --knwl_ref_file <PATH_OF_THE_TEST_UNSEEN_KNOWLEDGE_REFERENCE_OUTPUT_DATA> \
        --resp_ref_file <PATH_OF_THE_TEST_UNSEEN_RESPONSE_REFERENCE_OUTPUT_DATA>


# We provide the following script to process the raw data from Wizard of Internet
python ${DIR}/tasks/knwl_dialo/preprocessing.py \
        --func process_woi_dataset \
        --raw_file ${DIR}/tasks/knwl_dialo/data/wizard_of_internet/test.jsonl \
        --processed_file <PATH_OF_THE_PROCESSED_TEST_DATA> \
        --knwl_ref_file <PATH_OF_THE_TEST_KNOWLEDGE_REFERENCE_OUTPUT_DATA> \
        --resp_ref_file <PATH_OF_THE_TEST_RESPONSE_REFERENCE_OUTPUT_DATA>

# Obtain the knowledge generation prompts for each test dataset (Wizard of Wikipedia test seen/unseen and Wizard of Internet test)
python ${DIR}/tasks/knwl_dialo/preprocessing.py \
        --func get_knwl_gen_prompts \
        --test_file <PATH_OF_THE_PROCESSED_TEST_DATA> \
        --train_file <PATH_OF_THE_PROCESSED_WOW_TRAIN_DATA> \
        --model_file <PATH_OF_THE_DPR_MODEL> \
        --processed_file <PATH_OF_THE_OUTPUT_PROMPT_FILE> \
        --data_type <DATA_TYPE_OF_THE_INPUT_FILE>

# Obtain the response generation prompts
python ${DIR}/tasks/knwl_dialo/preprocessing.py \
        --func get_resp_gen_prompts \
        --train_file <PATH_OF_THE_PROCESSED_WOW_TRAIN_DATA> \
        --processed_file <PATH_OF_THE_OUTPUT_PROMPT_FILE>

