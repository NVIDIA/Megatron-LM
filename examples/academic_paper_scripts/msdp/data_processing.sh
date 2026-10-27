#!/bin/bash

# Data preparation for our framework: preprocessing the WoW and WoI datasets
# The datasets can be downloaded through the following links:
# WoW: https://parl.ai/projects/wizard_of_wikipedia/
# WoI: https://parl.ai/projects/sea/

DIR=`pwd`
# Before running the preprocessing, please download 
# the wizard of wikipedia and wizard datasets
WOW_DATA_FOLDER=<PATH_OF_WIZARD_OF_WIKIPEDIA_DATA_FOLDER>
WOI_DATA_FOLDER=<PATH_OF_WIZARD_OF_INTERNET_DATA_FOLDER>

# We provide examples for processing the raw data from Wizard of Wikipedia
# Processing the train dataset (train.json)
python ${DIR}/tasks/msdp/preprocessing.py \
        --func process_wow_dataset \
        --raw_file ${WOW_DATA_FOLDER}/train.json \
        --processed_file ${WOW_DATA_FOLDER}/train_processed.txt

# Processing test seen dataset (test_random_split.json)
python ${DIR}/tasks/msdp/preprocessing.py \
        --func process_wow_dataset \
        --raw_file ${WOW_DATA_FOLDER}/test_random_split.json \
        --processed_file ${WOW_DATA_FOLDER}/testseen_processed.txt \
        --knwl_ref_file ${WOW_DATA_FOLDER}/output_testseen_knowledge_reference.txt \
        --resp_ref_file ${WOW_DATA_FOLDER}/output_testseen_response_reference.txt

# processing test unseen dataset (test_topic_split.json)
python ${DIR}/tasks/msdp/preprocessing.py \
        --func process_wow_dataset \
        --raw_file ${WOW_DATA_FOLDER}/test_topic_split.json \
        --processed_file ${WOW_DATA_FOLDER}/testunseen_processed.txt \
        --knwl_ref_file ${WOW_DATA_FOLDER}/output_testunseen_knowledge_reference.txt \
        --resp_ref_file ${WOW_DATA_FOLDER}/output_testunseen_response_reference.txt


# We provide the following script to process the raw data from Wizard of Internet
# Processing the test dataset (test.jsonl)
python ${DIR}/tasks/msdp/preprocessing.py \
        --func process_woi_dataset \
        --raw_file ${WOI_DATA_FOLDER}/test.jsonl \
        --processed_file ${WOI_DATA_FOLDER}/test_processed.txt \
        --knwl_ref_file ${WOI_DATA_FOLDER}/output_test_knowledge_reference.txt \
        --resp_ref_file ${WOI_DATA_FOLDER}/output_test_response_reference.txt


# Get the knowledge generation prompts for the each test dataset in WoW and WoI
MODEL_FILE=<PATH_OF_THE_FINETUNED_DPR_MODEL> 
# WoW test seen
python ${DIR}/tasks/msdp/preprocessing.py \
        --func get_knwl_gen_prompts \
        --test_file ${WOW_DATA_FOLDER}/testseen_processed.txt \
        --train_file ${WOW_DATA_FOLDER}/train_processed.txt \
        --model_file ${MODEL_FILE} \
        --processed_file ${WOW_DATA_FOLDER}/output_testseen_knowledge_prompts.json \
        --data_type wow_seen

# WoW test unseen
python ${DIR}/tasks/msdp/preprocessing.py \
        --func get_knwl_gen_prompts \
        --test_file ${WOW_DATA_FOLDER}/testunseen_processed.txt \
        --train_file ${WOW_DATA_FOLDER}/train_processed.txt \
        --model_file ${MODEL_FILE} \
        --processed_file ${WOW_DATA_FOLDER}/output_testunseen_knowledge_prompts.json \
        --data_type wow_unseen

# WoI
python ${DIR}/tasks/msdp/preprocessing.py \
        --func get_knwl_gen_prompts \
        --test_file ${WOI_DATA_FOLDER}/test_processed.txt \
        --train_file ${WOW_DATA_FOLDER}/train_processed.txt \
        --model_file ${MODEL_FILE} \
        --processed_file ${WOI_DATA_FOLDER}/output_test_knowledge_prompts.json \
        --data_type woi


# Get the response generation prompts (can be applied for all the test datasets)
python ${DIR}/tasks/msdp/preprocessing.py \
        --func get_resp_gen_prompts \
        --train_file ${WOW_DATA_FOLDER}/train_processed.txt \
        --processed_file ${WOW_DATA_FOLDER}/output_response_prompts.txt

