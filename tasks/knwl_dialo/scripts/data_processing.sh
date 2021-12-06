#!/bin/bash

# Data preparation for our framework: preprocessing the WoW and WoI datasets
# The datasets can be downloaded through the following links:
# WoW: https://parl.ai/projects/wizard_of_wikipedia/
# WoI: https://parl.ai/projects/sea/

DIR=`pwd`

# We provide the following script to process the raw data from Wizard of Wikipedia
python ${DIR}/tasks/knwl_dialo/preprocessing.py --func process_wow_dataset --raw_file <PATH_OF_THE_INPUT_DATA> --processed_file <PATH_OF_THE_OUTPUT_DATA> --knwl_ref_file <PATH_OF_THE_KNOWLEDGE_REFERENCE_OUTPUT_DATA> --resp_ref_file <PATH_OF_THE_RESPONSE_REFERENCE_OUTPUT_DATA>

# We provide the following script to process the raw data from Wizard of Internet
python ${DIR}/tasks/knwl_dialo/preprocessing.py --func process_woi_dataset --raw_file <PATH_OF_THE_INPUT_DATA> --processed_file <PATH_OF_THE_OUTPUT_DATA> --knwl_ref_file <PATH_OF_THE_KNOWLEDGE_REFERENCE_OUTPUT_DATA> --resp_ref_file <PATH_OF_THE_RESPONSE_REFERENCE_OUTPUT_DATA>

# Obtain the knowledge generation prompts
python ${DIR}/tasks/knwl_dialo/preprocessing.py --func get_knwl_gen_prompts --test_file <PATH_OF_THE_PROCESSED_TEST_DATA> --train_file <PATH_OF_THE_PROCESSED_TRAIN_DATA> --model_file <PATH_OF_THE_DPR_MODEL> --processed_file <PATH_OF_THE_OUTPUT_FILE> --data_type <DATA_TYPE_OF_THE_INPUT_FILE>

# Obtain the response generation prompts
python ${DIR}/tasks/knwl_dialo/preprocessing.py --func get_resp_gen_prompts --train_file <PATH_OF_THE_PROCESSED_TRAIN_DATA> --processed_file <PATH_OF_THE_OUTPUT_FILE>

