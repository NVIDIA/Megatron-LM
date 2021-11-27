#!/bin/bash

DIR=`pwd`
mkdir -p $DIR/tasks/knwl_dialo/data

# We provide the following script to process the raw data from Wizard of Wikipedia
python ${DIR}/tasks/knwl_dialo/preprocessing.py --func process_wow_dataset --input_file <PATH_OF_THE_INPUT_DATA> --output_file <PATH_OF_THE_OUTPUT_DATA>

# We provide the following script to process the raw data from Wizard of Internet
python ${DIR}/tasks/knwl_dialo/preprocessing.py --func process_woi_dataset --input_file <PATH_OF_THE_INPUT_DATA> --output_file <PATH_OF_THE_OUTPUT_DATA>

# Alternatively, we recommend you to directly download the already processed file through:
# wget 

