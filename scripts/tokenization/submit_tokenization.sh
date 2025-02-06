#!/bin/bash

# ⚠️ WARNING ⚠️
# Make sure to prepare the dumps before tokenizing the data!
# Check scripts/tokenization/prepare_dumps.py
# ⚠️ WARNING ⚠️

NUMBER_OF_DATATROVE_TASKS=20
TOKENIZER=meta-llama/Llama-3.1-70B
TOKENIZER_NAME=Llama-3.1-70B
DATASET_NAME=fineweb-2

MEGATRON_LM_DIR=/capstor/scratch/cscs/$USER/Megatron-LM
PATH_TO_PREPROCESSING_METADATA=$MEGATRON_LM_DIR/datasets/$DATASET_NAME
PATH_TO_DATATROVE_LOGGING_DIR=$MEGATRON_LM_DIR/logs/datatrove
PATH_TO_SLURM_LOGGING_DIR=$MEGATRON_LM_DIR/logs/slurm/tokenization-$TOKENIZER_NAME-$DATASET_NAME
PATH_TO_OUTPUT_FOLDER=/iopsstor/scratch/cscs/$USER/datasets

DATASET_OUTPUT_FOLDER_NAME=$PATH_TO_OUTPUT_FOLDER/$TOKENIZER_NAME/$DATASET_NAME
CSV_RESULTS_FILE=$PATH_TO_PREPROCESSING_METADATA/tokenize-$TOKENIZER_NAME-$DATASET_NAME.csv

mkdir -p $DATASET_OUTPUT_FOLDER_NAME
mkdir -p $PATH_TO_SLURM_LOGGING_DIR
mkdir -p $PATH_TO_PREPROCESSING_METADATA/completed_dumps
ln -sfn $DATASET_OUTPUT_FOLDER_NAME $PATH_TO_PREPROCESSING_METADATA/tokenized-dir-link

echo "slurm_job_id,node,start,end,paths_file,output_folder,dataset_total_size,processed_total_size,number_of_workers_per_node,time,bw,total_tokens_processed,throughput (Million Tokens/Second/Node)" > $CSV_RESULTS_FILE
# Iterate through all dumps paths files
for paths_file in "$PATH_TO_PREPROCESSING_METADATA/dumps"/*; do
  dump=$(grep -oP '(?<=paths_file_)\d+(?=\.txt)' <<< $paths_file)
  output_folder=$DATASET_OUTPUT_FOLDER_NAME/dump_$dump
  logging_dir=$PATH_TO_DATATROVE_LOGGING_DIR/$TOKENIZER_NAME/$DATASET_NAME/dump_$dump
  sbatch --job-name=tokenize-$DATASET_NAME --output=$PATH_TO_SLURM_LOGGING_DIR/R-%x-%j.out --error=$PATH_TO_SLURM_LOGGING_DIR/R-%x-%j.err $MEGATRON_LM_DIR/scripts/tokenization/tokenize.sh $PATH_TO_PREPROCESSING_METADATA/raw-dataset-link $output_folder $TOKENIZER $logging_dir $CSV_RESULTS_FILE $paths_file $NUMBER_OF_DATATROVE_TASKS $MEGATRON_LM_DIR
done
