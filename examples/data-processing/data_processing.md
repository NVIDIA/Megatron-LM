# Tokenize

## Overview

The provided Python script is designed to facilitate the submission of tokenization jobs to Azure Batch for large-scale data processing. It supports both local and Azure compute targets for tokenization tasks. This document will provide an overview of the script's functionality, its command-line arguments, and usage instructions.

## Prerequisites

Before using this script, ensure that you have the following prerequisites in place:

1. Azure Subscription: You must have an active Azure subscription with access to Azure Batch and Blob Storage.

2. Azure Configuration: You should have the necessary Azure configuration details such as subscription ID, resource group, workspace name, and SAS token if applicable.

3. Input Data: Prepare the input data in JSONL format that you want to tokenize. The script processes each JSONL file as a separate job. You should put all the data inside one single folder and use that folder as the input data path. 

4. Tokenizer Model: You should have a pre-trained tokenizer model and related files (e.g., vocab file, merge file) if required for tokenization. Currently tokenizer module `nemo` supports huggingface and sentencepiece tokenizer and `megatron-lm` supports `sentencepiece` tokenizer. Please note that `megatron-lm` still follows legacy format in `megatron-lm` ([check out this](https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/tokenizer/tokenizer.py#L416)). 

## Script Functionality

The script performs the following main tasks:

1. Parses command-line arguments to configure job submission parameters.

2. Optionally overloads Azure configuration settings with command-line arguments.

3. Retrieves information about input data files (JSONL shards) and optionally filters out files that have already been processed.

4. Submits tokenization jobs either to a local compute target or Azure Batch based on the chosen compute target.

5. For Azure Batch, it generates job YAML files and submits the jobs using Azure Machine Learning service.

## Command-line Arguments

The script accepts several command-line arguments to customize its behavior:

### Azure Login Parameters:

- `--az-subscription`: Azure subscription ID.
- `--az-resource-group`: Azure resource group name.
- `--az-workspace-name`: Azure Machine Learning workspace name.
- `--az-sas-token`: Azure Blob Storage SAS token.
- `--az-sample-yaml-job-file`: Path to a sample Azure Batch job YAML file.
- `--az-configs`: Path to a JSON file containing Azure configuration settings.

### I/O Parameters:

- `--input-folder-path`: Path to the folder containing JSONL input files for tokenization.
- `--bin-idx-folder-path`: Path to the folder where the tokenized output (bin and idx files) will be saved.
- `--tokenizer-module`: Tokenization module to use (e.g., 'megatron', 'nemo').
- `--tokenizer-model`: Path to the tokenizer model file.
- `--tokenizer-type`: Type of the tokenizer model (required).
- `--vocab-file`: Path to the vocabulary file (optional).
- `--merge-file`: Path to the merge file (optional).
- `--num-proc`: Number of workers per node for tokenization.
- `--log-interval`: Logging interval for progress updates.
- `--overwrite`: Flag to indicate whether to overwrite pre-existing bin-idx files.

### Miscellaneous Parameters:

- `--compute-target`: Compute target for tokenization ('local' or 'azure').
- `--dry-run`: Simulate the tokenization run without actually submitting jobs.

## Usage

To use the script, follow these steps:

1. Prepare your input data in the JSONL format and ensure you have the necessary tokenizer model and files.

2. Configure the Azure settings either through command-line arguments or by providing a JSON configuration file.

3. Specify the input and output folders, tokenizer module, tokenizer type, and other relevant parameters.

4. Run the script, providing the required command-line arguments. For example:

   ```bash
   python examples/data-processing/tokenize_shards.py --az-subscription <subscription_id> --az-resource-group <resource_group> --az-workspace-name <workspace_name> --az-sas-token <sas_token> --az-sample-yaml-job-file <sample_job_file.yaml> --input-folder-path <input_folder> --bin-idx-folder-path <output_folder> --tokenizer-module megatron --tokenizer-type <tokenizer_type> --tokenizer-model <model_path> --num-proc 16
   ```

5. The script will analyze the input data, check for existing tokenized files (if not overwriting), and submit tokenization jobs to the specified compute target.

6. Monitor the progress of tokenization jobs, and the script will handle job submission to Azure Batch if selected as the compute target.

7. Once the tokenization is complete, the tokenized output will be saved in the specified output folder.

8. Some sample script can be found at `examples/pretrain-allam/data-processing/tokenize` and `examples/pretrain-llama/data-processing/tokenize`.

## Note

- Make sure to customize the `prefix_command` variable in the script to match the actual path to your tokenization script if you are using Azure Batch.

- It is essential to configure the Azure settings and provide the necessary permissions to access the Azure resources.

- Use the `--dry-run` flag to test the script without actually submitting jobs to Azure Batch.

- The script can be extended or modified to support additional tokenization modules or compute targets as needed.

- Please note that `--compute-target=local or azure` works different way. `--compute-target=azure` submits one shard at a time as a job to process, each of the job assumes they have `--num-proc` cpu cores in the node. `--compute-target=local` uses `--num-proc` to tokenize one single shard. If it has more than `--num-proc` shards, it calculates number of concurrent file to be processed by `os.cpu_count()//args.num_proc`.