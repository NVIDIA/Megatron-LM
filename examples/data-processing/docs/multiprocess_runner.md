# Documentation for multiprocess_runner.py

## Overview
This script is designed to facilitate multi-process data processing, particularly for tokenization of text data using either the Megatron or NeMo tokenizer. It allows you to parallelize the tokenization process across multiple input files and generate tokenized output in binary format. Below is a detailed explanation of the script's functionality and its usage.

## Prerequisites
Before using this script, you need to have the following prerequisites in place:

1. Python 3.x installed on your system.
2. The required tokenizer module (Megatron or NeMo) must be installed along with its dependencies.
3. Pre-trained tokenizer model for the chosen module.
4. Input JSONL files containing the text data to be tokenized.

## Command-line Arguments
The script accepts several command-line arguments to configure its behavior:

- `--glob-input-path` (required): Specifies a glob pattern for locating input JSONL files.
- `--output-folder` (required): Path to the folder where the tokenized output files will be saved.
- `--tokenizer-module` (default: 'megatron', choices: ['megatron', 'nemo']): Specifies which tokenizer module to use (Megatron or NeMo).
- `--tokenizer-type` (required): Specifies the type of the tokenizer model.
- `--tokenizer-model` (required): Path to the tokenizer model.
- `--vocab-file` (default: None): Path to the vocabulary file of the tokenizer (only required for some tokenizers).
- `--merge-file` (default: None): Path to the merge file of the tokenizer (only required for some tokenizers).
- `--per-file-workers` (required): Number of workers (processes) per input file.
- `--num-file-workers` (required): Number of input files to be processed simultaneously.
- `--log-interval` (default: 1000): Interval between progress updates during tokenization.

## Script Execution
Here's how the script works:

1. It parses the command-line arguments provided when running the script.

2. For each input file matching the glob pattern specified by `--glob-input-path`, it launches a parallel process to tokenize the data.

3. The tokenization process is determined by the selected tokenizer module (`--tokenizer-module`), which can be either 'megatron' or 'nemo'.

4. The script constructs a command based on the chosen tokenizer module and specified arguments, such as the input file, output prefix, tokenizer model, vocabulary file, merge file, number of workers, and log interval.

5. It then executes the constructed command using `subprocess.check_output()` to tokenize the input file.

6. Once tokenization is complete for a file, the script returns the file's path, indicating successful processing.

7. All tokenization tasks are executed concurrently using a process pool, with the number of workers specified by `--num-file-workers`.

8. The script provides progress updates at intervals specified by `--log-interval`.

## Example Usage
Here's an example of how to use this script:

```bash
python multi_process_data_processing.py \
  --glob-input-path "/path/to/input/files/*.jsonl" \
  --output-folder "/path/to/output" \
  --tokenizer-module "megatron" \
  --tokenizer-type "bpe" \
  --tokenizer-model "/path/to/tokenizer/model" \
  --vocab-file "/path/to/vocab/file" \
  --merge-file "/path/to/merge/file" \
  --per-file-workers 4 \
  --num-file-workers 8 \
  --log-interval 1000
```

In this example, the script will process JSONL files matching the specified glob pattern in parallel, tokenize them using the Megatron tokenizer, and save the tokenized output in the specified output folder.