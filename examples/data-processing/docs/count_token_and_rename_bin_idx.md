# Doc for count_token_and_rename_bin_idx.py

## Overview

The `count_token_and_rename_bin_idx.py` script is designed to work with indexed dataset files typically used in natural language processing tasks. It counts the number of tokens in these files and renames them based on the token count, document count, and sentence count. This script is particularly useful for managing and organizing large datasets.

Current Naming Format:

```
{language}_{domain}_{subdomain}_split_{split_id}_text_document_dc={document_count}_sc={sentence_count}_tc={token_count}"
```

## Prerequisites

Before using this script, ensure that you have the following prerequisites installed:

- Python 3.x
- `argparse` library
- `tqdm` library
- `transformers` library
- `megatron` library (if required by your dataset)

## Usage

To run the script, execute it using Python from the command line as follows:

```bash
python count_token_and_rename_bin_idx.py --source-prefix-paths "../DUMPED/*"
```

- `--source-prefix-paths`: This argument is required and should be a glob path to the folder where all the bin and idx files are located. The script will process files in this folder.

## Script Functions

### `get_args()`

This function parses command-line arguments using `argparse` and returns the parsed arguments. It also appends the `megatron` library path to `sys.path` if provided.

### `test_indexed_dataset(source_prefix_path: str)`

This function tests an indexed dataset located at the given `source_prefix_path`. It uses functions from the `megatron` library to access and analyze the dataset. The function returns the document count, sentence count, and token count of the dataset.

### `main()`

The main function of the script:

- Parses command-line arguments.
- Obtains a list of bin file paths using `glob` and sorts them.
- Iterates through the list of bin files, testing each indexed dataset using `test_indexed_dataset()`.
- Prints information about the dataset, such as document count, sentence count, and token count.
- Renames the bin and idx files based on the counts, if necessary.
  - For example, if the file name contains "dc=", "sc=", or "tc=" along with the respective counts, it verifies the counts and renames the files accordingly.
  - If no counts are found in the file name, it renames the files with counts of document, sentence, and token.
- Uses `subprocess` to execute shell commands to rename files.

## Example

Here's an example of how to run the script:

```bash
python count_token_and_rename_bin_idx.py --source-prefix-paths "../DUMPED/*"
```

This will process indexed dataset files located in the specified folder, count tokens, and rename the files based on the counts.

Please note that this script may have dependencies on specific libraries and dataset structures, so ensure that it is adapted to your specific use case as needed.