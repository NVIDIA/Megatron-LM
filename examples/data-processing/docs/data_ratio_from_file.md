# Doc for data_ratio_from_file.py

# Script Documentation

## Overview
This script is designed to preprocess data files and generate information for language model training. It takes input parameters and performs various operations like selecting data files, calculating language probabilities, and generating output information. Below is the documentation for the script.

The script calculates the sampling probability distribution of each of the iterator (loaded from `*.bin`, `*.idx`).

## Usage
You can use the script with the following command line arguments:

- `--source-prefix-paths`: A glob path to the folder where all the binary (`bin`) and index (`idx`) files are located. This is an alternative to providing file names in a JSON format (`--prefix-paths-from-json`).

- `--prefix-paths-from-json`: A JSON file containing a list of file names. This is an alternative to specifying `--source-prefix-paths`.

- `--domain-ratio-from-json`: A JSON file specifying domain multipliers for languages.

- `--lang-select-prob-json`: A JSON file indicating the language selection probabilities.

- `--exclude-iterator-json`: A JSON file listing restricted iterator names.

- `--total-token`: The total number of tokens to be sampled.

- `--verbose`: A flag to print additional information.

- `--export-for-script`: The path to a file where the output will be exported in Megatron format.

- `--prefix-for-file-path`: An additional prefix to be added to the file path.

## Script Functionality
The script performs the following main tasks:

1. Parse command line arguments using the `argparse` module.

2. Check for conflicts between `--source-prefix-paths` and `--prefix-paths-from-json` arguments.

3. Remove trailing slashes from `--prefix-for-file-path`.

4. Load necessary JSON files (`domain-ratio-from-json`, `lang-select-prob-json`, `exclude-iterator-json`) and store their data in corresponding dictionaries/lists.

5. Process files from the specified paths or JSON, and calculate language-wise data distribution.

6. Calculate probabilities for iterator selection and store them in a list.

7. Ensure that the language probabilities match the total probability distribution.

8. Print statistics about the total token count by language.

9. If `--export-for-script` is provided, create a script file in the Megatron format with the selected files.

10. Display language-wise token distribution.

## Example Usage
Here's an example of how to use the script:

```bash
python script.py \
  --source-prefix-paths /path/to/data/files/ \
  --domain-ratio-from-json domain_ratios.json \
  --lang-select-prob-json lang_probabilities.json \
  --exclude-iterator-json excluded_iterators.json \
  --total-token 1000000 \
  --verbose \
  --export-for-script output_script.sh \
  --prefix-for-file-path /data/
```

## Output
Usually you can copy paste the output to your bash script. However if you want to bring the data argument programatically, you can use `--export-for-script` argument. It will create a bash file with a DATA_PATH bash variable. In your megatron launcher script, just simply source the `--export-for-script` and use the DATA_PATH variable as `${DATA_PATH[@]}`. 

## Notes
- Make sure to provide the correct paths to the required JSON files.
- The script is designed for a specific data preprocessing task and may require adjustments for different use cases.
- Note unless you have all the `*.bin` / `*.idx` downloaded in your local drive it is recommended to use `--prefix-paths-from-json` arguments. For `--prefix-paths-from-json` you only have to send a data signature of the `*.bin` / `*.idx`. Run the following command to create a data signature file,

```
python examples/data-processing/remote_list.py --help
```