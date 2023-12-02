# Doc for data_ratio_from_file.py

## Introduction
This Python script is designed to process and select data sources based on certain criteria of the pretraining. It takes into account various parameters and input sources such as JSON files, file paths, and probabilities to create a data distribution by language.

## Usage
To run the script, you need to provide the following command-line arguments:

- `--source-prefix-paths`: A glob path to the folder where all the binary and index files are located. This argument is mutually exclusive with `--prefix-paths-from-json`.
- `--prefix-paths-from-json`: A JSON file containing a list of file names. This argument is mutually exclusive with `--source-prefix-paths`.
- `--domain-ratio-from-json`: Path to a JSON file that contains domain multipliers for different languages.
- `--lang-select-prob-json`: Path to a JSON file that indicates the language selection probabilities.
- `--exclude-iterator-json`: Path to a JSON file that lists restricted iterator names.
- `--total-token`: The total number of tokens to be sampled.
- `--verbose`: Optional. If specified, the script will print additional information.
- `--output-for-script`: Optional. If specified, the script will print output in a format suitable for a bash script.

## Main Functionality
Here is an overview of what the script does:

1. Parses command-line arguments using `argparse`.
2. Validates the arguments to ensure that `--source-prefix-paths` and `--prefix-paths-from-json` are mutually exclusive.
3. Reads input data from JSON files and sets up the necessary data structures.
4. Processes the data and calculates language-specific token distributions.
5. Calculates iterator selection probabilities based on language, domain, and domain multipliers.
6. Prints the selected iterators and token distributions.

## Detailed Explanation
- The script first parses command-line arguments using the `argparse` library.
- It checks if either `--source-prefix-paths` or `--prefix-paths-from-json` is provided and asserts that they are mutually exclusive.
- The script then reads input data from JSON files, including domain ratios, language selection probabilities, and excluded iterator names.
- It processes the binary file paths based on certain naming conventions and filters out iterators based on the exclusion list.
- For each valid iterator, it calculates language-specific token distributions and iterator selection probabilities.
- The script prints the selected iterators, token distributions, and language-wise token distribution in JSON format.

## Output
The script generates various output, including the list of selected iterators, token distributions by language, and language-wise token distribution. The output can be printed to the console or formatted for a bash script, depending on the specified options.

## Conclusion
This script serves as a tool for selecting and processing data sources based on specific criteria, such as language, domain, and token distribution, for various natural language processing tasks. It provides flexibility in handling input data from both file paths and JSON files and allows for detailed control over data selection and distribution.