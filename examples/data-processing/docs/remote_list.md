# Script Documentation: File Listing and Azure Job Submission

## Overview

This script is designed to perform two primary tasks:

List files in a specified folder or Azure Blob Storage container.

The script is highly configurable and can be used for various scenarios, such as batch processing large files stored in Azure Blob Storage.

## Usage

To use the script, you need to provide command-line arguments. Here are the available arguments:

### Azure Login Parameters

- `--az-subscription`: Azure subscription id.
- `--az-resource-group`: Azure resource group name.
- `--az-workspace-name`: Azure Machine Learning workspace name.
- `--az-sas-token`: Azure Blob Storage Shared Access Signature (SAS) token.
- `--az-sample-yaml-job-file`: Path to a sample job YAML file.
- `--az-configs`: Path to a JSON file containing Azure configuration settings.

### Input/Output Parameters

- `--input-folder-path`: Compute target folder path to input JSONL files. Each job will process a single JSONL file.

### Miscellaneous Parameters

- `--compute-target`: Compute target for job submission. Choose between 'local' (for local execution) or 'azure' (for Azure execution).
- `--dry-run`: Simulate the run before submitting jobs. When specified, the script will not actually submit jobs; it will only display what would be done.

## Functions

### `get_args()`

This function parses the command-line arguments using the `argparse` library and returns an `argparse.Namespace` object containing the parsed arguments.

### `azcopy_list(path: str, sas_token: str) -> Dict[str, int]`

This function lists files in an Azure Blob Storage container using the `azcopy` command-line tool. It takes the path to the container and the SAS token as input and returns a dictionary where keys are file names, and values are their respective sizes in bytes.

### `list_files_with_size(folder_path: str) -> Dict[str, int]`

This function lists files in a local folder and returns a dictionary where keys are file paths, and values are their respective sizes in bytes.

### `get_shard_info(args: argparse.Namespace, shard_folder: str) -> Dict[str, int]`

This function determines the method to obtain shard information based on the compute target. If the target is 'local,' it uses `list_files_with_size`, and if the target is 'azure,' it uses `azcopy_list`. It returns a dictionary with shard information.

## Main Execution

The script begins execution in the `__main__` block. It parses the command-line arguments, retrieves shard information, and prints the shard names and sizes in gigabytes.

## Example Usage

Here's an example of how to use the script:

```bash
python script_name.py --compute-target azure --az-subscription "your_subscription_id" --az-resource-group "your_resource_group" --az-workspace-name "your_workspace" --az-sas-token "your_sas_token" --input-folder-path "your_folder_path"
```

This command will list files in the specified Azure Blob Storage container and print their names and sizes.

## Notes

- Ensure you have the necessary Azure credentials and permissions to access the specified Azure resources.
- Make sure to install and configure the `azcopy` tool for Azure operations.
- The script can be extended to include additional job submission logic as needed for your specific use case.