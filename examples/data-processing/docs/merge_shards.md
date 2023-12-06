# Doc for merge_shards.py

The provided Python script is designed to facilitate the merging of data shards stored on Azure Blob Storage or locally, depending on the specified compute target. The script is intended to be used for batch job submission and can merge shards based on a given shard size limit.

## Prerequisites

Before using this script, make sure you have the following prerequisites:

1. Python 3.x installed on your machine.
2. Azure CLI installed and configured if you plan to use Azure Blob Storage for shard storage.
3. Appropriate permissions to read data from the input folder and write data to the output folder, based on your chosen compute target.

## Usage

To run the script, follow these steps:

1. Open a terminal or command prompt.

2. Navigate to the directory where the script is located.

3. Execute the script with the desired command-line arguments.

   Example:

   ```bash
   python script_name.py --input-folder-path input_folder --output-folder-path output_folder --shard-size 1000000000 --prefix-name merged_output --compute-target azure
   ```

   Replace `script_name.py` with the actual name of the script file, and adjust the command-line arguments as needed.

## Command-Line Arguments

The script accepts several command-line arguments to customize its behavior:

### Azure Login Parameters (Group):

- `--az-subscription`: Azure subscription ID.
- `--az-resource-group`: Azure resource group name.
- `--az-workspace-name`: Azure workspace name.
- `--az-sas-token`: Azure Blob Storage SAS token.
- `--az-sample-yaml-job-file`: Path to a sample job file for Azure batch job submission.
- `--az-configs`: Path to a JSON configuration file containing Azure-related settings.

### Mandatory Parameters:

- `--input-folder-path`: The Azure Blob folder path or local folder path where the data shards are located. (Required)
- `--output-folder-path`: The Azure Blob folder path or local folder path where the merged output will be stored. (Required)
- `--shard-size`: The estimated size (in bytes) of each merged shard. (Required)
- `--prefix-name`: The prefix of the output file name. (Required)

### Miscellaneous Parameters (Group):

- `--compute-target`: Specifies the compute target. Can be either "local" or "azure." Both `--input-folder-path` and `--output-folder-path` should use the same compute target. (Default: "azure")
- `--dry-run`: Simulate the run before submitting jobs. If specified, no jobs will be submitted; the script will only print what it would do. (Optional)

## Script Functions

The script consists of several functions and logic:

1. `get_args()`: Parses command-line arguments using the `argparse` library and returns the parsed arguments.

2. `azcopy_list(path, sas_token)`: Lists files in an Azure Blob Storage folder using the `azcopy` command-line tool and returns a dictionary with shard names and sizes.

3. `group_shards(shard_dict, shard_size_limit)`: Groups shards based on their sizes, ensuring that the total size of each group does not exceed the specified shard size limit.

4. `remote_submit_jobs(args, groups)`: Submits Azure batch jobs for merging shards. It reads a sample YAML job file, replaces command parameters, and submits the job for each shard group.

5. `local_submit_job(args, groups)`: Merges shards locally by executing a shell command for each shard group.

6. `submit_jobs(args, groups)`: Determines the appropriate submission method based on the chosen compute target and calls either `remote_submit_jobs` or `local_submit_job`.

7. `list_files_with_size(folder_path)`: Lists files and their sizes in a local folder.

8. `get_shard_info(args, shard_folder)`: Retrieves shard information based on the compute target. Calls either `list_files_with_size` for local or `azcopy_list` for Azure Blob Storage.

9. The script's main block parses command-line arguments, retrieves shard information, groups the shards, and submits jobs accordingly based on the chosen compute target.

## Example Usage

Here's an example of how to use the script:

```bash
python script_name.py --input-folder-path azure_blob_folder --output-folder-path azure_blob_output_folder --shard-size 1000000000 --prefix-name merged_output --compute-target azure
```

This command will submit Azure batch jobs to merge data shards stored in `azure_blob_folder` and save the merged output in `azure_blob_output_folder`.

Please ensure that you have the necessary Azure credentials and permissions to perform these operations on Azure Blob Storage.

Note:

- If your folder has too many smaller sized file the script may fail due to long length of bash arguments. In that case you may have to process the data first. We had this issue in the slim pajama. Please take a look at the script at `examples/data-processing/merge_shards_runner/slim_pajama_process.sh`. 
   - Solution to this problem: 
      - Write the shard names in a text file.
      - Instead of sending file names in bash args, send them via the text file.
      - read the text file in the `data-processing/remote_scripts/remote_merge_shard.sh`
      - make sure you have backward compatibility intact (read merge file list from arguments) in the script. 
- merge shard can only merge. It cannot split. If your folder has a larger file than `--shard-size` it'll raise an argument. 