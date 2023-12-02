import os
import json
import yaml
import argparse 
import subprocess
from typing import Dict
from datetime import datetime

def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Listing files in a folder.")
    group = parser.add_argument_group(title='Azure login params.')
    group.add_argument("--az-subscription", default=None, type = str, help="Azure subscription id.")
    group.add_argument("--az-resource-group", default=None, type = str, help="Azure subscription id.")
    group.add_argument("--az-workspace-name", default=None, type = str, help="Azure subscription id.")
    group.add_argument("--az-sas-token", default=None, type = str, help="Azure blob SAS token")
    group.add_argument("--az-sample-yaml-job-file", default=None, type = str, help="Path to a sample job file.")
    group.add_argument("--az-configs", default=None, type = str, help="Path to a sample job file.")

    group = parser.add_argument_group(title='I/O params.')
    group.add_argument("--input-folder-path", type = str, help="Compute target folder path to input jsonl files, each job process a single jsonl file.")

    group = parser.add_argument_group(title='Misc. params.')
    parser.add_argument("--compute-target", type = str, default='azure', choices=['local', 'azure'], help="Conpute targets. Both --input-folder-path, --bin-idx-folder-path"
                        " and --tokenizer-model should use same compute target. TODO: Enable cross compute.")
    group.add_argument("--dry-run", action='store_true', help="Simulate run before submitting jobs.")
    
    args = parser.parse_args()

    if args.az_configs is not None:
        args.az_configs = json.load(open(args.az_configs))
        if args.az_subscription is not None:
            print("Overloading config args from  --az-subscription")
            args.az_configs['az-subscription'] = args.az_subscription
        if args.az_resource_group is not None:
            print("Overloading config args from  --az-resource-group")
            args.az_configs['az-resource-group'] = args.az_resource_group
        if args.az_workspace_name is not None:
            print("Overloading config args from  --az-workspace-name")
            args.az_configs['az-workspace-name'] = args.az_workspace_name
        if args.az_sas_token is not None:
            print("Overloading config args from  --az-sas-token")
            args.az_configs['az-sas-token'] = args.az_sas_token
        if args.az_sample_yaml_job_file is not None:
            print("Overloading config args from  --az-sample-yaml-job-file")
            args.az_configs['az-sample-yaml-job-file'] = args.az_sample_yaml_job_file

    return args

def azcopy_list(path: str, sas_token: str) -> Dict[str, int]:
    shard_dict = {}
    path_with_sas_token = f"{path}?{sas_token}"
    cmd = f"azcopy list \"{path_with_sas_token}\" --output-type json --machine-readable"
    out = subprocess.check_output(cmd, shell=True)
    out = out.decode('utf-8')
    for line in out.split("\n"):
        if line.strip() == "": continue
        pre_shard_dt = json.loads(line)
        message = pre_shard_dt['MessageContent']
        if message.strip() == "": continue
        pre_shard_name = message.split(";")[0].replace("INFO: ", "")
        pre_shard_size = int(message.split(";")[1].replace("Content Length: ", ""))
        shard_dict[pre_shard_name] = pre_shard_size
    return shard_dict

def list_files_with_size(folder_path: str) -> Dict[str, int]:
    file_path_with_size = {}
    if os.path.isfile(folder_path):
        file_path_with_size[folder_path] = os.path.getsize(folder_path)
        return file_path_with_size
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            file_path = os.path.join(root, file)
            size = os.path.getsize(file_path)
            file_path_with_size[file_path] = size
    return file_path_with_size

def submit_jobs(args: argparse.Namespace, input_shard_dict: Dict[str, int]) -> None:
    if args.compute_target == "local":
        local_submit_job(args)
    elif args.compute_target == "azure":
        return azure_submit_jobs(args, input_shard_dict)
    else:
        raise NotImplementedError()

def get_shard_info(args: argparse.Namespace, shard_folder: str) -> Dict[str, int]:
    if args.compute_target == "local":
        return list_files_with_size(shard_folder)
    elif args.compute_target == "azure":
        return azcopy_list(shard_folder, args.az_configs['az-sas-token'])
    else:
        raise NotImplementedError()

if __name__ == "__main__":
    args =  get_args()
    input_shard_dict = get_shard_info(args, args.input_folder_path)
    for shard_name, n_byte in input_shard_dict.items():
        print(f"{shard_name}: {n_byte//1000000000} GB")    
    
