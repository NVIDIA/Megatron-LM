import os
import json
import yaml
import argparse 
import subprocess
from typing import Dict
from datetime import datetime

def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Azure batch job submission for processing a shard")
    group = parser.add_argument_group(title='Azure login params.')
    group.add_argument("--az-subscription", default=None, type = str, help="Azure subscription id.")
    group.add_argument("--az-resource-group", default=None, type = str, help="Azure subscription id.")
    group.add_argument("--az-workspace-name", default=None, type = str, help="Azure subscription id.")
    group.add_argument("--az-sas-token", default=None, type = str, help="Azure blob SAS token")
    group.add_argument("--az-sample-yaml-job-file", default=None, type = str, help="Path to a sample job file.")
    group.add_argument("--az-configs", default=None, type = str, help="Path to a sample job file.")

    group = parser.add_argument_group(title='Remote I/O params.')
    group.add_argument("--input-folder-path", type = str, help="Compute target folder path to input jsonl files, each job process a single jsonl file.")
    group.add_argument("--output-folder-path", type = str, help="Compute target folder path to output folder path.")
    group.add_argument("--config-path-for-process-function", type = str, help="Config path for process function.")
    group.add_argument("--num-shards", type = int, default=1, help="How many shards to be processed in one pass.")

    group = parser.add_argument_group(title='Misc. params.')
    parser.add_argument("--compute-target", type = str, default='azure', choices=['local', 'azure'], help="Conpute targets. Both --input-folder-path, --output-folder-path"
                        " TODO: Enable cross compute.")
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

def match_pre_existing_bin_idx(input_shard_dict: Dict[str, int], output_shards_dict: Dict[str, int]) -> Dict[str, int]:
    de_dup_shard_collection = {}
    for shard_name, n_byte in input_shard_dict.items():
        is_done = False
        for pre_exist_shard_name, _ in output_shards_dict.items():
            if pre_exist_shard_name.startswith(shard_name):
                print("Skipping...! Shard already in output folder."
                      " Potentially already processed by other process.")
                is_done = True
                break
        if is_done is False:
            de_dup_shard_collection[shard_name] = n_byte
    return de_dup_shard_collection

def parse_command(args, input_shard_dict):
    shard_queue = []

    def parse(queue):
        cmd = f"python examples/data-processing/remote_scripts/remote_process_shard.py "
        cmd = cmd + f' --shard-names ' + ' '.join([ name for name, size in queue])
        cmd = cmd + f' --input-folder-path {args.input_folder_path}'
        cmd = cmd + f' --output-folder-path {args.output_folder_path}'
        cmd = cmd + f' --config-path-for-process-function {args.config_path_for_process_function}'
        cmd = cmd + f' --compute-target {args.compute_target}'
        _size = sum([size for name, size in queue])
        return cmd, _size
    
    launch_cnt = 1
    for idx, (shard_name, size) in enumerate(input_shard_dict.items()):
        shard_queue.append((shard_name, size))
        if len(shard_queue) == args.num_shards:
            cmd, size = parse(shard_queue)
            print(f"RUN [{launch_cnt}][{size/1000000000}GB]: {cmd}")
            yield cmd
            launch_cnt += 1
            shard_queue = []
    if len(shard_queue) > 0:
        cmd, size = parse(shard_queue)
        print(f"RUN [{launch_cnt}][{size/1000000000}GB]: {cmd}")
        yield cmd

def local_submit_job(args: argparse.Namespace) -> None:
    for cmd in parse_command(args, input_shard_dict):
        if not args.dry_run:
            subprocess.check_output(cmd, shell=True)

def azure_submit_jobs(args: argparse.Namespace, input_shard_dict: Dict[str, int]) -> None:
    with open(args.az_configs['az-sample-yaml-job-file']) as fileptr:
        data = yaml.safe_load(fileptr)
    sas_token = args.az_configs['az-sas-token']
    for cmd in parse_command(args, input_shard_dict):
        if not args.dry_run:
            shard_name = cmd.split("--shard-name")[1].split()[0]
            data['command'] = cmd
            data['code'] = "../"
            data['display_name'] = shard_name
            prefix_path = '.temp/'
            os.makedirs(prefix_path, exist_ok=True)
            az_yaml_file = os.path.join(
                prefix_path, 
                f'process_shard_{datetime.now().strftime("%Y-%m-%d-%H-%M-%S")}.yaml'
            )
            with open(az_yaml_file, 'w') as wrt_ptr:
                yaml.dump(data, wrt_ptr, default_flow_style=False)
            cmd = f"az ml job create "
            cmd = cmd + f' --subscription {args.az_configs["az-subscription"]}'
            cmd = cmd + f' --resource-group {args.az_configs["az-resource-group"]}'
            cmd = cmd + f' --workspace-name {args.az_configs["az-workspace-name"]}'
            cmd = cmd + f' --file {az_yaml_file}'
            subprocess.check_output(cmd, shell=True)

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
    submit_jobs(args, input_shard_dict)
