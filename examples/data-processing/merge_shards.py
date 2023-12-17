import os
import yaml
import json
import copy
import argparse 
import subprocess
from datetime import datetime

def get_args():
    parser = argparse.ArgumentParser(description="Azure batch job submission for merging shards.")
    group = parser.add_argument_group(title='Azure login params.')
    group.add_argument("--az-subscription", default=None, type = str, help="Azure subscription id.")
    group.add_argument("--az-resource-group", default=None, type = str, help="Azure subscription id.")
    group.add_argument("--az-workspace-name", default=None, type = str, help="Azure subscription id.")
    group.add_argument("--az-sas-token", default=None, type = str, help="Azure blob SAS token")
    group.add_argument("--az-sample-yaml-job-file", default=None, type = str, help="Path to a sample job file.")
    group.add_argument("--az-configs", default=None, type = str, help="Path to a sample job file.")
    
    parser.add_argument("--input-folder-path", type = str, help="Azure blob folder path.", required=True)
    parser.add_argument("--output-folder-path", type = str, help="Azure blob folder path to output folder for bin and idx", required=True)
    parser.add_argument("--shard-size", type = int, help="Estimated size of the each merged shard. (TODO: Fix to exact size)", required=True)
    parser.add_argument("--prefix-name", type = str, help="Prefix of the output file name.", required=True)
    parser.add_argument("--use-file-input", action='store_true', help="Send the merge shard names in a file instead of arguments. "
                        "This is required when you merge too many small files.", required=True)
    
    group = parser.add_argument_group(title='Misc. params.')
    parser.add_argument("--compute-target", type = str, default='azure', choices=['local', 'azure'], help="Conpute targets. Both --input-folder-path and --output-folder-path should use same compute target. TODO: Enable cross compute.")
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

def azcopy_list(path, sas_token):
    shard_dict = {}
    path_with_sas_token = f"{path}?{sas_token}"
    cmd = f"azcopy list \"{path_with_sas_token}\" --output-type json --machine-readable"
    out = subprocess.check_output(cmd, shell=True)
    out = out.decode('utf-8')
    for line in out.split("\n"):
        print(line)
        if line.strip() == "": continue
        pre_shard_dt = json.loads(line)
        message = pre_shard_dt['MessageContent']
        if message.strip() == "": continue
        if message.startswith("INFO: azcopy"): continue
        pre_shard_name = message.rsplit(";", 1)[0].replace("INFO: ", "")
        pre_shard_size = int(message.rsplit(";", 1)[1].replace("Content Length: ", ""))
        shard_dict[pre_shard_name] = pre_shard_size
    return shard_dict
            
def group_shards(shard_dict, shard_size_limit):
    groups, curr_group, curr_group_size = [], [], 0
    for shard_name, size_in_byte in  shard_dict.items():
        assert size_in_byte < shard_size_limit
        curr_group_size += size_in_byte
        curr_group.append(shard_name)
        if curr_group_size > shard_size_limit:
            curr_group_size -= size_in_byte
            curr_group.pop()
            groups.append((curr_group_size,copy.deepcopy(curr_group)))
            curr_group = [shard_name]
            curr_group_size = size_in_byte
    if len(curr_group) > 0: groups.append((curr_group_size, curr_group))
    return groups

def remote_submit_jobs(args, groups):
    with open(args.az_configs['az-sample-yaml-job-file']) as fileptr:
        data = yaml.safe_load(fileptr)
    sas_token = args.az_configs['az-sas-token']
    if args.input_folder_path[-1] == "/": args.input_folder_path = args.input_folder_path.rstrip('/\\')
    if args.output_folder_path.endswith("/"): args.output_folder_path = args.output_folder_path.rstrip('/\\')
    prefix_command = f"""bash examples/data-processing/remote_scripts/remote_merge_shard.sh \\
\"{args.input_folder_path}\" \\
\"{args.output_folder_path}\" \\
\"{sas_token}\" \\
\"{args.compute_target}\" \\
\"{args.prefix_name}"""
    for idx, (size, shards) in enumerate(groups):
        command = prefix_command
        command += f"{idx:03}.jsonl\""
        if args.use_file_input:
            input_file_path = f"input_{args.prefix_name}{idx:03}.txt"
            with open(input_file_path, 'w') as wrt_ptr:
                for shard in shards:
                    wrt_ptr.write(f"{shard}\n")
            command += " " + input_file_path
        else:
            for shard in shards:
                command += f" \\\n\"{shard}\""
        print(f"[{idx}][{args.prefix_name}{idx:03}.jsonl][{size/1000000000:.2f}GB] {command}")
        if not args.dry_run:
            data['command'] = command
            data['code'] = "../"
            prefix_path = '.temp/'
            os.makedirs(prefix_path, exist_ok=True)
            az_yaml_file = os.path.join(prefix_path, f'merge_shard_{datetime.now().strftime("%Y-%m-%d-%H-%M-%S")}.yaml')
            with open(az_yaml_file, 'w') as wrt_ptr:
                yaml.dump(data, wrt_ptr, default_flow_style=False)
            print(f"Submitting job via : {az_yaml_file}")
            cmd = f"az ml job create --subscription {args.az_configs['az-subscription']} --resource-group {args.az_configs['az-resource-group']} --workspace-name {args.az_configs['az-workspace-name']} --file {az_yaml_file}"
            try:
                subprocess.check_output(cmd, shell=True)
            except:
                raise
        
def local_submit_job(args, groups):
    if args.input_folder_path[-1] == "/": args.input_folder_path = args.input_folder_path.rstrip('/\\')
    if args.output_folder_path.endswith("/"): args.output_folder_path = args.output_folder_path.rstrip('/\\')
    prefix_command = f"""bash examples/data-processing/remote_scripts/remote_merge_shard.sh \\
\"{args.input_folder_path}\" \\
\"{args.output_folder_path}\" \\
\"none\" \\
\"{args.compute_target}\" \\
\"{args.prefix_name}"""
    for idx, (size, shards) in enumerate(groups):
        command = prefix_command
        command += f"{idx:03}.jsonl\""
        if args.use_file_input:
            input_file_path = f"input_{args.prefix_name}{idx:03}.txt"
            with open(input_file_path, 'w') as wrt_ptr:
                for shard in shards:
                    wrt_ptr.write(f"{shard}\n")
            command += " " + input_file_path
        else:
            for shard in shards:
                command += f" \\\n\"{shard}\""
        print(f"[{args.prefix_name}][{idx}] {size/1000000000} GB")
        if not args.dry_run:
            try:
                print(command)
                subprocess.check_output(command, shell=True)
            except:
                raise

def submit_jobs(args, groups):
    if args.compute_target == "local":
        return local_submit_job(args, groups)
    elif args.compute_target == "azure":
        return remote_submit_jobs(args, groups)
    else:
        raise NotImplementedError()

def list_files_with_size(folder_path):
    file_path_with_size = {}
    print(folder_path)
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            file_path = os.path.join(root, file)
            size = os.path.getsize(file_path)
            file_path_with_size[file_path] = size
    return file_path_with_size

def get_shard_info(args, shard_folder):
    if args.compute_target == "local":
        return list_files_with_size(shard_folder)
    elif args.compute_target == "azure":
        return azcopy_list(shard_folder, args.az_configs['az-sas-token'])
    else:
        raise NotImplementedError()
    
if __name__ == "__main__":
    args =  get_args()
    shard_dict = get_shard_info(args, args.input_folder_path)
    groups = group_shards(shard_dict, args.shard_size)
    submit_jobs(args, groups)
