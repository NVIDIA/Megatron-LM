import os
import yaml
import json
import copy
import argparse 
import subprocess

def get_args():
    parser = argparse.ArgumentParser(description="Azure batch job submission for tokenization")
    group = parser.add_argument_group(title='Azure login params.')
    group.add_argument("--az-subscription", default=None, type = str, help="Azure subscription id.")
    group.add_argument("--az-resource-group", default=None, type = str, help="Azure subscription id.")
    group.add_argument("--az-workspace-name", default=None, type = str, help="Azure subscription id.")
    group.add_argument("--az-sas-token", default=None, type = str, help="Azure blob SAS token")
    group.add_argument("--az-sample-yaml-job-file", default=None, type = str, help="Path to a sample job file.")
    group.add_argument("--az-configs", default=None, type = str, help="Path to a sample job file.")
    
    parser.add_argument("--input-folder-path", type = str, help="Azure blob folder path.", required=True)
    parser.add_argument("--output-folder-path", type = str, help="Azure blob folder path to output folder for bin and idx", required=True)
    parser.add_argument("--shard-size", type = int, help="Size of the each merged shard.", required=True)
    parser.add_argument("--prefix-name", type = str, help="Prefix of the output file name.", required=True)
    
    group = parser.add_argument_group(title='Misc. params.')
    parser.add_argument("--compute-target", type = str, default='azure', choices=['local', 'azure'], help="Conpute targets.")
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
        if line.strip() == "": continue
        pre_shard_dt = json.loads(line)
        message = pre_shard_dt['MessageContent']
        if message.strip() == "": continue
        pre_shard_name = message.split(";")[0].replace("INFO: ", "")
        pre_shard_size = int(message.split(";")[1].replace("Content Length: ", ""))
        shard_dict[pre_shard_name] = pre_shard_size
    return shard_dict
            
def group_shards(shard_dict, shard_size_limit):
    groups, curr_group, curr_group_size = [], [], 0
    for shard_name, size_in_byte in  shard_dict.items():
        assert size_in_byte < shard_size_limit
        curr_group_size += size_in_byte
        if curr_group_size > shard_size_limit:
            groups.append((curr_group_size,copy.deepcopy(curr_group)))
            curr_group = []
            curr_group_size = 0
        curr_group.append(shard_name)
    if len(curr_group) > 0: groups.append((curr_group_size, curr_group))
    return groups

def azure_submit_jobs(args, groups, script_path):
    output_folder = os.path.dirname(script_path)
    with open(args.az_configs['az-sample-yaml-job-file']) as fileptr:
        data = yaml.safe_load(fileptr)
    sas_token = args.az_configs['az-sas-token']
    prefix_command = f"""bash examples/pretrain-llama/data-processing/merge_shard/remote_merge_shard.sh \\
\"{args.input_folder_path}\" \\
\"{args.output_folder_path}\" \\
\"{sas_token}\" \\
\"{args.prefix_name}"""
    for idx, (size, shards) in enumerate(groups):
        command = prefix_command
        command += f"{idx:03}.jsonl\""
        for shard in shards:
            command += f" \\\n\"{shard}\""
        if not args.dry_run:
            data['command'] = command
            az_yaml_file = os.path.join(output_folder, 'output.yaml')
            with open(az_yaml_file, 'w') as wrt_ptr:
                yaml.dump(data, wrt_ptr, default_flow_style=False)
            cmd = f"az ml job create --subscription {args.az_configs['az-subscription']} --resource-group {args.az_configs['az-resource-group']} --workspace-name {args.az_configs['az-workspace-name']} --file {az_yaml_file}"
            subprocess.check_output(cmd, shell=True)
        print(f"[{args.prefix_name}][{idx}] {size/1000000000}GB")

def local_submit_job(args, groups, script_path):
    output_folder = os.path.dirname(script_path)
    with open(args.sample_yaml_job_file) as fileptr:
        data = yaml.safe_load(fileptr)
    sas_token = args.az_configs['az-sas-token']
    prefix_command = f"""bash examples/pretrain-llama/data-processing/merge_shard/remote_merge_shard.sh \\
\"{args.input_folder_path}\" \\
\"{args.output_folder_path}\" \\
\"{sas_token}\" \\
\"{args.prefix_name}"""
    for idx, (size, shards) in enumerate(groups):
        command = prefix_command
        command += f"{idx:03}.jsonl\""
        for shard in shards:
            command += f" \\\n\"{shard}\""
        if not args.dry_run:
            subprocess.check_output(command, shell=True)
        print(f"[{args.prefix_name}][{idx}] {size/1000000000}GB")

def submit_jobs(args, groups, script_path):
    if args.compute_target == "local":
        raise NotImplementedError()
    elif args.compute_target == "azure":
        return azure_submit_jobs(args, groups, script_path)
    else:
        raise NotImplementedError()

def list_shard_info(args, shard_folder):
    if args.compute_target == "local":
        raise NotImplementedError()
    elif args.compute_target == "azure":
        return azcopy_list(shard_folder, args.az_configs['az-sas-token'])
    else:
        raise NotImplementedError()
    
if __name__ == "__main__":
    script_path = os.path.abspath(__file__)
    args =  get_args()
    shard_dict = list_shard_info(args, args.input_folder_path)
    groups = group_shards(shard_dict, args.shard_size)
    submit_jobs(args, groups, script_path)
