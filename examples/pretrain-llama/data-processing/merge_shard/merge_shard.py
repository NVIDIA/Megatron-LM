import os
import yaml
import json
import copy
import argparse 
import subprocess

def get_args():
    parser = argparse.ArgumentParser(description="Azure batch job submission for tokenization")
    parser.add_argument("--az-subscription", type = str, help="Azure subscription id.", required=True)
    parser.add_argument("--az-resource-group", type = str, help="Azure subscription id.", required=True)
    parser.add_argument("--az-workspace-name", type = str, help="Azure subscription id.", required=True)
    parser.add_argument("--az-blob-input-folder", type = str, help="Azure blob folder path.", required=True)
    parser.add_argument("--az-blob-output-folder-path", type = str, help="Azure blob folder path to output folder for bin and idx", required=True)
    parser.add_argument("--az-sas-token", type = str, help="Azure blob SAS token", required=True)
    parser.add_argument("--shard-size", type = int, help="Size of the each merged shard.", required=True)
    parser.add_argument("--sample-yaml-job-file", type = str, help="Path to a sample job file.", required=True)
    parser.add_argument("--prefix-name", type = str, help="Prefix of the output file name.", required=True)
    parser.add_argument("--dry-run", action='store_true', help="Simulate run before submitting jobs.")
    args = parser.parse_args()
    return args

def azcopy_list(path, sas_token):
    shards = []
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
        shards.append({
            'name': pre_shard_name,
            'size_in_byte': pre_shard_size
        })
    shards = sorted(shards, key=lambda x: x['name'])
    return shards
            
def group_shards(shards, shard_size_limit):
    groups, curr_group, curr_group_size = [], [], 0
    for dt in shards:
        shard_name = dt['name']
        assert dt['size_in_byte'] < shard_size_limit
        curr_group_size += dt['size_in_byte']
        if curr_group_size > shard_size_limit:
            groups.append((curr_group_size,copy.deepcopy(curr_group)))
            curr_group = []
            curr_group_size = 0
        curr_group.append(shard_name)
    if len(curr_group) > 0: groups.append((curr_group_size, curr_group))
    return groups

def write_jobs(args, groups, script_path):
    output_folder = os.path.dirname(script_path)
    with open(args.sample_yaml_job_file) as fileptr:
        data = yaml.safe_load(fileptr)
    
    prefix_command = f"""bash examples/pretrain-llama/data-processing/merge_shard/remote_merge_shard.sh \\
\"{args.az_blob_input_folder}\" \\
\"{args.az_blob_output_folder_path}\" \\
\"{args.az_sas_token}\" \\
\"{args.prefix_name}"""
    for idx, (size, shards) in enumerate(groups):
        command = prefix_command
        command += f"{idx:03}.jsonl\""
        for shard in shards:
            command += f" \\\n\"{shard}\""
        # print(command)
        if args.dry_run:
            # print(f"RUN {size}: {command}")
            pass
        else:
            data['command'] = command
            az_yaml_file = os.path.join(output_folder, 'output.yaml')
            with open(az_yaml_file, 'w') as wrt_ptr:
                yaml.dump(data, wrt_ptr, default_flow_style=False)
            cmd = f"az ml job create --subscription {args.az_subscription} --resource-group {args.az_resource_group} --workspace-name {args.az_workspace_name} --file {az_yaml_file}"
            subprocess.check_output(cmd, shell=True)
    for idx, (size, _) in enumerate(groups):
        print(f"[{args.prefix_name}][{idx}] {size/1000000000}GB")


if __name__ == "__main__":
    script_path = os.path.abspath(__file__)
    args =  get_args()
    shards = azcopy_list(args.az_blob_input_folder, args.az_sas_token)
    groups = group_shards(shards, args.shard_size)
    write_jobs(args, groups, script_path)


    