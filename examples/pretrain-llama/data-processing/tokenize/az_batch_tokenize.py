import os
import json
import yaml
import argparse 
import subprocess

def get_args():
    parser = argparse.ArgumentParser(description="Azure batch job submission for tokenization")
    parser.add_argument("--az-subscription", type = str, help="Azure subscription id.", required=True)
    parser.add_argument("--az-resource-group", type = str, help="Azure subscription id.", required=True)
    parser.add_argument("--az-workspace-name", type = str, help="Azure subscription id.", required=True)
    parser.add_argument("--az-blob-input-folder-path", type = str, help="Azure blob folder path to input jsonl files, each job process a single jsonl file.", required=True)
    parser.add_argument("--az-blob-bin-idx-folder-path", type = str, help="Azure blob folder path to output folder for bin and idx", required=True)
    parser.add_argument("--az-tokenizer-model", type = str, help="Azure path to the tokenizer.model file", required=True)
    parser.add_argument("--tokenizer-type", type = str, help="Type of the tokenizer model", required=True)
    parser.add_argument("--az-sas-token", type = str, help="Azure blob SAS token", required=True)
    parser.add_argument("--az-num-proc", type = int, help="Number of workers per node", required=True)
    parser.add_argument('--az-log-interval', type=int, default=10000, help='Loggin interval between progress updates in azure node')
    parser.add_argument("--sample-yaml-job-file", type = str, help="Path to a sample job file.", required=True)
    parser.add_argument("--overwrite", action='store_true', help="Overwrite pre-existing bin-idx.")
    parser.add_argument("--dry-run", action='store_true', help="Simulate run before submitting jobs.")
    args = parser.parse_args()
    return args


def write_jobs(args):
    pass

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

def match_pre_existing_bin_idx(input_shard_dict, output_shards_dict):
    de_dup_shard_collection = {}
    for shard_name, n_byte in input_shard_dict.items():
        is_done = False
        for pre_exist_shard_name, _ in output_shards_dict.items():
            if pre_exist_shard_name.startswith(shard_name):
                print("Skipping...! Shard already in output folder. Potentially already processed by other process.")
                is_done = True
                break
        if is_done is False:
            de_dup_shard_collection[shard_name] = n_byte
    return de_dup_shard_collection

def write_jobs(args, input_shard_dict, script_path):
    output_folder = os.path.dirname(script_path)
    with open(args.sample_yaml_job_file) as fileptr:
        data = yaml.safe_load(fileptr)
    
    prefix_command = f"""bash examples/pretrain-llama/data-processing/tokenize/remote_az_batch_tokenize.sh """
    for idx, (shard_name, size) in enumerate(input_shard_dict.items()):
        cmd= prefix_command
        cmd = cmd + f' \"{shard_name}\"'
        cmd = cmd + f' \"{args.az_blob_input_folder_path}\"'
        cmd = cmd + f' \"{args.az_blob_bin_idx_folder_path}\"'
        cmd = cmd + f' \"{args.tokenizer_type}\"'
        cmd = cmd + f' \"{args.az_tokenizer_model}\"'
        cmd = cmd + f' \"{args.az_num_proc}\"'
        cmd = cmd + f' \"{args.az_log_interval}\"'
        cmd = cmd + f' \"{args.az_sas_token}\"'
        
        print(f"RUN [{idx}][{shard_name}][{size/1000000000}GB]: {cmd}")
        if not args.dry_run:
            data['command'] = cmd
            az_yaml_file = os.path.join(output_folder, 'output.yaml')
            with open(az_yaml_file, 'w') as wrt_ptr:
                yaml.dump(data, wrt_ptr, default_flow_style=False)
            # cmd = f"az ml job create --subscription {args.az_subscription} --resource-group {args.az_resource_group} --workspace-name {args.az_workspace_name} --file {az_yaml_file}"
            subprocess.check_output(cmd, shell=True)
        break

if __name__ == "__main__":
    script_path = os.path.abspath(__file__)
    args =  get_args()
    input_shard_dict = azcopy_list(args.az_blob_input_folder_path, args.az_sas_token)
    output_shards_dict = azcopy_list(args.az_blob_bin_idx_folder_path, args.az_sas_token)
    if not args.overwrite:
        input_shard_dict = match_pre_existing_bin_idx(input_shard_dict, output_shards_dict)
    write_jobs(args, input_shard_dict, script_path)
    