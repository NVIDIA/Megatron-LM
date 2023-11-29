import os
import json
import yaml
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

    group = parser.add_argument_group(title='I/O params.')
    group.add_argument("--input-folder-path", type = str, help="Compute target folder path to input jsonl files, each job process a single jsonl file.")
    group.add_argument("--bin-idx-folder-path", type = str, help="Compute target folder path to output folder for bin and idx")
    parser.add_argument("--tokenizer-module", type = str, default='megatron', choices=['megatron', 'nemo'], help="Which version of script will be used to tokenize the data.")
    group.add_argument("--tokenizer-model", type = str, help="Compute target path to the tokenizer.model file")
    group.add_argument("--tokenizer-type", type = str, help="Type of the tokenizer model", required=True)
    group.add_argument("--num-proc", type = int, help="Compute target number of workers per node", required=True)
    group.add_argument('--log-interval', type=int, default=10000, help='Loggin interval between tokenizer progress updates.')
    group.add_argument("--overwrite", action='store_true', help="Overwrite pre-existing bin-idx.")

    group = parser.add_argument_group(title='Misc. params.')
    parser.add_argument("--compute-target", type = str, default='azure', choices=['local', 'azure'], help="Conpute targets. Both --input-folder-path, --bin-idx-folder-path and --tokenizer-model should use same compute target. TODO: Enable cross compute.")
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

def local_submit_job(args):
    num_file_workers = os.cpu_count()//args.num_proc
    cmd = f"""python examples/data-processing/multiprocess_runner.py --glob-input-path {args.input_folder_path} --output-folder {args.bin_idx_folder_path} --tokenizer-model {args.tokenizer_model} --tokenizer-type {args.tokenizer_model} --num-proc {args.num_proc} --num-file-workers {num_file_workers}"""
    subprocess.check_output(cmd, shell=True)

def azure_submit_jobs(args, input_shard_dict, script_path):
    output_folder = os.path.dirname(script_path)
    with open(args.az_configs['az-sample-yaml-job-file']) as fileptr:
        data = yaml.safe_load(fileptr)
    sas_token = args.az_configs['az-sas-token']
    prefix_command = f"""bash examples/data-processing/remote_az_batch_tokenize.sh """
    for idx, (shard_name, size) in enumerate(input_shard_dict.items()):
        cmd= prefix_command
        cmd = cmd + f' \"{shard_name}\"'
        cmd = cmd + f' \"{args.input_folder_path}\"'
        cmd = cmd + f' \"{args.bin_idx_folder_path}\"'
        cmd = cmd + f' \"{args.tokenizer_module}\"'
        cmd = cmd + f' \"{args.tokenizer_type}\"'
        cmd = cmd + f' \"{args.tokenizer_model}\"'
        cmd = cmd + f' \"{args.num_proc}\"'
        cmd = cmd + f' \"{args.log_interval}\"'
        cmd = cmd + f' \"{sas_token}\"'
        
        print(f"RUN [{idx}][{shard_name}][{size/1000000000}GB]: {cmd}")
        if not args.dry_run:
            data['command'] = cmd
            az_yaml_file = os.path.join(output_folder, 'output.yaml')
            with open(az_yaml_file, 'w') as wrt_ptr:
                yaml.dump(data, wrt_ptr, default_flow_style=False)
            cmd = f"az ml job create --subscription {args.az_configs['az-subscription']} --resource-group {args.az_configs['az-resource-group']} --workspace-name {args.az_configs['az-workspace-name']} --file {az_yaml_file}"
            subprocess.check_output(cmd, shell=True)

def list_files_with_size(folder_path):
    file_path_with_size = {}
    print(folder_path)
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            file_path = os.path.join(root, file)
            size = os.path.getsize(file_path)
            file_path_with_size[file_path] = size
    return file_path_with_size

def submit_jobs(args, input_shard_dict, script_path):
    if args.compute_target == "local":
        local_submit_job(args)
    elif args.compute_target == "azure":
        return azure_submit_jobs(args, input_shard_dict, script_path)
    else:
        raise NotImplementedError()

def get_shard_info(args, shard_folder):
    if args.compute_target == "local":
        return list_files_with_size(shard_folder)
    elif args.compute_target == "azure":
        return azcopy_list(shard_folder, args.az_configs['az-sas-token'])
    else:
        raise NotImplementedError()

if __name__ == "__main__":
    script_path = os.path.abspath(__file__)
    args =  get_args()
    input_shard_dict = get_shard_info(args, args.input_folder_path)
    print(input_shard_dict)
    # output_shards_dict = get_shard_info(args, args.bin_idx_folder_path)
    # if not args.overwrite:
    #     input_shard_dict = match_pre_existing_bin_idx(input_shard_dict, output_shards_dict)
    # submit_jobs(args, input_shard_dict, script_path)
