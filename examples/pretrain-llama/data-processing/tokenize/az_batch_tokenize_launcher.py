import yaml
import argparse 
import subprocess

def get_args():
    parser = argparse.ArgumentParser(description="Azure batch job submission for tokenization")
    parser.add_argument("--az-blob-input-folder-path", type = str, help="Azure blob folder path to input jsonl files, each job process a single jsonl file.", required=True)
    parser.add_argument("--az-blob-bin-idx-folder-path", type = str, help="Azure blob folder path to output folder for bin and idx", required=True)
    parser.add_argument("--az-tokenizer-model", type = str, help="Azure path to the tokenizer.model file", required=True)
    parser.add_argument("--az-sas-token", type = str, help="Azure blob SAS token", required=True)
    parser.add_argument("--az-num-proc", type = int, help="Number of workers per node", required=True)
    parser.add_argument('--az-log-interval', type=int, default=10000, help='Loggin interval between progress updates in azure node')
    args = parser.parse_args()
    return args

def write_jobs(args):
    pass

def azcopy_list(path, sas_token):
    path_with_sas_token = f"{path}?{sas_token}"
    cmd = f"azcopy list {path_with_sas_token}"
    print(cmd)
    # subprocess.check_output("")
if __name__ == "__main__":
    args =  get_args()
    azcopy_list(args.az_blob_input_folder_path, args.az_sas_token)
    