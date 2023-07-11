"""
Use this script to retrieve the codeparrot dataset from hugging face and convert to a
json data file. This json will then be converted to a format understood by the Megatron LM
code.

Usage:
python retrieve_dataset_codeparrot.py codeparrot_data.json

"""
import sys
from datasets import load_dataset

def retrieve_code_parrot_data(output_path):
    train_data = load_dataset('codeparrot/codeparrot-clean-train', split='train')
    train_data.to_json(output_path, lines=True)

if __name__ == "__main__":
    retrieve_code_parrot_data(sys.argv[1])
