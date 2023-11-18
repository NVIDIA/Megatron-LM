import json
import argparse
from tqdm import tqdm
from transformers import LlamaTokenizer
from nemo.collections.nlp.data.language_modeling.megatron.gpt_dataset import get_indexed_dataset_

"""
example:

python detokenize.py \
    --source_prefix_path "en_books_books_split_02_76522573_wc.jsonl_text_document" \
    --tokenizer_path "tokenizer_7bv2.model"
"""

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source_prefix_path", type=str, required=True)
    parser.add_argument("--tokenizer_path", type=str, required=True)
    return parser.parse_args()

def detokenize(source_prefix_path: str, tokenizer_path: str):
    tokenizer = LlamaTokenizer.from_pretrained(tokenizer_path)
    dataset = get_indexed_dataset_(source_prefix_path, 'mmap', False)
    print(f'detokenizing to {source_prefix_path}.jsonl:')
    with open(source_prefix_path+'.jsonl', 'w') as f:
        for i in tqdm(range(len(dataset))):
            doc = list(dataset.get(i))
            original = tokenizer.decode(doc)
            f.write(f'{json.dumps(original)}\n')

def main():
    args = get_args()
    detokenize(args.source_prefix_path, args.tokenizer_path)

if __name__ == '__main__':
    main()