import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))

import json
import glob
import tqdm
import argparse
from concurrent.futures.process import ProcessPoolExecutor
from functools import partial
from megatron.tokenizer import build_tokenizer



def chat_prompt(args, tokenizer, line):
    data = json.loads(line)
    template = args.chatml_template
    try:
        _dict_keys = {key: data[key] for key in args.json_keys}
        bos_token = tokenizer.tokenizer.id_to_piece(tokenizer.bos_id)
        eos_token = tokenizer.tokenizer.id_to_piece(tokenizer.eos_id)
        format_data = template.format(
            bos_token = bos_token,
            eos_token = eos_token,
            **_dict_keys
        )
    except KeyError:
        raise ValueError(f"data doesn't have same keys as json-keys, {args.json_keys=} {json.dumps(data, indent=4)}")
    except:
        raise ValueError(f"Template Formatting Error: \
                            {bos_token=}, {eos_token=}, \
                            {args.json_keys=}, \
                            {args.chatml_template=}")
    data[args.output_json_key] = format_data
    return json.dumps(data)


def get_args():
    parser = argparse.ArgumentParser()
    group = parser.add_argument_group(title='input data')
    group.add_argument('--input', type=str, required=True,
                       help='Path to input JSON')
    group.add_argument('--json-keys', nargs='+', default=['text'],
                       help='space separate listed of keys to extract from json')
    group.add_argument('--chatml-template', type=str, default=None, required=True,
                       help='Additional prompt template for chat-formatted text. '
                            'Formatted text must match json keys in the input file.')
    group.add_argument('--output-json-key', type=str, default='tokennized_sample',
                       help='Name of the new json key after projecting relevant '
                            '--json-keys to the --chatml-template and then '
                            'apply tokenizer on it.')
    group = parser.add_argument_group(title='tokenizer')
    group.add_argument('--tokenizer-type', type=str, default='YTTMTokenizer',
                       choices=['BertWordPieceLowerCase','BertWordPieceCase',
                                'GPT2BPETokenizer', 'SentencePieceTokenizer',
                                'Llama2Tokenizer'],
                       help='What type of tokenizer to use.')
    group.add_argument('--tokenizer-model', type=str, default=None,
                       help='YTTM tokenizer model.')
    group.add_argument('--vocab-file', type=str, default=None,
                       help='Path to the vocab file')
    group.add_argument('--merge-file', type=str, default=None,
                       help='Path to the BPE merge file (if necessary).')
    group.add_argument('--workers', type=int, default=8,
                       help='Number of worker processes to launch')
    group.add_argument('--output-prefix', type=str, required=True,
                       help='Path to binary output file without suffix')
    args = parser.parse_args()

    # some default/dummy values for the tokenizer
    args.rank = 1
    args.make_vocab_size_divisible_by = 128
    args.tensor_model_parallel_size = 1
    args.vocab_extra_ids = 0

    return args


def main():
    args = get_args()

    in_file_names = glob.glob(args.input)
    tokenizer = build_tokenizer(args)
    _chat_prompt = partial(chat_prompt, args, tokenizer)
    for file_name in in_file_names:
        base_file_name = os.path.basename(file_name)
        output_file_name = args.output_prefix +base_file_name
        with open(file_name) as read_ptr, \
            open(output_file_name, "w") as wrt_ptr, \
            ProcessPoolExecutor(max_workers=args.workers) as executor_map:
            input_data = []
            for line in tqdm.tqdm(read_ptr, f"[READ] {base_file_name}"):
                input_data.append(line)
            for _out in tqdm.tqdm(
                executor_map.map(_chat_prompt, input_data, chunksize=2048), 
                desc=f"[PROJECT & WRITE] {base_file_name}"):
                wrt_ptr.write(f"{_out}\n")
                

if __name__ == '__main__':

    main()

