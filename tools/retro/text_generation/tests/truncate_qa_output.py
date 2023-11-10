#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sys


# In[2]:


import argparse

def get_args():
    parser = argparse.ArgumentParser()
    group = parser.add_argument_group(title='input data')
    group.add_argument('--input', type=str, required=False,
                       help='Path to input JSON')
    group.add_argument('--json-keys', nargs='+', default=['text'],
                       help='space separate listed of keys to extract from json')
    group.add_argument('--split-sentences', action='store_true',
                       help='Split documents into sentences.')
    group.add_argument('--keep-newlines', action='store_true',
                       help='Keep newlines between sentences when splitting.')

    group = parser.add_argument_group(title='tokenizer')
    group.add_argument('--tokenizer-type', type=str, required=False,
                       choices=['BertWordPieceLowerCase','BertWordPieceCase',
                                'GPT2BPETokenizer'],
                       help='What type of tokenizer to use.')
    group.add_argument('--vocab-file', type=str, default=None,
                       help='Path to the vocab file')
    group.add_argument('--merge-file', type=str, default=None,
                       help='Path to the BPE merge file (if necessary).')
    group.add_argument('--append-eod', action='store_true',
                       help='Append an <eod> token to the end of a document.')


    group = parser.add_argument_group(title='output data')
    group.add_argument('--output-prefix', type=str, required=False,
                       help='Path to binary output file without suffix')
    group.add_argument('--dataset-impl', type=str, default='mmap',
                       choices=['lazy', 'cached', 'mmap'])

    group = parser.add_argument_group(title='runtime')
    group.add_argument('--workers', type=int, default=1,
                       help='Number of worker processes to launch')
    group.add_argument('--log-interval', type=int, default=100,
                       help='Interval between progress updates')
    group.add_argument('-f', type=str, default='',
                   help='Make jupyter happy')
    args = parser.parse_args()
    args.keep_empty = False

#     if args.tokenizer_type.lower().startswith('bert'):
#         if not args.split_sentences:
#             print("Bert tokenizer detected, are you sure you don't want to split sentences?")

    # some default/dummy values for the tokenizer
    args.rank = 0
    args.make_vocab_size_divisible_by = 128
    args.tensor_model_parallel_size = 1
    args.vocab_extra_ids = 0

    return args

args = get_args()


# In[4]:


args.tokenizer_type = "GPT2BPETokenizer"
args.vocab_file = "../megatron-lm//gpt2-vocab.json"
args.merge_file = "../megatron-lm/gpt2-merges.txt"

prediction_files = []
ckpt_path = "/lustre/fsw/adlr/adlr-nlp/boxinw/github-version/retro/Megatron-LM/checkpoints/applications/gpt3-800m-pretraining-retro-fitting/"
prediction_files.append(ckpt_path + "retro-generate-short-nq_5_2_843m_test_greedy_0_20000_195312.txt")


# In[11]:




# In[12]:



def truncate_32(prediction_file):
    with open(prediction_file) as f:
        lines = f.readlines()
    print(len(lines))    
    tokens = [megatron_tokenizer.tokenize(line) for line in lines]    
    import numpy as np
    print(np.mean([len(token) for token in tokens]))
    truncated_tokens = [token[:32] for token in tokens]    
    new_lines = [megatron_tokenizer.detokenize(token) for token in truncated_tokens]

    with open(prediction_file + ".truncate32.txt", "w") as f:
        for line in new_lines:
            line = line[:line.find("<|endoftext|>")].strip().replace("\n", " ")
            f.write(line + '\n')
    print(prediction_file + ".truncate32.txt")


def truncate_20(prediction_file):
    with open(prediction_file) as f:
        lines = f.readlines()
    print(len(lines))    
    tokens = [megatron_tokenizer.tokenize(line) for line in lines]    
    import numpy as np
    print(np.mean([len(token) for token in tokens]))
    truncated_tokens = [token[:20] for token in tokens]    
    new_lines = [megatron_tokenizer.detokenize(token) for token in truncated_tokens]

    with open(prediction_file + ".truncate20.txt", "w") as f:
        for line in new_lines:
            line = line[:line.find("<|endoftext|>")].strip().replace("\n", " ")
            f.write(line + '\n')
    print(prediction_file + ".truncate20.txt")


# In[24]:


def truncate_10(prediction_file):
    with open(prediction_file) as f:
        lines = f.readlines()
    print(len(lines))    
    tokens = [megatron_tokenizer.tokenize(line) for line in lines]    
    import numpy as np
    print(np.mean([len(token) for token in tokens]))
    truncated_tokens = [token[:10] for token in tokens]    
    new_lines = [megatron_tokenizer.detokenize(token) for token in truncated_tokens]

    with open(prediction_file + ".truncate10.txt", "w") as f:
        for line in new_lines:
            line = line[:line.find("<|endoftext|>")].strip().replace("\n", " ")
            f.write(line + '\n')
    print(prediction_file + ".truncate10.txt")


# In[26]:

def truncate_period(prediction_file):
    with open(prediction_file) as f:
        lines = f.readlines()
    print(len(lines))

    with (open(prediction_file + ".period.txt", "w")) as f:
        for line in lines:
            line = line[:line.find(".")]
            # line = line[line.find(":") + 1:]
            line = line.strip().replace("\n", " ")
            f.write(line + '\n')
    print(prediction_file + ".period.txt")

for f in prediction_files:
    # truncate_32(f)
    # truncate_20(f)
    # truncate_10(f)
    truncate_period(f)


# In[ ]:




