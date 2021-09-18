import torch
import sys
import os
from collections import OrderedDict


def dump_data(datum, name_list=[]):
    if type(datum) in (dict, OrderedDict):
        for k, v in datum.items():
            dump_data(v, name_list+[str(k)])
    elif type(datum) in (list, tuple):
        for v in datum:
            dump_data(v, name_list)
    elif torch.is_tensor(datum):
        prefix = '.'.join(name_list)
        print(f'[tensor] {prefix} = {datum.shape}')
    else:
        #pass 
        prefix = '.'.join(name_list)
        print(f'[other] {prefix} = {datum}')

def main():
    if len(sys.argv) < 2:
        print(f'Usage: {sys.argv[0]} <checkpoint file>')
        exit(1)

    ckpt_file = sys.argv[1]
    if not os.path.isfile(ckpt_file):
        print(f'{ckpt_file} is not a valid file')
        exit(1)

    print(f'loading checkpoint file: {ckpt_file}')
    sd = torch.load(ckpt_file)
    dump_data(sd)

    quit()


if __name__ == "__main__":
    main()
