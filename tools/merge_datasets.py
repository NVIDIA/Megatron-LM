import os
import sys
import json
import argparse
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),
                                             os.path.pardir)))

from megatron.data import indexed_dataset


def main(args):

    prefixes = set()
    for basename in os.listdir(args.input):
        prefix, ext = os.path.splitext(basename)

        if prefix in prefixes:
            continue

        if not os.path.isfile(os.path.join(args.input, basename)):
            continue

        ext_pair = '.bin' if ext == '.idx' else '.idx'
        assert os.path.isfile(os.path.join(args.input, prefix) + ext_pair), \
               f'ERROR: {ext_pair} file not provided for {os.path.join(args.input, prefix)}'

        prefixes.add(prefix)

    builder = None
    for prefix in sorted(prefixes):
        if builder is None:
            dataset = indexed_dataset.make_dataset(os.path.join(args.input, prefix), 'infer')

            if isinstance(dataset, indexed_dataset.MMapIndexedDataset):
                builder = indexed_dataset.MMapIndexedDatasetBuilder(args.output_prefix + '.bin', dtype=dataset._index.dtype)
            else:
                builder = indexed_dataset.IndexedDatasetBuilder(args.output_prefix + '.bin')

            del dataset

        builder.merge_file_(os.path.join(args.input, prefix))

    builder.finalize(args.output_prefix + '.idx')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    group = parser.add_argument_group(title='input data')
    group.add_argument('--input', type=str, required=True,
                       help='Path to directory containing all document files to merge')

    group = parser.add_argument_group(title='output data')
    group.add_argument('--output-prefix', type=str, required=True,
                       help='Path to binary output file without suffix')

    args = parser.parse_args()

    assert os.path.isdir(args.input), \
           f'ERROR: {args.input} is not a directory or does not exist'

    assert os.path.isdir(os.path.dirname(args.output_prefix)), \
           f'ERROR: {os.path.dirname(args.output_prefix)} is not a directory or does not exist'

    main(args)

