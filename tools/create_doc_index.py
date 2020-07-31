import sys
sys.path.append('../')

from megatron.indexer import IndexBuilder
from megatron.initialize import initialize_megatron


def main():
    """Create a BlockData data structure by running an IndexBuilder over an ICT Dataset
    - Include all args needed for initial model specification

    Other key args:
        --block-data-path: path to write to
        --ict-load or --realm-load: path to checkpoint with which to embed
        --data-path and --titles-data-path: paths for dataset
        --indexer-log-interval: reporting interval
        --indexer-batch-size: size specific for indexer jobs

    Check README.md for example script
    """

    initialize_megatron(extra_args_provider=None,
                        args_defaults={'tokenizer_type': 'BertWordPieceLowerCase'})
    index_builder = IndexBuilder()
    index_builder.build_and_save_index()


if __name__ == "__main__":
    main()

