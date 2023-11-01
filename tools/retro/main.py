# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.

"""Preprocess data for Retro.

Stages (see argument '--retro-tasks'):
- Build chunk database (DB).
- Build index (train, add).
- Query pretraining neighbors.
"""

import json
import os
import torch

from megatron import get_args, initialize_megatron, print_rank_0
from megatron.global_vars import set_retro_args
from tools.retro.db import build_db
from tools.retro.index import add_to_index, build_index, train_index
from tools.retro.query import query_pretraining_neighbors
from tools.retro.utils import get_args_path


def add_retro_args(parser):
    """Retro preprocesing arguments.

    *Note* : Arguments prefixed with '--retro-gpt-*' or '--retro-bert-*' are
    included and named as such to more easily handle managing both models
    running at the same time. Megatron is not optimized to run two models at
    once, so this naming convention makes it clearer.
    """

    group = parser.add_argument_group(title="Retro preprocessing.")

    # Basic args.
    group.add_argument("--retro-tasks", default="build",
                       help="Comma-separated list of tasks to run. Run entire "
                       "preprocesing pipeline by using '--retro-tasks build'. "
                       "Alternatively, run individual stages with tasks (in "
                       "this order) 'db-build', 'index-build', or "
                       "'query-pretraining-neighbors'. For example, "
                       "'--retro-tasks db-build,index-build,"
                       "query-pretraining-neighbors' is equivalent to "
                       "'--retro-tasks build'; or the argument can contain "
                       "a subset of these tasks. Stages must always be run "
                       "in the correct order (listed above).")
    group.add_argument("--retro-block-size", type=int, default=100000,
                       help="Number of chunks to process at a time when "
                       "generating Bert embeddings and querying the search "
                       "index. Partial results for each block are generally "
                       "saved to disk in separate files.")
    group.add_argument("--retro-doc-block-size", type=int, default=100000,
                       help="Number of documents to processe at time when "
                       "processing token datasets into chunk databases. The "
                       "partial chunk database for each block is saved into "
                       "a separate file.")

    # GPT args.
    group.add_argument('--retro-gpt-seed', type=int, default=1234,
                       help='Random seed used for python, numpy, '
                       'pytorch, and cuda.')
    group.add_argument('--retro-gpt-data-path', nargs='*', required=True,
                       help='Path to the training dataset. Accepted format:'
                       '1) a single data path, 2) multiple datasets in the'
                       'form: dataset1-weight dataset1-path dataset2-weight '
                       'dataset2-path ... It is used with --split when a '
                       'single dataset used for all three: train, valid '
                       'and test. It is exclusive to the other '
                       '--*-data-path args')
    group.add_argument('--retro-gpt-split', type=str, default='969,30,1',
                       help='Comma-separated list of proportions for training,'
                       ' validation, and test split. For example the split '
                       '`90,5,5` will use 90%% of data for training, 5%% for '
                       'validation and 5%% for test.')
    group.add_argument("--retro-gpt-eval-interval", type=int, required=True,
                       help="GPT evaluation interval.")
    group.add_argument("--retro-gpt-eval-iters", type=int, required=True,
                       help="GPT evaluation iterations.")
    group.add_argument("--retro-gpt-tokenizer-type", required=True,
                       help="GPT tokenizer type.")
    group.add_argument("--retro-gpt-vocab-file", help="GPT vocab file.")
    group.add_argument("--retro-gpt-merge-file", help="GPT merge file.")
    group.add_argument("--retro-gpt-tokenizer-model",
                       help="GPT tokenizer model file.")
    group.add_argument("--retro-gpt-seq-length", type=int, required=True,
                       help="GPT sequence length.")
    group.add_argument("--retro-gpt-global-batch-size", type=int, required=True,
                       help="GPT global batch size.")
    group.add_argument("--retro-gpt-chunk-length", type=int, default=64,
                       help="GPT chunk length.")

    # Bert args.
    group.add_argument("--retro-bert-vocab-file", required=True,
                       help="Bert vocab file.")
    group.add_argument("--retro-bert-tokenizer-type", required=True,
                       help="Bert tokenizer type (for when using "
                       "'--bert-embedder-type megatron').")
    group.add_argument("--retro-bert-batch-size", type=int, default=128,
                       help="Micro-batch size for processing Bert embeddings.")
    group.add_argument("--retro-bert-max-chunk-length", type=int, default=256,
                       help="Maximum sequence length for Bert embeddings. "
                       "(Named 'chunk' here in reference to these Bert "
                       "sequences being converted from GPT chunks.)")

    # Index args.
    group.add_argument("--retro-index-nfeats", "-f", type=int, default=1024,
                       help="Dimension of Bert embeddings. Bert-large is "
                       "commonly used, so this value defaults to 1024.")
    group.add_argument("--retro-index-type", default="faiss-par-add",
                       choices=["faiss-base", "faiss-par-add"],
                       help="A 'faiss-base' index is a simple, un-optimized "
                       "wrapper around a Faiss index. A 'faiss-par-add' index "
                       "optimizes the 'add()' method by making it multi-node "
                       "and multi-process, but with bit-wise equivalent "
                       "results.")
    group.add_argument("--retro-index-str", required=True,
                       help="Index string used for calling "
                       "faiss.index_factory(). For example, "
                       "'IVF262144_HNSW32,Flat' or "
                       "'OPQ32_256,IVF4194304_HNSW32,PQ32'.")
    group.add_argument("--retro-index-ntrain", type=int, required=True,
                       help="Number of database chunks to use for training "
                       "the index. This value must be less or equal to the "
                       "total number of chunks in the database.")
    group.add_argument("--retro-index-train-load-fraction",
                       type=float, default=1.,
                       help="Fraction of sampled chunks to use for training "
                       "the index. Useful when our total sampled embeddings "
                       "use too much memory; lowering the load fraction is "
                       "less costly than re-embedding a new sampled dataset "
                       "from scratch.")
    group.add_argument("--retro-index-add-load-fraction",
                       type=float, default=1.,
                       help="Fraction of database chunks to use for adding to "
                       "the index. Useful when our total index size would "
                       "use too much memory; lowering the load fraction is "
                       "less costly than re-designing our token datasets.")
    group.add_argument("--retro-index-no-delete-training-embeddings",
                       action='store_false',
                       dest="retro_index_delete_training_embeddings",
                       help="Skip deleting training embeddings for the search "
                       "index. Useful for debugging.")
    group.add_argument("--retro-index-no-delete-added-codes",
                       action='store_false',
                       dest="retro_index_delete_added_codes",
                       help="Skip deleting added codes for the search "
                       "index. Useful for debugging.")

    # Query args.
    group.add_argument("--retro-query-ef-search", type=int, default=256,
                       help="Index ef-search parameter for HNSW during querying.")
    group.add_argument("--retro-query-nprobe", type=int, default=65536,
                       help="Index nprobe parameter for IVF during querying.")
    group.add_argument("--retro-query-num-neighbors-query", type=int, default=200,
                       help="Number of neighbors to retrieve when calling "
                       "index.search().")
    group.add_argument("--retro-query-num-neighbors-save", type=int, default=20,
                       help="Number of neighbors to save to disk after "
                       "the index's returned neighbors. If longer than target "
                       "value, neighbors truncated; and if shorter than target "
                       "value, neighbors are padded with -1's.")

    # Enforce argument naming convention.
    for action in group._group_actions:
        prefix = action.dest.split("_")[0]
        assert prefix == "retro", \
            "Retro args must be prefixed with '--retro-*', for consistent " \
            "styling. Please fix '%s'." % ", ".join(action.option_strings)

    return parser


def save_args(args):
    '''Save copy of args within retro workdir.'''

    def default_dump(obj):
        if isinstance(obj, torch.dtype):
            return str(obj)
        else:
            raise Exception("specialize for <%s>." % type(obj).__name__)

    if torch.distributed.get_rank() == 0:
        args_path = get_args_path(args.retro_workdir)
        with open(args_path, "w") as f:
            json.dump(vars(args), f, indent=4, default=default_dump)

    torch.distributed.barrier()


if __name__ == "__main__":

    # Initalize Megatron.
    initialize_megatron(extra_args_provider=add_retro_args)

    # Split retro tasks.
    args = get_args()
    args.retro_tasks = args.retro_tasks.split(",")

    # Save/set retro args.
    os.makedirs(args.retro_workdir, exist_ok=True)
    save_args(args)
    set_retro_args(args)

    # Select task to run.
    for task in args.retro_tasks:

        print_rank_0("start '%s'." % task)

        # Run all stages.
        if task == "build":
            build_db()
            torch.distributed.barrier()
            build_index()
            torch.distributed.barrier()
            query_pretraining_neighbors()

        # DB (i.e., chunk db).
        elif task == "db-build":
            build_db()

        # Index.
        elif task == "index-build":
            build_index() # calls both train + add.
        elif task == "index-train":
            train_index() # train only
        elif task == "index-add":
            add_to_index() # add only

        # Pretraining.
        elif task == "query-pretraining-neighbors":
            query_pretraining_neighbors()

        else:
            raise Exception("specialize for task '%s'." % task)

        torch.distributed.barrier()

        print_rank_0("end '%s'." % task)
