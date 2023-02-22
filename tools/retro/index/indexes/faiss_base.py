# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.

"""
This class implements a simple, un-optimized wrapper around a Faiss index, that
implements the Index interface (see ..index.py). While this class is
instantiable, it is meant to be extended with optimizations in classes that
inherit from this class (see FaissParAddIndex, for an example).
"""

from datetime import timedelta
import os
import torch
from tqdm import tqdm

from megatron import get_retro_args, print_rank_0
from tools.bert_embedding import BertEmbedder
from tools.retro.external_libs import faiss
from tools.retro.index import Index
from tools.retro.index.utils import num_samples_to_block_ranges


class FaissBaseIndex(Index):

    def _train(self, input_data_loader):
        '''Train index (rank 0's method).'''

        args = get_retro_args()

        assert torch.distributed.get_rank() == 0

        # Set num threads (torch.distributed reset it to 1).
        # faiss.omp_set_num_threads(32)
        faiss.omp_set_num_threads(64)
        # faiss.omp_set_num_threads(128)

        empty_index_path = self.get_empty_index_path()

        # Index already exists? -> return.
        if os.path.isfile(empty_index_path):
            return

        # Load data.
        inp = input_data_loader()

        # Init index.
        index = faiss.index_factory(args.retro_index_nfeats,
                                    args.retro_index_str)

        # Move to GPU.
        index_ivf = faiss.extract_index_ivf(index)
        clustering_index = \
            faiss.index_cpu_to_all_gpus(faiss.IndexFlatL2(index_ivf.d))
        index_ivf.clustering_index = clustering_index
        self.c_verbose(index, True)
        self.c_verbose(index_ivf, True)
        self.c_verbose(index_ivf.quantizer, True)
        self.c_verbose(index_ivf.clustering_index, True)

        # Train index.
        index.train(inp)

        # Save index.
        faiss.write_index(index, empty_index_path)

    def train(self, input_data_loader):
        '''Train index.'''

        # Single process only.
        if torch.distributed.get_rank() == 0:
            self._train(input_data_loader)

        torch.distributed.barrier()

    def _add(self, text_dataset):
        '''Add to index (rank 0's method).'''

        assert torch.distributed.get_rank() == 0

        args = get_retro_args()

        dataset_sample_ranges = num_samples_to_block_ranges(len(text_dataset))

        # Set num threads (torch.distributed reset it to 1).
        faiss.omp_set_num_threads(64)

        # Bert embedder.
        embedder = BertEmbedder(args.retro_bert_batch_size,
                                args.retro_bert_max_chunk_length,
                                args.bert_embedder_type)

        # Empty/added index paths.
        empty_index_path = self.get_empty_index_path()
        added_index_path = self.get_added_index_path()

        # Skip adding, if index exists.
        if os.path.isfile(added_index_path):
            return

        # Read trained index.
        index = faiss.read_index(empty_index_path)

        # Iterate data blocks & add.
        for sample_range in tqdm(dataset_sample_ranges, "faiss_base.add"):

            # Embed text.
            embeds = self.embed_text_dataset_block(
                embedder, text_dataset, sample_range)

            # Add to index.
            index.add(embeds)

        # Write index.
        faiss.write_index(index, added_index_path)

    def add(self, text_dataset):
        '''Add to index.'''

        # Single process only.
        if torch.distributed.get_rank() == 0:
            self._add(text_dataset)

        # Wait for rank 0.
        torch.distributed.barrier()

        # Get output index path, for return.
        return self.get_added_index_path()
