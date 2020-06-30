from collections import defaultdict
import itertools
import os
import pickle
import shutil

import faiss
import numpy as np
import torch

from megatron import get_args, mpu


def detach(tensor):
    return tensor.detach().cpu().numpy()


class BlockData(object):
    """Serializable data structure for holding data for blocks -- embeddings and necessary metadata for REALM"""
    def __init__(self, block_data_path=None, rank=None):
        self.embed_data = dict()
        self.meta_data = dict()
        if block_data_path is None:
            args = get_args()
            block_data_path = args.block_data_path
            rank = args.rank
        self.block_data_path = block_data_path
        self.rank = rank

        block_data_name = os.path.splitext(self.block_data_path)[0]
        self.temp_dir_name = block_data_name + '_tmp'

    def state(self):
        return {
            'embed_data': self.embed_data,
            'meta_data': self.meta_data,
        }

    def clear(self):
        """Clear the embedding data structures to save memory.
        The metadata ends up getting used, and is also much smaller in dimensionality
        so it isn't really worth clearing.
        """
        self.embed_data = dict()

    @classmethod
    def load_from_file(cls, fname):
        print("\n> Unpickling BlockData", flush=True)
        state_dict = pickle.load(open(fname, 'rb'))
        print(">> Finished unpickling BlockData\n", flush=True)

        new_index = cls()
        new_index.embed_data = state_dict['embed_data']
        new_index.meta_data = state_dict['meta_data']
        return new_index

    def add_block_data(self, block_indices, block_embeds, block_metas, allow_overwrite=False):
        for idx, embed, meta in zip(block_indices, block_embeds, block_metas):
            if not allow_overwrite and idx in self.embed_data:
                raise ValueError("Unexpectedly tried to overwrite block data")

            self.embed_data[idx] = np.float16(embed)
            self.meta_data[idx] = meta

    def save_shard(self):
        if not os.path.isdir(self.temp_dir_name):
            os.makedirs(self.temp_dir_name, exist_ok=True)

        # save the data for each shard
        with open('{}/{}.pkl'.format(self.temp_dir_name, self.rank), 'wb') as data_file:
            pickle.dump(self.state(), data_file)

    def merge_shards_and_save(self):
        """Combine all the shards made using self.save_shard()"""
        shard_names = os.listdir(self.temp_dir_name)
        seen_own_shard = False

        for fname in os.listdir(self.temp_dir_name):
            shard_rank = int(os.path.splitext(fname)[0])
            if shard_rank == self.rank:
                seen_own_shard = True
                continue

            with open('{}/{}'.format(self.temp_dir_name, fname), 'rb') as f:
                data = pickle.load(f)
                old_size = len(self.embed_data)
                shard_size = len(data['embed_data'])

                # add the shard's data and check to make sure there is no overlap
                self.embed_data.update(data['embed_data'])
                self.meta_data.update(data['meta_data'])
                assert len(self.embed_data) == old_size + shard_size

        assert seen_own_shard

        # save the consolidated shards and remove temporary directory
        with open(self.block_data_path, 'wb') as final_file:
            pickle.dump(self.state(), final_file)
        shutil.rmtree(self.temp_dir_name, ignore_errors=True)

        print("Finished merging {} shards for a total of {} embeds".format(
            len(shard_names), len(self.embed_data)), flush=True)


class FaissMIPSIndex(object):
    """Wrapper object for a BlockData which similarity search via FAISS under the hood"""
    def __init__(self, index_type, embed_size, use_gpu=False):
        self.index_type = index_type
        self.embed_size = embed_size
        self.use_gpu = use_gpu
        self.id_map = dict()

        self.block_mips_index = None
        self._set_block_index()

    def _set_block_index(self):
        INDEX_TYPES = ['flat_ip']
        if self.index_type not in INDEX_TYPES:
            raise ValueError("Invalid index type specified")

        print("\n> Building index", flush=True)
        self.block_mips_index = faiss.index_factory(self.embed_size, 'Flat', faiss.METRIC_INNER_PRODUCT)

        if self.use_gpu:
            # create resources and config for GpuIndex
            res = faiss.StandardGpuResources()
            config = faiss.GpuIndexFlatConfig()
            config.device = torch.cuda.current_device()
            config.useFloat16 = True

            self.block_mips_index = faiss.GpuIndexFlat(res, self.block_mips_index, config)
            print(">>> Finished building index on GPU {}\n".format(self.block_mips_index.getDevice()), flush=True)
        else:
            # CPU index supports IDs so wrap with IDMap
            self.block_mips_index = faiss.IndexIDMap(self.block_mips_index)
            print(">> Finished building index\n", flush=True)

    def reset_index(self):
        """Delete existing index and create anew"""
        del self.block_mips_index
        self._set_block_index()

    def add_block_embed_data(self, all_block_data):
        """Add the embedding of each block to the underlying FAISS index"""
        block_indices, block_embeds = zip(*all_block_data.embed_data.items())
        if self.use_gpu:
            for i, idx in enumerate(block_indices):
                self.id_map[i] = idx

        all_block_data.clear()
        if self.use_gpu:
            self.block_mips_index.add(np.float32(np.array(block_embeds)))
        else:
            self.block_mips_index.add_with_ids(np.float32(np.array(block_embeds)), np.array(block_indices))

    def search_mips_index(self, query_embeds, top_k, reconstruct=True):
        """Get the top-k blocks by the index distance metric.

        :param reconstruct: if True: return a [num_queries x k x embed_dim] array of blocks
                            if False: return [num_queries x k] array of distances, and another for indices
        """
        query_embeds = np.float32(detach(query_embeds))

        with torch.no_grad():
            if reconstruct:
                top_k_block_embeds = self.block_mips_index.search_and_reconstruct(query_embeds, top_k)
                return top_k_block_embeds
            else:
                distances, block_indices = self.block_mips_index.search(query_embeds, top_k)
                if self.use_gpu:
                    fresh_indices = np.zeros(block_indices.shape)
                    for i, j in itertools.product(block_indices.shape):
                        fresh_indices[i, j] = self.id_map[block_indices[i, j]]
                    block_indices = fresh_indices
                return distances, block_indices
