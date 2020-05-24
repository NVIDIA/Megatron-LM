from collections import defaultdict
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
    def __init__(self):
        self.embed_data = dict()
        self.meta_data = dict()
        self.temp_dir_name = 'temp_block_data'

    def state(self):
        return {
            'embed_data': self.embed_data,
            'meta_data': self.meta_data
        }

    def clear(self):
        """Clear the data structures to save memory"""
        self.embed_data = dict()
        self.meta_data = dict()

    @classmethod
    def load_from_file(cls, fname):
        print("\n> Unpickling block data", flush=True)
        state_dict = pickle.load(open(fname, 'rb'))
        print(">> Finished unpickling block data\n", flush=True)

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

    def save_shard(self, rank):
        if not os.path.isdir(self.temp_dir_name):
            os.mkdir(self.temp_dir_name)

        # save the data for each shard
        with open('{}/{}.pkl'.format(self.temp_dir_name, rank), 'wb') as data_file:
            pickle.dump(self.state(), data_file)

    def consolidate_shards_and_save(self, ignore_shard=0):
        """Combine all the shards made using self.save_shard()"""
        fnames = os.listdir(self.temp_dir_name)
        for fname in fnames:
            with open('{}/{}'.format(self.temp_dir_name, fname), 'rb') as f:
                data = pickle.load(f)

                old_size = len(self.embed_data)
                shard_size = len(data['embed_data'])
                self.embed_data.update(data['embed_data'])
                self.meta_data.update(data['meta_data'])
                # assert (len(self.embed_data) == old_size + shard_size) or (str(ignore_shard) in fname)

        args = get_args()
        with open(args.block_data_path, 'wb') as final_file:
            pickle.dump(self.state(), final_file)
        shutil.rmtree(self.temp_dir_name, ignore_errors=True)


class FaissMIPSIndex(object):
    def __init__(self, index_type, embed_size, use_gpu=False):
        self.index_type = index_type
        self.embed_size = embed_size
        self.use_gpu = use_gpu
        self.id_map = dict()

        # alsh
        self.m = 5
        self.u = 0.99
        self.max_norm = None
        self.block_mips_index = None
        self._set_block_index()

    def _set_block_index(self):
        INDEX_TYPES = ['flat_ip']
        if self.index_type not in INDEX_TYPES:
            raise ValueError("Invalid index type specified")

        print("\n> Building index", flush=True)
        self.block_mips_index = faiss.index_factory(self.embed_size, 'Flat', faiss.METRIC_INNER_PRODUCT)
        if not self.use_gpu:
            self.block_mips_index = faiss.IndexIDMap(self.block_mips_index)
        print(">> Finished building index", flush=True)

        if self.use_gpu:
            res = faiss.StandardGpuResources()
            # self.block_mips_index = faiss.index_cpu_to_gpu(res, device, self.block_mips_index)
            config = faiss.GpuIndexFlatConfig()
            config.device = torch.cuda.current_device()
            config.useFloat16 = True
            self.block_mips_index = faiss.GpuIndexFlat(res, self.block_mips_index, config)
            print(">>> Loaded Faiss index on GPU {}\n".format(self.block_mips_index.getDevice()), flush=True)

    def reset_index(self):
        self._set_block_index()

    def add_block_embed_data(self, all_block_data, clear_block_data=False):
        """Add the embedding of each block to the underlying FAISS index"""
        block_indices, block_embeds = zip(*all_block_data.embed_data.items())
        if self.use_gpu:
            for i, idx in enumerate(block_indices):
                self.id_map[i] = idx
        if clear_block_data:
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
        if self.index_type == 'flat_l2':
            query_embeds = self.alsh_query_preprocess_fn(query_embeds)
        query_embeds = np.float32(detach(query_embeds))
        # query_embeds = query_embeds.float()

        with torch.no_grad():
            if reconstruct:
                top_k_block_embeds = self.block_mips_index.search_and_reconstruct(query_embeds, top_k)
                return top_k_block_embeds
            else:
                distances, block_indices = self.block_mips_index.search(query_embeds, top_k)
                if self.use_gpu:
                    fresh_indices = np.zeros(block_indices.shape)
                    for i in range(block_indices.shape[0]):
                        for j in range(block_indices.shape[1]):
                            fresh_indices[i, j] = self.id_map[block_indices[i, j]]
                    block_indices = fresh_indices
                return distances, block_indices

    # functions below are for ALSH, which currently isn't being used

    def get_norm_powers_and_halves_array(self, embeds):
        norm = np.linalg.norm(embeds, axis=1)
        norm_powers = [np.multiply(norm, norm)]  # squared L2 norms of all
        for i in range(self.m - 1):
            norm_powers.append(np.multiply(norm_powers[-1], norm_powers[-1]))
        # [num_blocks x self.m]
        norm_powers = np.transpose(np.array(norm_powers))
        halves_array = 0.5 * np.ones(norm_powers.shape)

        return norm_powers, halves_array

    def alsh_block_preprocess_fn(self, block_embeds):
        block_embeds = np.array(block_embeds)
        if self.max_norm is None:
            self.max_norm = max(np.linalg.norm(block_embeds, axis=1))
        if self.max_norm > 1:
            block_embeds = self.u / self.max_norm * block_embeds
        norm_powers, halves_array = self.get_norm_powers_and_halves_array(block_embeds)

        # P'(S(x)) for all x in block_embeds
        return np.float32(np.concatenate((block_embeds, norm_powers, halves_array), axis=1))

    def alsh_query_preprocess_fn(self, query_embeds):
        max_norm = max(np.linalg.norm(query_embeds, axis=1))
        if max_norm > 1:
            query_embeds = self.u / max_norm * query_embeds
        norm_powers, halves_array = self.get_norm_powers_and_halves_array(query_embeds)

        # Q'(S(x)) for all x in query_embeds
        return np.float32(np.concatenate((query_embeds, halves_array, norm_powers), axis=1))


# This was the original hashing scheme, not used anymore

class RandProjectionLSHIndex(object):
    """Class for holding hashed data"""
    def __init__(self, embed_size, num_buckets, whiten=True, seed=0):
        np.random.seed(seed)
        self.hash_data = defaultdict(list)
        hash_matrix = 2 * np.random.rand(embed_size, int(num_buckets / 2)) - 1
        self.hash_matrix = hash_matrix / np.linalg.norm(hash_matrix, axis=0).reshape(1, -1)
        self.embed_mean = None
        self.embed_whitener = None
        self.whiten = whiten

    def state(self):
        state = {
            'hash_data': self.hash_data,
            'hash_matrix': self.hash_matrix,
            'embed_mean': self.embed_mean,
            'embed_whitener': self.embed_whitener,
        }
        return state

    def save_to_file(self):
        args = get_args()
        with open(args.block_index_path, 'wb') as index_file:
            pickle.dump(self.state(), index_file)

    @classmethod
    def load_from_file(cls, fname):
        print(" > Unpickling block hash data")
        state_dict = pickle.load(open(fname, 'rb'))
        print(" > Finished unpickling")
        hash_matrix = state_dict['hash_matrix']

        new_index = cls(hash_matrix.shape[0], hash_matrix.shape[1] * 2)
        new_index.hash_data = state_dict['hash_data']
        new_index.embed_mean = state_dict.get('embed_mean')
        new_index.embed_whitener = state_dict.get('embed_whitener')
        new_index.hash_matrix = hash_matrix

        return new_index

    def get_block_bucket(self, hash):
        return self.hash_data[hash]

    def hash_embeds(self, embeds, write_block_data=None):
        """Hash a tensor of embeddings using a random projection matrix"""
        embed_scores_pos = torch.matmul(embeds, torch.cuda.FloatTensor(self.hash_matrix).type(embeds.dtype))
        embed_scores = torch.cat((embed_scores_pos, -embed_scores_pos), axis=1)
        embed_hashes = detach(torch.argmax(embed_scores, axis=1))

        if write_block_data is not None:
            for hash, indices in zip(embed_hashes, write_block_data):
                self.hash_data[hash].append(indices)

        return embed_hashes

    def hash_whitened_block_embeds(self, block_data):
        """Transform all block embeds to have zero mean and unit covariance
        when treated as samples from a distribution"""
        block_idx, all_embeds = zip(*block_data.embed_data.items())
        arr_embeds = np.transpose(np.array(all_embeds))

        mean = np.mean(arr_embeds, axis=1).reshape(-1, 1)
        centered = arr_embeds - mean
        inv_cov = np.linalg.inv(np.cov(arr_embeds))
        whitener = np.transpose(np.linalg.cholesky(inv_cov))
        whitened = np.float16(np.transpose(whitener.dot(centered)))

        self.embed_mean = mean.reshape(-1)
        self.embed_whitener = whitener
        self.hash_data = defaultdict(list)
        batch_size = 16384
        i = 0

        args = get_args()
        with torch.no_grad():
            while True:
                if args.debug:
                    print(i, flush=True)
                batch_slice = slice(i * batch_size, (i + 1) * batch_size)
                batch_embed = torch.cuda.HalfTensor(whitened[batch_slice])
                batch_meta = [block_data.meta_data[idx] for idx in block_idx[batch_slice]]
                if len(batch_meta) == 0:
                    break

                self.hash_embeds(batch_embed, batch_meta)
                i += 1

    def exact_mips_equals(self, query_embeds, all_block_data, norm_blocks):
        """For each query, determine whether the mips block is in the correct hash bucket"""
        shuffled_block_idx, block_embeds = zip(*all_block_data.items())
        if norm_blocks:
            block_embeds = block_embeds / np.linalg.norm(block_embeds, axis=1).reshape(-1, 1)
        with torch.no_grad():
            query_hashes = self.hash_embeds(query_embeds)

            # [num_query x num_blocks]
            inner_products = torch.matmul(torch.cuda.HalfTensor(query_embeds),
                                          torch.cuda.HalfTensor(np.transpose(np.array(block_embeds))))
            max_inner_product_idxes = detach(torch.argmax(inner_products, axis=1))
            best_blocks = np.array([all_block_data[shuffled_block_idx[idx]] for idx in max_inner_product_idxes])
            best_block_hashes = self.hash_embeds(best_blocks)

            print('Query hashes: ', query_hashes)
            print('Block hashes: ', best_block_hashes)
            equal_arr = np.equal(query_hashes, best_block_hashes).astype(int)

            # array of zeros and ones which can be used for counting success
            return equal_arr

    def exact_mips_test(self, num_queries, all_block_data, norm_blocks):
        if self.whiten:
            if self.embed_mean is None:
                self.hash_whitened_block_embeds(all_block_data)
            embed_size = self.hash_matrix.shape[0]
            query_embeds = np.random.multivariate_normal(np.zeros(embed_size), np.eye(embed_size), num_queries)
            query_embeds = query_embeds / np.linalg.norm(query_embeds, axis=1).reshape(-1, 1)
        else:
            block_idx, all_embeds = zip(*all_block_data.items())
            arr_embeds = np.transpose(np.array(all_embeds))

            mean = np.mean(arr_embeds, axis=1).reshape(-1, 1)
            cov = np.cov(arr_embeds)
            query_embeds = np.random.multivariate_normal(mean, cov, num_queries)

        equal_arr = self.exact_mips_equals(query_embeds, all_block_data, norm_blocks)
        print("Num correct: ", sum(equal_arr), " Fraction correct: ", sum(equal_arr) / equal_arr.size)
        print(equal_arr)

