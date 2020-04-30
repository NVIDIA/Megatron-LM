from collections import defaultdict
import os
import pickle
import shutil

import numpy as np
import torch
from torch.nn.parallel.distributed import DistributedDataParallel as torchDDP

from megatron import get_args
from megatron import mpu
from megatron.checkpointing import get_checkpoint_tracker_filename, get_checkpoint_name
from megatron.data.bert_dataset import get_indexed_dataset_
from megatron.data.ict_dataset import InverseClozeDataset
from megatron.data.samplers import DistributedBatchSampler
from megatron.initialize import initialize_megatron
from megatron.model import REALMRetriever
from megatron.training import get_model
from pretrain_bert_ict import get_batch, model_provider


def detach(tensor):
    return tensor.detach().cpu().numpy()


class HashedIndex(object):
    """Class for holding hashed data"""
    def __init__(self, embed_size, num_buckets, whiten=False, seed=0):
        np.random.seed(seed)
        self.block_data = defaultdict(list)
        self.hash_data = defaultdict(list)
        hash_matrix = np.random.rand(embed_size, int(num_buckets / 2))
        self.hash_matrix = hash_matrix / np.linalg.norm(hash_matrix, axis=0).reshape(1, -1)
        self.embed_mean = None
        self.embed_whitener = None
        self.whiten = whiten
        self.m = 5

    def state(self):
        state = {
            'block_data': self.block_data,
            'hash_data': self.hash_data,
            'hash_matrix': self.hash_matrix,
            'embed_mean': self.embed_mean,
            'embed_whitener': self.embed_whitener,
        }
        return state

    def get_block_bucket(self, hash):
        return self.hash_data[hash]

    def get_block_embed(self, block_idx):
        return self.block_data[block_idx]

    def hash_embeds(self, embeds, block_data=None):
        """Hash a tensor of embeddings using a random projection matrix"""
        embed_scores_pos = torch.matmul(embeds, torch.cuda.FloatTensor(self.hash_matrix))
        embed_scores = torch.cat((embed_scores_pos, -embed_scores_pos), axis=1)
        embed_hashes = detach(torch.argmax(embed_scores, axis=1))

        if block_data is not None:
            for hash, indices in zip(embed_hashes, block_data):
                self.hash_data[hash].append(indices)

        return embed_hashes

    def assign_block_embeds(self, block_indices, block_embeds, allow_overwrite=False):
        """Assign the embeddings for each block index into a hash map"""
        for idx, embed in zip(block_indices, block_embeds):
            if not allow_overwrite and int(idx) in self.block_data:
                raise ValueError("Attempted to overwrite a read-only HashedIndex")
            self.block_data[int(idx)] = np.float16(embed)

    def save_shard(self, rank):
        dir_name = 'block_hash_data'
        if not os.path.isdir(dir_name):
            os.mkdir(dir_name)

        # save the data for each shard
        with open('{}/{}.pkl'.format(dir_name, rank), 'wb') as data_file:
            pickle.dump(self.state(), data_file)

    def consolidate_shards_and_save(self, ignore_shard=0):
        """Combine all the shards made using self.save_shard()"""
        dir_name = 'block_hash_data'
        fnames = os.listdir(dir_name)
        for fname in fnames:
            with open('{}/{}'.format(dir_name, fname), 'rb') as f:
                data = pickle.load(f)
                assert np.array_equal(data['hash_matrix'], self.hash_matrix)

                old_size = len(self.block_data)
                shard_size = len(data['block_data'])
                self.block_data.update(data['block_data'])
                assert (len(self.block_data) == old_size + shard_size) or (str(ignore_shard) in fname)

                if not self.whiten:
                    for bucket, items in data['hash_data'].items():
                        self.hash_data[bucket].extend(items)

        if self.whiten:
            self.whiten_block_embeds()

        args = get_args()
        with open(args.hash_data_path, 'wb') as final_file:
            pickle.dump(self.state(), final_file)
        shutil.rmtree(dir_name, ignore_errors=True)

    def clear(self):
        """Clear the data structures to save memory"""
        self.block_data = dict()
        self.hash_data = defaultdict(list)

    def whiten_block_embeds(self):
        """Transform all block embeds to have zero mean and unit covariance
        when treated as samples from a distribution"""
        block_idx, all_embeds = zip(*self.block_data.items())
        arr_embeds = np.transpose(np.array(all_embeds))

        mean = np.mean(arr_embeds, axis=1).reshape(-1, 1)
        centered = arr_embeds - mean
        inv_cov = np.linalg.inv(np.cov(arr_embeds))
        whitener = np.transpose(np.linalg.cholesky(inv_cov))
        whitened = np.float16(np.transpose(whitener.dot(centered)))

        self.embed_mean = mean.reshape(-1)
        self.embed_whitener = whitener
        self.block_data = dict(zip(block_idx, list(whitened)))
        self.hash_data = defaultdict(list)
        batch_size = 16384
        i = 0

        with torch.no_grad():
            hashing_tensor = torch.cuda.HalfTensor(self.hash_matrix)
            while True:
                batch_slice = slice(i * batch_size, (i + 1) * batch_size)
                batch_embed = torch.cuda.HalfTensor(whitened[batch_slice])
                batch_block_idx = block_idx[batch_slice]
                if batch_embed.size == 0:
                    break

                hash_scores_pos = torch.matmul(batch_embed, hashing_tensor)
                embed_scores = torch.cat((hash_scores_pos, -hash_scores_pos), axis=1)
                embed_hashes = detach(torch.argmax(embed_scores, axis=1))
                for hash, embed in zip(list(embed_hashes), list(detach(batch_embed))):
                    # [int] instead of [array<int>] since this is just for analysis rn
                    self.hash_data[hash].append(batch_block_idx)

    def create_block_data_index(self):
        import faiss
        self.block_idx, block_embeds = zip(*self.block_data.items())
        block_embeds = np.array(block_embeds)

        index = faiss.IndexFlatL2(block_embeds.shape[1])
        index.add(block_embeds)
        print('Total blocks in index: ', index.ntotal)
        self.block_index = index

    def exact_mips_equals(self, query_embeds):
        """For each query, determine whether the mips block is in the correct hash bucket"""
        _, block_embeds = zip(*self.block_data.items())
        with torch.no_grad():
            # get hashes for the queries
            hash_scores_pos = torch.matmul(torch.cuda.HalfTensor(query_embeds), torch.cuda.HalfTensor(self.hash_matrix))
            hash_scores = torch.cat((hash_scores_pos, -hash_scores_pos), axis=1)
            query_hashes = detach(torch.argmax(hash_scores, axis=1))

            # [num_query x num_blocks]
            inner_products = torch.matmul(torch.cuda.HalfTensor(query_embeds),
                                          torch.cuda.HalfTensor(np.transpose(np.array(block_embeds))))
            max_inner_product_idxes = detach(torch.argmax(inner_products, axis=1))
            best_blocks = [self.block_data[idx] for idx in max_inner_product_idxes]
            best_blocks_tensor = torch.cuda.HalfTensor(np.array(best_blocks))
            # bb = best_blocks
            bb_hash_scores_pos = torch.matmul(torch.cuda.HalfTensor(best_blocks_tensor), torch.cuda.HalfTensor(self.hash_matrix))
            bb_hash_scores = torch.cat((bb_hash_scores_pos, -bb_hash_scores_pos), axis=1)
            best_block_hashes = detach(torch.argmax(bb_hash_scores, axis=1))
            equal_arr = np.equal(query_hashes, best_block_hashes).astype(int)

            # array of zeros and ones which can be used for counting success
            return equal_arr

    def exact_mips_test(self, whitened):
        if whitened:
            if self.embed_mean is None:
                self.whiten_block_embeds()
            query_embeds = np.random.multivariate_normal(np.zeros(128), np.eye(128), 256)
        else:
            block_idx, all_embeds = zip(*self.block_data.items())
            arr_embeds = np.transpose(np.array(all_embeds))

            mean = np.mean(arr_embeds, axis=1).reshape(-1, 1)
            cov = np.cov(arr_embeds)
            query_embeds = np.random.multivariate_normal(mean, cov, 256)

        equal_arr = self.exact_mips_equals(query_embeds)
        print("Num correct: ", sum(equal_arr), " Fraction correct: ", sum(equal_arr) / equal_arr.size)

    @classmethod
    def load_from_file(cls, fname):
        print(" > Unpickling block hash data")
        state_dict = pickle.load(open(fname, 'rb'))
        print(" > Finished unpickling")
        hash_matrix = state_dict['hash_matrix']

        new_index = HashedIndex(hash_matrix.shape[0], hash_matrix.shape[1] * 2)
        new_index.block_data = state_dict['block_data']
        new_index.hash_data = state_dict['hash_data']
        new_index.hash_matrix = hash_matrix

        return new_index


def test_retriever():
    initialize_megatron(extra_args_provider=None,
                        args_defaults={'tokenizer_type': 'BertWordPieceLowerCase'})
    args = get_args()
    model = load_ict_checkpoint(only_block_model=True)
    model.eval()
    dataset = get_ict_dataset()
    hashed_index = HashedIndex.load_from_file(args.hash_data_path)
    retriever = REALMRetriever(model, dataset, hashed_index)

    strs = [
        "The last monarch from the house of windsor",
        "married to Elvis Presley",
        "tallest building in the world today",
        "who makes graphics cards"
    ]

    for s in strs:
        retriever.retrieve_evidence_blocks_text(s)


def main():

    # TODO
    # consider broadcasting/all-reducing all in memory rather than using the filesystem
    # create a different process group in the same nccl world - don't have to use chkpts on disc or transfer things on disc
    # torch distributed new group, constains a list of rank, gives back a group which I can hand to the collective operations
    # create a training process group, indexing process group
    # pass the training group to the distributed DDP, instead of the large world process group
    # use indexing process group for the shard-combining
    # communication group between process "8" and process "0" which tells training group that there's a new index
    # also, process 0 sends process 8 the new model

    # if i want to launch a separate process for indexing, may have to work with environment variables to
    # allocate the resources well. Have to subsequently assign the correct gpus to the indexing job
    # consider initializing everything in a single group and break off processes based on the ranks

    # for debugging purposes, make it so that the training process group checks every some number of intervals
    # and if it isn't ready, then wait so that it's consistent. Start with using the filesystem

    initialize_megatron(extra_args_provider=None,
                        args_defaults={'tokenizer_type': 'BertWordPieceLowerCase'})
    args = get_args()
    model = load_ict_checkpoint(only_block_model=True, no_grad=True)
    model.eval()
    dataset = get_ict_dataset()
    data_iter = iter(get_one_epoch_dataloader(dataset))
    hashed_index = HashedIndex(embed_size=128, num_buckets=4096, whiten=True)

    i = 1
    total = 0
    whiten = False
    while True:
        try:
            query_tokens, query_pad_mask, \
            block_tokens, block_pad_mask, block_indices = get_batch(data_iter)
        except:
            break

        block_indices = detach(block_indices)
        block_logits = model(None, None, block_tokens, block_pad_mask, only_block=True)

        # If whitened, then hashing needs to be done after whitening the block embeds
        # which is done in consolidate_shards_and_save()
        if not whiten:
            hashed_index.hash_embeds(block_logits, block_indices)
        hashed_index.assign_block_embeds(block_indices[:, 3], detach(block_logits))

        total += block_indices.size
        i += 1
        if i % 20 == 0:
            print('Batch {:10d} | Total {:10d}'.format(i, total), flush=True)
            if args.debug:
                break

    hashed_index.save_shard(args.rank)
    torch.distributed.barrier()
    del model

    if args.rank == 0:
        hashed_index.consolidate_shards_and_save()
    else:
        hashed_index.clear()


def load_ict_checkpoint(only_query_model=False, only_block_model=False, no_grad=False):
    args = get_args()
    model = get_model(lambda: model_provider(only_query_model, only_block_model))

    if isinstance(model, torchDDP):
        model = model.module
    tracker_filename = get_checkpoint_tracker_filename(args.ict_load)
    with open(tracker_filename, 'r') as f:
        iteration = int(f.read().strip())

    assert iteration > 0
    checkpoint_name = get_checkpoint_name(args.ict_load, iteration, False)
    if mpu.get_data_parallel_rank() == 0:
        print('global rank {} is loading checkpoint {}'.format(
            torch.distributed.get_rank(), checkpoint_name))

    state_dict = torch.load(checkpoint_name, map_location='cpu')
    if only_query_model:
        state_dict['model'].pop('context_model')
    if only_block_model:
        state_dict['model'].pop('question_model')
    if no_grad:
        with torch.no_grad():
            model.load_state_dict(state_dict['model'])
    else:
        model.load_state_dict(state_dict['model'])
    torch.distributed.barrier()

    if mpu.get_data_parallel_rank() == 0:
        print(' successfully loaded {}'.format(checkpoint_name))

    return model


def get_ict_dataset():
    args = get_args()
    block_dataset = get_indexed_dataset_(args.data_path, 'mmap', True)
    titles_dataset = get_indexed_dataset_(args.titles_data_path, 'mmap', True)

    kwargs = dict(
        name='full',
        block_dataset=block_dataset,
        title_dataset=titles_dataset,
        data_prefix=args.data_path,
        num_epochs=1,
        max_num_samples=None,
        max_seq_length=288,  # doesn't matter
        short_seq_prob=0.0001,  # doesn't matter
        seed=1
    )
    dataset = InverseClozeDataset(**kwargs)
    return dataset


def get_one_epoch_dataloader(dataset):
    args = get_args()

    world_size = mpu.get_data_parallel_world_size()
    rank = mpu.get_data_parallel_rank()
    global_batch_size = args.batch_size * world_size
    num_workers = args.num_workers

    sampler = torch.utils.data.SequentialSampler(dataset)
    batch_sampler = DistributedBatchSampler(sampler,
                                            batch_size=global_batch_size,
                                            drop_last=True,
                                            rank=rank,
                                            world_size=world_size)

    return torch.utils.data.DataLoader(dataset,
                                       batch_sampler=batch_sampler,
                                       num_workers=num_workers,
                                       pin_memory=True)


if __name__ == "__main__":
    main()
