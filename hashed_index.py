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
from megatron.training import get_model
from pretrain_bert_ict import get_batch, model_provider


def detach(tensor):
    return tensor.detach().cpu().numpy()


class HashedIndex(object):
    """Class for holding hashed data"""
    def __init__(self, embed_size, num_buckets, seed=0):
        np.random.seed(seed)
        self.block_data = defaultdict(list)
        self.hash_data = defaultdict(list)
        self.hash_matrix = np.random.rand(embed_size, num_buckets / 2)

    def state(self):
        state = {
            'block_data': self.block_data,
            'hash_data': self.hash_data,
            'hash_matrix': self.hash_matrix
        }
        return state

    def get_block_bucket(self, hash):
        return self.hash_data[hash]

    def get_block_embed(self, block_idx):
        return self.block_data[block_idx]

    def hash_embeds(self, embeds, block_data=None):
        """Hash a tensor of embeddings using a random projection matrix"""
        embed_scores_pos = torch.matmul(embeds, torch.cuda.HalfTensor(self.hash_matrix))
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
            self.block_data[int(idx)] = embed

    def save_shard(self, rank):
        dir_name = 'block_hash_data'
        if not os.path.isdir(dir_name):
            os.mkdir(dir_name)

        # save the data for each shard
        with open('{}/{}.pkl'.format(dir_name, rank), 'wb') as data_file:
            pickle.dump(self.state(), data_file)

    def consolidate_shards_and_save(self):
        """Combine all the shards made using self.save_shard()"""
        dir_name = 'block_hash_data'
        fnames = os.listdir(dir_name)
        for fname in fnames:
            with open('{}/{}'.format(dir_name, fname), 'rb') as f:
                data = pickle.load(f)
                assert data['hash_matrix'] == self.hash_matrix

                old_size = len(self.block_data)
                shard_size = len(data['block_data'])
                self.block_data.update(data['block_data'])
                assert len(self.block_data) == old_size + shard_size

                for bucket, items in data['hash_data'].items():
                    self.hash_data[bucket].extend(items)

        with open('block_hash_data.pkl', 'wb') as final_file:
            pickle.dump(self.state(), final_file)
        shutil.rmtree(dir_name, ignore_errors=True)

    def clear(self):
        """Clear the data structures to save memory"""
        self.block_data = defaultdict(list)
        self.hash_data = defaultdict(list)


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

    initialize_megatron(extra_args_provider=None,
                        args_defaults={'tokenizer_type': 'BertWordPieceLowerCase'})
    args = get_args()
    model = load_checkpoint()
    model.eval()
    dataset = get_dataset()
    data_iter = iter(get_dataloader(dataset))
    hashed_index = HashedIndex(embed_size=128, num_buckets=2048)

    i = 0
    while True:
        try:
            query_tokens, query_pad_mask, \
            block_tokens, block_pad_mask, block_indices = get_batch(data_iter)
        except:
            break

        actual_model = model.module.module
        block_indices = detach(block_indices)

        block_logits = actual_model.embed_block(block_tokens, block_pad_mask)
        hashed_index.hash_embeds(block_logits, block_indices)
        hashed_index.assign_block_embeds(block_indices, detach(block_logits))

        if i % 100 == 0:
            print(i, flush=True)
        i += 1

    hashed_index.save_shard(args.rank)
    torch.distributed.barrier()
    del model

    if mpu.get_data_parallel_rank() == 0:
        hashed_index.consolidate_shards_and_save()
    else:
        hashed_index.clear()


def load_checkpoint():
    args = get_args()
    model = get_model(model_provider)

    if isinstance(model, torchDDP):
        model = model.module
    tracker_filename = get_checkpoint_tracker_filename(args.load)
    with open(tracker_filename, 'r') as f:
        iteration = int(f.read().strip())

    assert iteration > 0
    checkpoint_name = get_checkpoint_name(args.load, iteration, False)
    if mpu.get_data_parallel_rank() == 0:
        print('global rank {} is loading checkpoint {}'.format(
            torch.distributed.get_rank(), checkpoint_name))

    state_dict = torch.load(checkpoint_name, map_location='cpu')
    model.load_state_dict(state_dict['model'])
    torch.distributed.barrier()

    if mpu.get_data_parallel_rank() == 0:
        print(' successfully loaded {}'.format(checkpoint_name))

    return model


def get_dataset():
    args = get_args()
    block_dataset = get_indexed_dataset_(args.data_path, 'mmap', True)
    titles_dataset = get_indexed_dataset_(args.data_path + '-titles', 'mmap', True)

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


def get_dataloader(dataset):
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
