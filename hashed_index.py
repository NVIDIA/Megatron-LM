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


def embed_docs():
    initialize_megatron(extra_args_provider=None,
                        args_defaults={'tokenizer_type': 'BertWordPieceLowerCase'})
    args = get_args()
    model = load_checkpoint()
    model.eval()
    dataset = get_dataset()
    data_iter = iter(get_dataloader(dataset))

    hash_data = defaultdict(list)
    hash_matrix = torch.cuda.HalfTensor(np.random.rand(128, 1024))
    hash_data['matrix'] = hash_matrix

    block_data = defaultdict(list)
    i = 0
    while True:
        try:
            input_tokens, input_types, input_pad_mask, \
            block_tokens, block_token_types, block_pad_mask, block_indices = get_batch(data_iter)
        except:
            break

        input_logits, block_logits = model.module.module.forward(
            input_tokens, input_types, input_pad_mask, block_tokens, block_pad_mask, block_token_types)

        block_hash_pos = torch.matmul(block_logits, hash_matrix)
        block_hash_full = torch.cat((block_hash_pos, -block_hash_pos), axis=1)
        block_hashes = detach(torch.argmax(block_hash_full, axis=1))
        for hash, indices_array in zip(block_hashes, block_indices):
            hash_data[int(hash)].append(detach(indices_array))

        block_logits = detach(block_logits)
        # originally this has [start_idx, end_idx, doc_idx, block_idx]
        block_indices = detach(block_indices)[:, 3]
        for logits, idx in zip(block_logits, block_indices):
            block_data[int(idx)] = logits

        if i % 100 == 0:
            print(i, flush=True)
        i += 1

    dir_name = 'block_hash_data'
    if not os.path.isdir(dir_name):
        os.mkdir(dir_name)

    # save the data for each shard
    with open('{}/{}.pkl'.format(dir_name, args.rank), 'wb') as data_file:
        all_data = {'block_data': block_data, 'hash_data': hash_data}
        pickle.dump(all_data, data_file)

    torch.distributed.barrier()

    all_data.clear()
    del all_data
    del model

    # rank 0 process consolidates shards and saves into final file
    if mpu.get_data_parallel_rank() == 0:
        all_block_data = defaultdict(dict)
        dir_name = 'block_hash_data'
        fnames = os.listdir(dir_name)
        for fname in fnames:
            with open('{}/{}'.format(dir_name, fname), 'rb') as f:
                data = pickle.load(f)
                all_block_data['hash_data'].update(data['hash_data'])
                all_block_data['block_data'].update(data['block_data'])

        with open('block_hash_data.pkl', 'wb') as final_file:
            pickle.dump(all_block_data, final_file)
        shutil.rmtree(dir_name, ignore_errors=True)


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
    embed_docs()
