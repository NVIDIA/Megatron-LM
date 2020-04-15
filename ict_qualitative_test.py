from collections import defaultdict
import pickle

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


def main():
    initialize_megatron(extra_args_provider=None,
                        args_defaults={'tokenizer_type': 'BertWordPieceLowerCase'})
    args = get_args()
    model = load_checkpoint()
    model.eval()
    dataset = get_dataset()
    data_iter = iter(get_dataloader(dataset))

    hash_data = defaultdict(list)
    hash_matrix = torch.cuda.HalfTensor(np.random.rand(128, 1024))

    all_input_tokens = []
    all_input_logits = []
    all_block_tokens = []
    all_block_logits = []

    i = 0
    while True:
        try:
            input_tokens, input_types, input_pad_mask, \
            block_tokens, block_token_types, block_pad_mask, block_indices = get_batch(data_iter)
        except StopIteration:
            break
        input_logits, block_logits, _ = model.module.module.forward(
            input_tokens, input_types, input_pad_mask, block_tokens, block_pad_mask, block_token_types, return_logits=True)

        block_hash_pos = torch.matmul(block_logits, hash_matrix)
        block_hash_full = torch.cat((block_hash_pos, -block_hash_pos), axis=1)
        block_hashes = torch.argmax(block_hash_full, axis=1).detach().cpu().numpy()
        for hash, idx in zip(block_hashes, block_indices):
            hash_data[int(hash)].append(int(idx))

        all_input_tokens.append(input_tokens.detach().cpu().numpy())
        all_input_logits.append(input_logits.detach().cpu().numpy())
        all_block_tokens.append(block_tokens.detach().cpu().numpy())
        all_block_logits.append(block_logits.detach().cpu().numpy())

        if i % 100 == 0:
            print(i, flush=True)
            print(len(all_block_tokens), flush=True)
            print(block_tokens.shape, flush=True)
        i += 1

        if i == 10:
            break

    all_input_tokens = np.array(all_input_tokens).reshape(-1, args.seq_length)
    all_input_logits = np.array(all_input_logits).reshape(-1, 128)
    all_block_tokens = np.array(all_block_tokens).reshape(-1, args.seq_length)
    all_block_logits = np.array(all_block_logits).reshape(-1, 128)
    np.save('input_tokens.npy', all_input_tokens)
    np.save('input_logits.npy', all_input_logits)
    np.save('block_tokens.npy', all_block_tokens)
    np.save('block_logits.npy', all_block_logits)

    for hash, block_indices in hash_data.items():
        hash_data[hash] = np.array(block_indices)

    hash_data['matrix'] = hash_matrix
    with open('hash_data.pkl', 'wb') as hash_file:
        pickle.dump(hash_data, hash_file)


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
        context_dataset=block_dataset,
        titles_dataset=titles_dataset,
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
