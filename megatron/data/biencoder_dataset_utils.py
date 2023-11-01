import os
import time

import numpy as np
import torch

from megatron import get_args, get_tokenizer, print_rank_0
from megatron.core import mpu, tensor_parallel
from megatron.data.dataset_utils import create_masked_lm_predictions, \
                                            pad_and_convert_to_numpy
from megatron.data.data_samplers import MegatronPretrainingSampler

def make_attention_mask(source_block, target_block):
    """
    Returns a 2-dimensional (2-D) attention mask
    :param source_block: 1-D array
    :param target_block: 1-D array
    """
    mask = (target_block[None, :] >= 1) * (source_block[:, None] >= 1)
    mask = mask.astype(np.int64)
    # (source_length, target_length)
    return mask

def get_one_epoch_dataloader(dataset, micro_batch_size=None):
    """Specifically one epoch to be used in an indexing job."""
    args = get_args()

    if micro_batch_size is None:
        micro_batch_size = args.micro_batch_size
    num_workers = args.num_workers

    # Use megatron's sampler with consumed samples set to 0 as
    # this is only for evaluation and don't intend to resume half way.
    # Also, set the drop last to false as don't intend to remove
    # the last batch
    batch_sampler = MegatronPretrainingSampler(
        total_samples=len(dataset),
        consumed_samples=0,
        micro_batch_size=args.micro_batch_size,
        data_parallel_rank=mpu.get_data_parallel_rank(),
        data_parallel_size=mpu.get_data_parallel_world_size(),
        drop_last=False)

    return torch.utils.data.DataLoader(dataset,
                                       batch_sampler=batch_sampler,
                                       num_workers=num_workers,
                                       pin_memory=True)


def get_ict_batch(data_iterator):
    # Items and their type.
    keys = ['query_tokens', 'query_mask',
            'context_tokens', 'context_mask', 'block_data']
    datatype = torch.int64

    # Broadcast data.
    if data_iterator is None:
        data = None
    else:
        data = next(data_iterator)
    data_b = tensor_parallel.broadcast_data(keys, data, datatype)

    # Unpack.
    query_tokens = data_b['query_tokens'].long()
    query_mask = data_b['query_mask'] < 0.5
    context_tokens = data_b['context_tokens'].long()
    context_mask = data_b['context_mask'] < 0.5
    block_indices = data_b['block_data'].long()

    return query_tokens, query_mask,\
           context_tokens, context_mask, block_indices


def join_str_list(str_list):
    """Join a list of strings, handling spaces appropriately"""
    result = ""
    for s in str_list:
        if s.startswith("##"):
            result += s[2:]
        else:
            result += " " + s
    return result


class BlockSampleData(object):
    """A struct for fully describing a fixed-size block of data as used in REALM

    :param start_idx: for first sentence of the block
    :param end_idx: for last sentence of the block (may be partially truncated in sample construction)
    :param doc_idx: the index of the document from which the block comes in the original indexed dataset
    :param block_idx: a unique integer identifier given to every block.
    """
    def __init__(self, start_idx, end_idx, doc_idx, block_idx):
        self.start_idx = start_idx
        self.end_idx = end_idx
        self.doc_idx = doc_idx
        self.block_idx = block_idx

    def as_array(self):
        return np.array([self.start_idx, self.end_idx, self.doc_idx, self.block_idx]).astype(np.int64)

    def as_tuple(self):
        return self.start_idx, self.end_idx, self.doc_idx, self.block_idx


class BlockSamplesMapping(object):
    def __init__(self, mapping_array):
        # make sure that the array is compatible with BlockSampleData
        assert mapping_array.shape[1] == 4
        self.mapping_array = mapping_array

    def __len__(self):
        return self.mapping_array.shape[0]

    def __getitem__(self, idx):
        """Get the data associated with an indexed sample."""
        sample_data = BlockSampleData(*self.mapping_array[idx])
        return sample_data


def get_block_samples_mapping(block_dataset, title_dataset, data_prefix, num_epochs,
                              max_num_samples, max_seq_length, seed, name, use_one_sent_docs=False):
    """Get samples mapping for a dataset over fixed size blocks. This function also requires
    a dataset of the titles for the source documents since their lengths must be taken into account.

    :return: samples_mapping (BlockSamplesMapping)
    """

    if not num_epochs:
        if not max_num_samples:
            raise ValueError("Need to specify either max_num_samples "
                             "or num_epochs")
        num_epochs = np.iinfo(np.int32).max - 1
    if not max_num_samples:
        max_num_samples = np.iinfo(np.int64).max - 1

    # Filename of the index mapping
    indexmap_filename = data_prefix
    indexmap_filename += '_{}_indexmap'.format(name)
    if num_epochs != (np.iinfo(np.int32).max - 1):
        indexmap_filename += '_{}ep'.format(num_epochs)
    if max_num_samples != (np.iinfo(np.int64).max - 1):
        indexmap_filename += '_{}mns'.format(max_num_samples)
    indexmap_filename += '_{}msl'.format(max_seq_length)
    indexmap_filename += '_{}s'.format(seed)
    if use_one_sent_docs:
        indexmap_filename += '_1sentok'
    indexmap_filename += '.npy'

    # Build the indexed mapping if not exist.
    if mpu.get_data_parallel_rank() == 0 and \
            not os.path.isfile(indexmap_filename):
        print(' > WARNING: could not find index map file {}, building '
              'the indices on rank 0 ...'.format(indexmap_filename))

        # Make sure the types match the helpers input types.
        assert block_dataset.document_indices.dtype == np.int64
        assert block_dataset.sequence_lengths.dtype == np.int32

        # Build samples mapping
        verbose = torch.distributed.get_rank() == 0
        start_time = time.time()
        print_rank_0(' > building samples index mapping for {} ...'.format(
            name))

        from megatron.core.datasets import helpers
        mapping_array = helpers.build_blocks_mapping(
            block_dataset.document_indices,
            block_dataset.sequence_lengths,
            title_dataset.sequence_lengths,
            num_epochs,
            max_num_samples,
            max_seq_length - 3,  # account for added tokens
            seed,
            verbose,
            use_one_sent_docs)


        print_rank_0(' > done building samples index mapping')
        np.save(indexmap_filename, mapping_array, allow_pickle=True)
        print_rank_0(' > saved the index mapping in {}'.format(
            indexmap_filename))
        # Make sure all the ranks have built the mapping
        print_rank_0(' > elapsed time to build and save samples mapping '
                     '(seconds): {:4f}'.format(
            time.time() - start_time))

    # This should be a barrier but nccl barrier assumes
    # device_index=rank which is not the case for model
    # parallel case
    counts = torch.cuda.LongTensor([1])
    torch.distributed.all_reduce(counts, group=mpu.get_data_parallel_group())
    assert counts[0].item() == torch.distributed.get_world_size(
        group=mpu.get_data_parallel_group())

    # Load indexed dataset.
    print_rank_0(' > loading indexed mapping from {}'.format(
        indexmap_filename))
    start_time = time.time()

    mapping_array = np.load(indexmap_filename, allow_pickle=True, mmap_mode='r')
    samples_mapping = BlockSamplesMapping(mapping_array)

    print_rank_0('    loaded indexed file in {:3.3f} seconds'.format(
        time.time() - start_time))
    print_rank_0('    total number of samples: {}'.format(
        mapping_array.shape[0]))

    return samples_mapping
