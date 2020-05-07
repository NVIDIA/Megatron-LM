import os
import time

import torch
from torch.nn.parallel.distributed import DistributedDataParallel as torchDDP

from megatron import get_args
from megatron import mpu
from megatron.checkpointing import get_checkpoint_tracker_filename, get_checkpoint_name
from megatron.data.bert_dataset import get_indexed_dataset_
from megatron.data.realm_dataset import ICTDataset
from megatron.data.realm_index import detach, BlockData, RandProjectionLSHIndex
from megatron.data.samplers import DistributedBatchSampler
from megatron.initialize import initialize_megatron
from megatron.model import REALMRetriever
from megatron.training import get_model
from pretrain_bert_ict import get_batch, model_provider


def test_retriever():
    # TODO: Update this because it's outdated and definitely won't run.
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
    ran_once = False
    while True:
        model = load_ict_checkpoint(only_block_model=True, no_grad=True, from_realm_chkpt=ran_once)
        model.eval()
        dataset = get_ict_dataset()
        data_iter = iter(get_one_epoch_dataloader(dataset))
        all_block_data = BlockData()
        hashed_index = RandProjectionLSHIndex(embed_size=128, num_buckets=32, whiten=True)

        i = 1
        total = 0
        while True:
            with torch.no_grad():
                try:
                    query_tokens, query_pad_mask, \
                    block_tokens, block_pad_mask, block_index_data = get_batch(data_iter)
                except:
                    break

                block_index_data = detach(block_index_data)
                block_indices = block_index_data[:, 3]
                block_meta = block_index_data[:, :3]

                block_logits = detach(model(None, None, block_tokens, block_pad_mask, only_block=True))
                all_block_data.add_block_data(block_indices, block_logits, block_meta)

                total += block_indices.size
                i += 1
                if i % 20 == 0:
                    print('Batch {:10d} | Total {:10d}'.format(i, total), flush=True)
                    if args.debug:
                        break

        all_block_data.save_shard(args.rank)
        torch.distributed.barrier()
        del model

        if args.rank == 0:
            all_block_data.consolidate_shards_and_save()
            hashed_index.hash_whitened_block_embeds(all_block_data)
            hashed_index.save_to_file()
        else:
            all_block_data.clear()

        ran_once = True
        set_index_com_file_ready()
        torch.distributed.barrier()
        while not check_model_com_file_ready():
            time.sleep(5)

        set_model_com_file_not_ready()


INDEX_COM_FILE = 'ready.index'
MODEL_COM_FILE = 'ready.model'


def setup_index_com_file():
    set_index_com_file_not_ready()


def set_index_com_file_not_ready():
    with open(INDEX_COM_FILE, 'w') as com_file:
        com_file.write('0')


def set_index_com_file_ready():
    with open(INDEX_COM_FILE, 'w') as com_file:
        com_file.write('1')


def check_index_com_file_ready():
    if os.path.exists(INDEX_COM_FILE):
        with open(INDEX_COM_FILE, 'r') as com_file:
            return bool(com_file.readline())

    return False


def setup_model_com_file():
    set_model_com_file_not_ready()


def set_model_com_file_not_ready():
    with open(MODEL_COM_FILE, 'w') as com_file:
        com_file.write('0')


def set_model_com_file_ready():
    with open(MODEL_COM_FILE, 'w') as com_file:
        com_file.write('1')


def check_model_com_file_ready():
    if os.path.exists(MODEL_COM_FILE):
        with open(MODEL_COM_FILE, 'r') as com_file:
            return bool(com_file.readline())

    return False


def load_ict_checkpoint(only_query_model=False, only_block_model=False, no_grad=False, from_realm_chkpt=False):
    args = get_args()
    model = get_model(lambda: model_provider(only_query_model, only_block_model))

    load_path = args.load if from_realm_chkpt else args.ict_load

    if isinstance(model, torchDDP):
        model = model.module
    tracker_filename = get_checkpoint_tracker_filename(load_path)
    with open(tracker_filename, 'r') as f:
        iteration = int(f.read().strip())

    assert iteration > 0
    checkpoint_name = get_checkpoint_name(load_path, iteration, False)
    if mpu.get_data_parallel_rank() == 0:
        print('global rank {} is loading checkpoint {}'.format(
            torch.distributed.get_rank(), checkpoint_name))

    state_dict = torch.load(checkpoint_name, map_location='cpu')
    ict_state_dict = state_dict['model']
    if from_realm_chkpt:
        ict_state_dict = ict_state_dict['retriever']['ict_model']

    if only_query_model:
        ict_state_dict.pop('context_model')
    if only_block_model:
        ict_state_dict.pop('question_model')
    if no_grad:
        with torch.no_grad():
            model.load_state_dict(ict_state_dict)
    else:
        model.load_state_dict(ict_state_dict)
    torch.distributed.barrier()

    if mpu.get_data_parallel_rank() == 0:
        print(' successfully loaded {}'.format(checkpoint_name))

    return model


def get_ict_dataset(use_titles=True):
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
        seed=1,
        use_titles=use_titles
    )
    dataset = ICTDataset(**kwargs)
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
