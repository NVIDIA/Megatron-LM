import os
import sys
import time

import torch
import torch.distributed as dist
from torch.nn.parallel.distributed import DistributedDataParallel as torchDDP

from megatron import get_args, get_adlr_autoresume, print_rank_0
from megatron import mpu
from megatron.checkpointing import get_checkpoint_tracker_filename, get_checkpoint_name
from megatron.data.bert_dataset import get_indexed_dataset_
from megatron.data.realm_dataset import ICTDataset
from megatron.data.realm_index import detach, BlockData, FaissMIPSIndex
from megatron.data.samplers import DistributedBatchSampler
from megatron.initialize import initialize_megatron
from megatron.model import REALMRetriever
from megatron.global_vars import set_global_variables
from megatron.mpu.initialize import get_index_ready, get_index_group, get_train_group, get_data_parallel_group, get_gloo_comm_group
from megatron.mpu.initialize import set_data_parallel_group, set_model_parallel_group, init_realm_groups
from megatron.initialize import init_distributed, _init_autoresume, _set_random_seed, _write_args_to_tensorboard
from megatron.training import get_model
from megatron.utils import check_adlr_autoresume_termination
from pretrain_bert_ict import get_batch, model_provider


INDEX_READY = None


def pprint(*args):
    print(*args, flush=True)


def initialize_and_run_async_megatron(extra_args_provider=None, args_defaults={},
                                      ignore_unknown_args=False, allow_no_cuda=False):
    if not allow_no_cuda:
        # Make sure cuda is available.
        assert torch.cuda.is_available(), 'Megatron requires CUDA.'

    # Parse args, build tokenizer, and set adlr-autoresume,
    # tensorboard-writer, and timers.
    set_global_variables(extra_args_provider=extra_args_provider,
                         args_defaults=args_defaults,
                         ignore_unknown_args=ignore_unknown_args)

    # instead of _initialize_distributed()
    init_distributed()
    setup_realm_groups_and_vars()
    global INDEX_READY
    INDEX_READY = get_index_ready()
    pprint('finished setting up groups')

    # Autoresume
    _init_autoresume()
    pprint('finished setting up autoresume')

    # Random seeds for reproducibility.
    args = get_args()
    if args.rank == 0:
        pprint('> setting random seeds to {} ...'.format(args.seed))
    _set_random_seed(args.seed)

    # Write arguments to tensorboard.
    _write_args_to_tensorboard()
    pprint('finished writing args to tensorboard')

    torch.distributed.barrier()

    if args.rank < args.max_training_rank:
        torch.distributed.barrier(get_data_parallel_group())
        pprint("All trainers ready.")
        return
    else:
        runner = AsyncIndexBuilder(args.rank)
        torch.distributed.barrier(get_data_parallel_group())
        pprint("All indexers ready.")
        runner.run_async()


def setup_realm_groups_and_vars():
    args = get_args()
    world_size = dist.get_world_size()
    max_training_rank = args.max_training_rank

    # assuming no model parallelism right now
    set_model_parallel_group(dist.new_group([args.rank]))
    init_realm_groups(max_training_rank, world_size)

    if args.rank < max_training_rank:
        set_data_parallel_group(get_train_group())
    else:
        set_data_parallel_group(get_index_group())


class IndexBuilder(object):
    def __init__(self):
        args = get_args()
        self.debug = args.debug
        self.rank = args.rank
        self.model = None
        self.dataloader = None
        self.block_data = None
        self.load_attributes()
        self.is_main_builder = args.rank == 0

    def load_attributes(self):
        self.model = load_ict_checkpoint(only_block_model=True, no_grad=True, from_realm_chkpt=False)
        self.model.eval()
        self.dataloader = iter(get_one_epoch_dataloader(get_ict_dataset()))
        self.block_data = BlockData()

    def build_and_save_index(self):
        i = 1
        total = 0
        while True:
            with torch.no_grad():
                try:
                    query_tokens, query_pad_mask, \
                    block_tokens, block_pad_mask, block_index_data = get_batch(self.dataloader)
                except:
                    break

                block_index_data = detach(block_index_data)
                block_indices = block_index_data[:, 3]
                block_meta = block_index_data[:, :3]

                block_logits = detach(self.model(None, None, block_tokens, block_pad_mask, only_block=True))
                self.block_data.add_block_data(block_indices, block_logits, block_meta)

                total += block_indices.size
                i += 1
                if i % 1000 == 0:
                    print('Batch {:10d} | Total {:10d}'.format(i, total), flush=True)
                    if self.debug:
                        break

        self.block_data.save_shard(self.rank)
        torch.distributed.barrier(get_data_parallel_group())
        del self.model

        if self.is_main_builder:
            self.block_data.consolidate_shards_and_save(ignore_shard=self.rank)
        self.block_data.clear()


class AsyncIndexBuilder(IndexBuilder):
    def __init__(self, rank):
        self.rank = rank
        args = get_args()
        self.is_main_builder = self.rank == args.max_training_rank
        self.main_builder_idx = args.max_training_rank
        self.debug = args.debug

        self.model = None
        self.dataloader = None
        self.block_data = None
        self.load_attributes()

        global INDEX_READY
        INDEX_READY = get_index_ready()

    def run_async(self):
        global INDEX_READY
        # synchronize for start
        dist.broadcast(INDEX_READY, 0, group=get_gloo_comm_group())
        while True:
            print("Starting (again!)", flush=True)
            self.build_and_save_index()
            self.send_index_ready_signal()
            while INDEX_READY == 1:
                print("Waiting for new model checkpoint.", flush=True)
                time.sleep(5)

            self.load_attributes()

    def load_attributes(self):
        try:
            self.model = load_ict_checkpoint(only_block_model=True, no_grad=True, from_realm_chkpt=True)
        except:
            print(">>>>> No realm chkpt available", flush=True)
            self.model = load_ict_checkpoint(only_block_model=True, no_grad=True, from_realm_chkpt=False)
        self.model.eval()
        self.dataloader = iter(get_one_epoch_dataloader(get_ict_dataset()))
        self.block_data = BlockData()

    def send_index_ready_signal(self):
        global INDEX_READY
        if self.is_main_builder:
            INDEX_READY = 1 - INDEX_READY
            print("Switched INDEX_READY", flush=True)
        torch.cuda.synchronize()

        # send handle
        dist.broadcast(INDEX_READY, self.main_builder_idx, group=get_gloo_comm_group(), async_op=True)

        # recv handle
        dist.broadcast(INDEX_READY, 0, group=get_gloo_comm_group())
        torch.distributed.barrier(get_data_parallel_group())


def load_ict_checkpoint(only_query_model=False, only_block_model=False, no_grad=False, from_realm_chkpt=False):
    args = get_args()
    model = get_model(lambda: model_provider(only_query_model, only_block_model))

    if isinstance(model, torchDDP):
        model = model.module

    load_path = args.load if from_realm_chkpt else args.ict_load

    tracker_filename = get_checkpoint_tracker_filename(load_path)
    with open(tracker_filename, 'r') as f:
        iteration = int(f.read().strip())

    # assert iteration > 0
    checkpoint_name = get_checkpoint_name(load_path, iteration, False)
    if mpu.get_data_parallel_rank() == 0:
        print('global rank {} is loading checkpoint {}'.format(
            torch.distributed.get_rank(), checkpoint_name))

    state_dict = torch.load(checkpoint_name, map_location='cpu')
    ict_state_dict = state_dict['model']
    if from_realm_chkpt:
        print(">>>> Attempting to get ict state dict from realm", flush=True)
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
    torch.distributed.barrier(get_data_parallel_group())

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
        max_seq_length=args.seq_length,
        short_seq_prob=0.0001,  # doesn't matter
        seed=1,
        query_in_block_prob=1,
        use_titles=use_titles
    )
    dataset = ICTDataset(**kwargs)
    return dataset


def get_one_epoch_dataloader(dataset, batch_size=None):
    args = get_args()

    world_size = mpu.get_data_parallel_world_size()
    rank = mpu.get_data_parallel_rank()
    if batch_size is None:
        batch_size = args.batch_size
    global_batch_size = batch_size * world_size
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
    initialize_megatron(extra_args_provider=None,
                        args_defaults={'tokenizer_type': 'BertWordPieceLowerCase'})
    index_builder = IndexBuilder()
    index_builder.build_and_save_index()

