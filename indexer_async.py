import os
import time

import torch
import torch.distributed as dist

from megatron import get_args
from megatron.global_vars import set_global_variables
from megatron.initialize import init_distributed, _init_autoresume, _set_random_seed, _write_args_to_tensorboard
from megatron.mpu.initialize import set_data_parallel_group, set_model_parallel_group

# Example: 4x8 for training, 1x8 for indexing.
# Assign args.rank < 32 to TRAIN_PROCESS_GROUP, args.rank >= to INDEX_PROCESS_GROUP
# can manually assign _MODEL_PARALLEL_GROUP to args.rank, _DATA_PARALLEL_GROUP to train or index process group
# for both, create a torchDDP accordingly because you need to set up the model to be data-parallel on each.

INDEX_READY = None
TRAIN_GROUP = None
INDEX_GROUP = None


# flow:
# index builder finishes first and sets INDEX_READY = 1.
# communicates by dist.broadcast(INDEX_READY, src=min_index_rank)
# index builder is now waiting for INDEX_READY = 0.
#
# at every iteration, trainer checks INDEX_READY = 1.
# when INDEX_READY = 1, reload the index, save model checkpoint and set INDEX_READY = 0.
# once done, trainer does dist.broadcast(INDEX_READY, src=min_train_rank)
# when INDEX_READY = 0, indexer loads up model checkpoint and begins again.

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
    setup_groups()
    pprint('finished setting up groups')

    # Autoresume
    _init_autoresume()
    pprint('finished setting up autoresume')

    # Random seeds for reproducibility.
    args = get_args()
    if args.rank == 0:
        pprint('> setting random seeds to {} ...'.format(args.seed))
    # _set_random_seed(args.seed)

    # Write arguments to tensorboard.
    _write_args_to_tensorboard()
    pprint('finished writing args to tensorboard')

    torch.distributed.barrier()
    global INDEX_READY
    INDEX_READY = torch.zeros(1).cuda()

    if args.rank < args.max_training_rank:
        runner = AsyncREALMTrainer(args.rank)
        torch.distributed.barrier(TRAIN_GROUP)
        pprint("All trainers ready.")
        runner.dummy_train_model()
    else:
        runner = AsyncIndexBuilder(args.rank)
        torch.distributed.barrier(INDEX_GROUP)
        pprint("All indexers ready.")
        runner.dummy_build_index()


def setup_groups():
    args = get_args()
    world_size = dist.get_world_size()
    max_training_rank = args.max_training_rank

    # assuming no model parallelism right now
    set_model_parallel_group(args.rank)

    global TRAIN_GROUP
    global INDEX_GROUP
    # important for batching and whatnot
    TRAIN_GROUP = dist.new_group(list(range(max_training_rank)))
    INDEX_GROUP = dist.new_group(list(range(max_training_rank, world_size)))

    if args.rank > max_training_rank:
        set_data_parallel_group(INDEX_GROUP)
    else:
        set_data_parallel_group(TRAIN_GROUP)


class AsyncIndexBuilder(object):
    def __init__(self, rank):
        self.rank = rank
        pprint("My rank: ", self.rank)

    def dummy_build_index(self):
        start_time = time.time()
        pprint("START: {}".format(time.ctime(start_time)))
        pprint("-" * 100)
        for i in range(5):
            # simulating building the index which takes 20 seconds
            time.sleep(10)
            pprint('built the index. Time: {}'.format(time.ctime(time.time())))
            args = get_args()

            global INDEX_READY
            if self.rank == args.max_training_rank:
                # broadcasting that the index is ready
                INDEX_READY = 1 - INDEX_READY
                send_handle = dist.broadcast(INDEX_READY, args.max_training_rank, async_op=True)
                pprint("Broadcasted index ready = ", INDEX_READY)
            else:
                send_recv_handle = dist.broadcast(INDEX_READY, args.max_training_rank, async_op=True)

            torch.distributed.barrier(INDEX_GROUP)
            pprint("Synced after broadcasting")

            recv_handle = dist.broadcast(INDEX_READY, 0, async_op=True)
            while INDEX_READY == 1:
                pprint('waiting for new model. Time: {}'.format(time.ctime(time.time())))
                time.sleep(1)


class AsyncREALMTrainer(object):
    def __init__(self, rank):
        self.rank = rank
        pprint("My rank: ", self.rank)

    def dummy_train_model(self):
        start_time = time.time()
        pprint("START: {}".format(time.ctime(start_time)))
        pprint("-" * 100)
        args = get_args()
        for i in range(5):
            global INDEX_READY
            recv_handle = dist.broadcast(INDEX_READY, args.max_training_rank, async_op=True)
            while True:
                if INDEX_READY == 1:
                    break

                assert self.rank != args.max_training_rank
                pprint('waiting for new index. Time: {}'.format(time.ctime(time.time())))
                time.sleep(2)

            # INDEX_READY is 1
            if self.rank == 0:
                INDEX_READY = 1 - INDEX_READY
                send_handle = dist.broadcast(INDEX_READY, 0, async_op=True)
                pprint("Broadcasted index ready = ", INDEX_READY)
            else:
                send_recv_handle = dist.broadcast(INDEX_READY, 0, async_op=True)

            torch.distributed.barrier(TRAIN_GROUP)
            pprint("Synced after broadcasting")


if __name__ == "__main__":
    initialize_and_run_async_megatron(args_defaults={'tokenizer_type': 'BertWordPieceLowerCase'})
