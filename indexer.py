import torch
import torch.distributed as dist
from torch.nn.parallel.distributed import DistributedDataParallel as torchDDP

from megatron import get_args
from megatron import mpu
from megatron.checkpointing import get_checkpoint_tracker_filename, get_checkpoint_name
from megatron.data.dataset_utils import get_indexed_dataset_
from megatron.data.ict_dataset import ICTDataset
from megatron.data.realm_index import detach, BlockData
from megatron.data.samplers import DistributedBatchSampler
from megatron.initialize import initialize_megatron
from megatron.training import get_model
from pretrain_ict import get_batch, general_ict_model_provider


def pprint(*args):
    print(*args, flush=True)


class IndexBuilder(object):
    """Object for taking one pass over a dataset and creating a BlockData of its embeddings"""
    def __init__(self):
        args = get_args()
        self.model = None
        self.dataloader = None
        self.block_data = None

        # need to know whether we're using a REALM checkpoint (args.load) or ICT checkpoint
        assert not (args.load and args.ict_load)
        self.using_realm_chkpt = args.ict_load is None

        self.load_attributes()
        self.is_main_builder = args.rank == 0
        self.iteration = self.total_processed = 0

    def load_attributes(self):
        """Load the necessary attributes: model, dataloader and empty BlockData"""
        self.model = load_ict_checkpoint(only_block_model=True, from_realm_chkpt=self.using_realm_chkpt)
        self.model.eval()
        self.dataloader = iter(get_one_epoch_dataloader(get_ict_dataset()))
        self.block_data = BlockData(load_from_path=False)

    def track_and_report_progress(self, batch_size):
        """Utility function for tracking progress"""
        self.iteration += 1
        self.total_processed += batch_size
        if self.iteration % 10 == 0:
            print('Batch {:10d} | Total {:10d}'.format(self.iteration, self.total_processed), flush=True)

    def build_and_save_index(self):
        """Goes through one epoch of the dataloader and adds all data to this instance's BlockData.

        The copy of BlockData is saved as a shard, which when run in a distributed setting will be
        consolidated by the rank 0 process and saved as a final pickled BlockData.
        """

        while True:
            try:
                # batch also has query_tokens and query_pad_data
                _, _, block_tokens, block_pad_mask, block_sample_data = get_batch(self.dataloader)
            except:
                break

            # detach, setup and add to BlockData
            unwrapped_model = self.model
            while not hasattr(unwrapped_model, 'embed_block'):
                unwrapped_model = unwrapped_model.module
            block_logits = detach(unwrapped_model.embed_block(block_tokens, block_pad_mask))

            detached_data = detach(block_sample_data)
            block_indices = detached_data[:, 3]
            block_metas = detached_data[:, :3]

            self.block_data.add_block_data(block_indices, block_logits, block_metas)
            self.track_and_report_progress(batch_size=block_tokens.shape[0])

        # This process signals to finalize its shard and then synchronize with the other processes
        self.block_data.save_shard()
        torch.distributed.barrier()
        del self.model

        # rank 0 process builds the final copy
        if self.is_main_builder:
            self.block_data.merge_shards_and_save()
        self.block_data.clear()


def load_ict_checkpoint(only_query_model=False, only_block_model=False, from_realm_chkpt=False):
    """load ICT checkpoints for indexing/retrieving. Arguments specify which parts of the state dict to actually use."""
    args = get_args()
    model = get_model(lambda: general_ict_model_provider(only_query_model, only_block_model))

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

    model.load_state_dict(ict_state_dict)
    torch.distributed.barrier()

    if mpu.get_data_parallel_rank() == 0:
        print(' successfully loaded {}'.format(checkpoint_name))

    return model


def get_ict_dataset(use_titles=True, query_in_block_prob=1):
    """Get a dataset which uses block samples mappings to get ICT/block indexing data"""
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
        seed=1,
        query_in_block_prob=query_in_block_prob,
        use_titles=use_titles,
        use_one_sent_docs=True
    )
    dataset = ICTDataset(**kwargs)
    return dataset


def get_one_epoch_dataloader(dataset, batch_size=None):
    """Specifically one epoch to be used in an indexing job."""
    args = get_args()

    world_size = mpu.get_data_parallel_world_size()
    rank = mpu.get_data_parallel_rank()
    if batch_size is None:
        batch_size = args.batch_size
    global_batch_size = batch_size * world_size
    num_workers = args.num_workers

    sampler = torch.utils.data.SequentialSampler(dataset)
    # importantly, drop_last must be False to get all the data.
    batch_sampler = DistributedBatchSampler(sampler,
                                            batch_size=global_batch_size,
                                            drop_last=False,
                                            rank=rank,
                                            world_size=world_size)

    return torch.utils.data.DataLoader(dataset,
                                       batch_sampler=batch_sampler,
                                       num_workers=num_workers,
                                       pin_memory=True)


if __name__ == "__main__":
    # This usage is for basic (as opposed to realm async) indexing jobs.
    initialize_megatron(extra_args_provider=None,
                        args_defaults={'tokenizer_type': 'BertWordPieceLowerCase'})
    index_builder = IndexBuilder()
    index_builder.build_and_save_index()

