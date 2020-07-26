import torch
import torch.distributed as dist

from megatron import get_args
from megatron import mpu
from megatron.checkpointing import load_ict_checkpoint
from megatron.data.ict_dataset import get_ict_dataset
from megatron.data.realm_dataset_utils import get_one_epoch_dataloader
from megatron.data.realm_index import detach, BlockData
from megatron.data.realm_dataset_utils import get_ict_batch
from megatron.model.realm_model import general_ict_model_provider
from megatron.training import get_model


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

        self.log_interval = args.indexer_log_interval
        self.batch_size = args.indexer_batch_size

        self.load_attributes()
        self.is_main_builder = mpu.get_data_parallel_rank() == 0
        self.num_total_builders = mpu.get_data_parallel_world_size()
        self.iteration = self.total_processed = 0

    def load_attributes(self):
        """Load the necessary attributes: model, dataloader and empty BlockData"""
        model = get_model(lambda: general_ict_model_provider(only_block_model=True))
        self.model = load_ict_checkpoint(model, only_block_model=True, from_realm_chkpt=self.using_realm_chkpt)
        self.model.eval()
        self.dataset = get_ict_dataset()
        self.dataloader = iter(get_one_epoch_dataloader(self.dataset, self.batch_size))
        self.block_data = BlockData(load_from_path=False)

    def track_and_report_progress(self, batch_size):
        """Utility function for tracking progress"""
        self.iteration += 1
        self.total_processed += batch_size * self.num_total_builders
        if self.is_main_builder and self.iteration % self.log_interval == 0:
            print('Batch {:10d} | Total {:10d}'.format(self.iteration, self.total_processed), flush=True)

    def build_and_save_index(self):
        """Goes through one epoch of the dataloader and adds all data to this instance's BlockData.

        The copy of BlockData is saved as a shard, which when run in a distributed setting will be
        consolidated by the rank 0 process and saved as a final pickled BlockData.
        """

        while True:
            try:
                # batch also has query_tokens and query_pad_data
                _, _, block_tokens, block_pad_mask, block_sample_data = get_ict_batch(self.dataloader)
            except (StopIteration, IndexError):
                break

            unwrapped_model = self.model
            while not hasattr(unwrapped_model, 'embed_block'):
                unwrapped_model = unwrapped_model.module

            # detach, separate fields and add to BlockData
            block_logits = detach(unwrapped_model.embed_block(block_tokens, block_pad_mask))
            detached_data = detach(block_sample_data)

            # block_sample_data is a 2D array [batch x 4]
            # with columns [start_idx, end_idx, doc_idx, block_idx] same as class BlockSampleData
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
            # make sure that every single piece of data was embedded
            assert len(self.block_data.embed_data) == len(self.dataset)
        self.block_data.clear()
