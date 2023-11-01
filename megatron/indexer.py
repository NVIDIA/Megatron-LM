import sys
import time
import torch
import torch.distributed as dist

from megatron import get_args, print_rank_0
from megatron.core import mpu
from megatron.checkpointing import load_biencoder_checkpoint
from megatron.data.orqa_wiki_dataset import get_open_retrieval_wiki_dataset
from megatron.data.orqa_wiki_dataset import get_open_retrieval_batch
from megatron.data.biencoder_dataset_utils import get_one_epoch_dataloader
from megatron.data.realm_index import detach, OpenRetreivalDataStore
from megatron.model.biencoder_model import get_model_provider
from megatron.training import get_model


class IndexBuilder(object):
    """
    Object for taking one pass over a dataset and creating a BlockData of its
    embeddings
    """
    def __init__(self):
        args = get_args()
        self.model = None
        self.dataloader = None
        self.evidence_embedder_obj = None
        self.biencoder_shared_query_context_model = \
            args.biencoder_shared_query_context_model

        # need to know whether we're using a REALM checkpoint (args.load)
        # or ICT checkpoint
        assert not (args.load and args.ict_load)

        self.log_interval = args.indexer_log_interval
        self.batch_size = args.indexer_batch_size

        self.load_attributes()
        self.is_main_builder = mpu.get_data_parallel_rank() == 0
        self.num_total_builders = mpu.get_data_parallel_world_size()
        self.iteration = self.total_processed = 0

    def load_attributes(self):
        """
        Load the necessary attributes: model, dataloader and empty BlockData
        """
        only_context_model = True
        if self.biencoder_shared_query_context_model:
            only_context_model = False

        model = get_model(get_model_provider(only_context_model=\
            only_context_model, biencoder_shared_query_context_model=\
            self.biencoder_shared_query_context_model))

        self.model = load_biencoder_checkpoint(model,
                only_context_model=only_context_model)

        assert len(self.model) == 1
        self.model[0].eval()

        self.dataset = get_open_retrieval_wiki_dataset()
        self.dataloader = iter(get_one_epoch_dataloader(self.dataset, \
            self.batch_size))

        self.evidence_embedder_obj = OpenRetreivalDataStore( \
            load_from_path=False)

    def track_and_report_progress(self, batch_size):
        """
        Utility function for tracking progress
        """
        self.iteration += 1
        self.total_processed += batch_size * self.num_total_builders
        if self.is_main_builder and self.iteration % self.log_interval == 0:
            print('Batch {:10d} | Total {:10d}'.format(self.iteration,
                self.total_processed), flush=True)

    def build_and_save_index(self):
        """
        Goes through one epoch of the dataloader and adds all data to this
        instance's BlockData.

        The copy of BlockData is saved as a shard, which when run in a
        distributed setting will be consolidated by the rank 0 process
        and saved as a final pickled BlockData.
        """
        assert len(self.model) == 1
        unwrapped_model = self.model[0]

        while not hasattr(unwrapped_model, 'embed_text'):
            unwrapped_model = unwrapped_model.module

        while True:
            try:
                # batch also has query_tokens and query_pad_data
                row_id, context_tokens, context_mask, context_types, \
                    context_pad_mask = get_open_retrieval_batch( \
                    self.dataloader)
            except (StopIteration, IndexError):
                break

            # TODO: can we add with torch.no_grad() to reduce memory usage
            # detach, separate fields and add to BlockData
            assert context_mask.dtype == torch.bool
            context_logits = unwrapped_model.embed_text(
                unwrapped_model.context_model, context_tokens, context_mask,
                context_types)

            context_logits = detach(context_logits)
            row_id = detach(row_id)

            self.evidence_embedder_obj.add_block_data(row_id, context_logits)
            self.track_and_report_progress(batch_size=len(row_id))

        # This process signals to finalize its shard and then synchronize with
        # the other processes
        self.evidence_embedder_obj.save_shard()
        torch.distributed.barrier()
        del self.model

        # rank 0 process builds the final copy
        if self.is_main_builder:
            self.evidence_embedder_obj.merge_shards_and_save()
            # make sure that every single piece of data was embedded
            assert len(self.evidence_embedder_obj.embed_data) == \
                len(self.dataset)
        self.evidence_embedder_obj.clear()

        # complete building the final copy
        torch.distributed.barrier()
