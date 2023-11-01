# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.

from functools import partial
import numpy as np
import os
import time
import torch
from torch.utils.data import BatchSampler, DataLoader, SequentialSampler, Subset
from torch.utils.data._utils.collate import default_collate
from tqdm import tqdm

from megatron import get_args, get_tokenizer, print_rank_0
from megatron import core
from megatron.arguments import core_transformer_config_from_args
from megatron.core.enums import ModelType
from megatron.core.pipeline_parallel import get_forward_backward_func
from megatron.model import BertModel
from megatron.training import setup_model_and_optimizer

from .dataset import BertEmbeddingDataset
from .external_libs import h5py
from .huggingface import HuggingfaceEmbedder
from .utils import get_missing_blocks_by_rank


def model_provider(pre_process=True, post_process=True):
    """Build the model."""

    print_rank_0(" > build Bert model.")

    args = get_args()
    config = core_transformer_config_from_args(args)
    num_tokentypes = 2 if args.bert_binary_head else 0
    model = BertModel(
        config=config,
        num_tokentypes=num_tokentypes,
        add_binary_head=args.bert_binary_head,
        parallel_output=True,
        pre_process=pre_process,
        post_process=post_process)

    return model


def get_batch(data_iterator):
    """Build the batch."""

    # Items and their type.
    keys = ['text', 'types', 'labels', 'is_random', 'loss_mask', 'padding_mask',
            'seq_length']
    datatype = torch.int64

    # Broadcast data.
    if data_iterator is not None:
        data = next(data_iterator)
    else:
        data = None
    data_b = core.tensor_parallel.broadcast_data(keys, data, datatype)

    # Unpack.
    tokens = data_b['text'].long()
    types = data_b['types'].long()
    sentence_order = data_b['is_random'].long()
    loss_mask = data_b['loss_mask'].float()
    lm_labels = data_b['labels'].long()
    padding_mask = data_b['padding_mask'].long()
    seq_lengths = data_b['seq_length'].long()

    return tokens, types, sentence_order, loss_mask, lm_labels, padding_mask, \
        seq_lengths


def loss_func(loss_mask, sentence_order, seq_lengths,
              output_tensor, non_loss_data):
    """Loss function. Sequence lengths returned here for progress print-outs."""
    assert non_loss_data
    return seq_lengths, output_tensor


def forward_step(data_iterator, model):
    """Forward step."""

    args = get_args()

    # Get the batch.
    tokens, types, sentence_order, loss_mask, lm_labels, padding_mask, \
        seq_lengths = get_batch(data_iterator)

    if not args.bert_binary_head:
        types = None

    # Forward pass through the model.
    output_tensor = model(tokens, padding_mask, tokentype_ids=types,
                          lm_labels=lm_labels)

    return output_tensor, partial(loss_func, loss_mask, sentence_order,
                                  seq_lengths)


def collate_batch(samples):
    """Collate samples of various lengths.

    This collate function handles samples with various sequence lengths, by
    padding 'text' arrays with pad_id, and other arrays with 0.
    """

    n_samples = len(samples)
    keys = list(samples[0].keys())
    tokenizer = get_tokenizer()

    # Max sample length across all samples.
    max_length_map = { key:0 for key in keys }
    for sample in samples:
        for key in keys:
            value_length = \
                len(sample[key]) if isinstance(sample[key], np.ndarray) else None
            max_length_map[key] = None \
                if value_length is None else \
                   max(max_length_map[key], value_length)

    # Pad samples.
    padded_samples = []
    for sample in samples:
        padded_sample = {}
        for key in keys:
            padded_sample[key] = \
                np.pad(
                    sample[key],
                    (0, max_length_map[key] - len(sample[key])),
                    mode="constant",
                    constant_values=tokenizer.pad_id if key == "text" else 0,
                ) \
                if isinstance(sample[key], np.ndarray) else \
                   sample[key]
        padded_samples.append(padded_sample)

    # Build batch with padded samples.
    batch = default_collate(padded_samples)

    return batch


def get_data_loader(dataset, batch_size):
    """Build data loader over data subset.

    Get a subset of the dataset (from start_idx -> end_idx), and wrap it in
    a sequential sampler and data loader.
    """

    args = get_args()

    # Sequential & batch samplers.
    batch_sampler = BatchSampler(
        sampler=SequentialSampler(dataset),
        batch_size=batch_size,
        drop_last=False,
    )

    # Data loader.
    data_loader = DataLoader(dataset,
                             batch_sampler=batch_sampler,
                             num_workers=args.num_workers,
                             pin_memory=True,
                             collate_fn=collate_batch)

    return data_loader


def embed_data_loader(models, data_loader):
    '''Iterate data loader and compute embeddings.'''

    # Verify no model parallelism.
    args = get_args()
    assert args.tensor_model_parallel_size == 1 and \
        args.pipeline_model_parallel_size == 1, \
        "since we call forward_step directly, only tp == pp == 1 allowed."

    # Data iterator.
    data_iterator = iter(data_loader)

    # Eval mode.
    for m in models:
        m.eval()

    # Embed.
    embeddings = []
    for _ in tqdm(range(len(data_loader)), "mt embed"):
        with torch.no_grad():
            result = forward_step(data_iterator, models[0])
            embeddings.append(result[0].detach().cpu().numpy())

    # Concatenate embeddings.
    embeddings = np.concatenate(embeddings, axis=0)

    return embeddings


class BertEmbedder:
    '''Compute Bert embeddings, from a text dataset.'''

    def __init__(self, batch_size, max_bert_seq_length, embedder_type):

        args = get_args()

        assert args.output_bert_embeddings

        self.models, optimizer, opt_param_scheduler = \
            setup_model_and_optimizer(model_provider,
                                      ModelType.encoder_or_decoder)
        self.batch_size = batch_size
        self.max_bert_seq_length = max_bert_seq_length

        # Init Huggingface, if in use.
        if embedder_type == "megatron":
            self.huggingface_embedder = None
        elif embedder_type == "huggingface":
            self.huggingface_embedder = HuggingfaceEmbedder(batch_size,
                                                            max_bert_seq_length)
        else:
            raise Exception("specialize for embedder type '%s'." % embedder_type)

    def embed_text_dataset(self, text_dataset):
        '''Embed a text dataset.'''

        # Huggingface.
        if self.huggingface_embedder:
            return self.huggingface_embedder.embed_text_dataset(text_dataset)

        # Wrap in a BertEmbeddingDataset to tokenize samples.
        bert_dataset = BertEmbeddingDataset(text_dataset,
                                            self.max_bert_seq_length)

        # Embed.
        data_loader = get_data_loader(bert_dataset, self.batch_size)
        embeddings = embed_data_loader(self.models, data_loader)

        return embeddings

    def embed_text(self, text):
        '''Embed a single text string.

        Primarily used for on-the-fly embeddings, particularly during
        analysis or debugging. For large scale, use 'embed_text_dataset()'.
        '''

        class SingleTextDataset(torch.utils.data.Dataset):
            '''Dataset that holds single string.'''
            def __init__(self, text):
                assert isinstance(text, str)
                self.text = text
            def __len__(self):
                return 1
            def __getitem__(self, i):
                return {"text": self.text}

        # Embed text.
        text_ds = SingleTextDataset(text)
        embed = self.embed_text_dataset(text_ds)[0]

        return embed


class DiskDataParallelBertEmbedder:
    '''Process embeddings in blocks & save to disk.'''

    def __init__(self, batch_size, max_bert_seq_length, block_size,
                 embedder_type):
        self.embedder = BertEmbedder(batch_size, max_bert_seq_length,
                                     embedder_type)
        self.block_size = block_size

    def embed_text_blocks(self, name, workdir, text_dataset,
                          missing_embedding_blocks):
        '''Process a text dataset in blocks.'''

        # Iterate blocks.
        for block_index, block_info in enumerate(missing_embedding_blocks):

            # Missing block lists are extended with None to have equal-length
            # lists. Skip the Nones.
            if block_info is not None:

                # Progress. (*note*: move world progress to here.)
                print_rank_0("embed '%s' block %d / %d ... %s." % (
                    name,
                    block_index,
                    len(missing_embedding_blocks),
                    block_info["path"],
                ))

                # Embed block.
                sub_dataset = Subset(text_dataset, range(*block_info["range"]))
                embeddings = self.embedder.embed_text_dataset(sub_dataset)

                # Save embeddings.
                f = h5py.File(block_info["path"], "w")
                f.create_dataset("data", data=embeddings)
                f.close()

            # Synchronize progress across all ranks. (for easier observation)
            print_rank_0(" > waiting for other ranks to finish block.")
            torch.distributed.barrier()

    def embed_text_dataset(self, name, workdir, text_dataset):
        '''Embed a text dataset.'''

        # Dataset workdir.
        os.makedirs(workdir, exist_ok=True)

        # Missing embedding blocks (stored on disk).
        def validate(f):
            assert f["data"].shape[1] == 1024
        n_missing_world, missing_embedding_blocks = get_missing_blocks_by_rank(
            workdir,
            len(text_dataset),
            self.block_size,
            validate=validate)

        # Prevent missing file race condition.
        torch.distributed.barrier()

        # Embed batches.
        self.embed_text_blocks(name, workdir, text_dataset,
                               missing_embedding_blocks)
