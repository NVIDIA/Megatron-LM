# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.

import json
import numpy as np
import os
import torch
import types

from megatron.global_vars import set_global_variables, set_retro_args
from megatron.initialize import (
    initialize_megatron,
    _initialize_distributed,
    _set_random_seed,
)
from tools.retro.db.utils import (
    get_indexed_dataset_infos as get_db_indexed_dataset_infos,
    get_merged_train_dataset as get_db_dataset,
)
from tools.retro.external_libs import h5py
from tools.retro.main import add_retro_args
from tools.retro.pretraining.retro_dataset import get_retro_datasets
from tools.retro.utils import get_args_path, get_bert_tokenizer, get_gpt_tokenizer


def shorten_str(s, n):
    s = "\\n".join(s.splitlines())
    return s if len(s) <= n else "%s ... %s" % (s[:n//2], s[-n//2:])


class retro:

    args = None

    ##############################################
    # initialize.
    ##############################################

    @classmethod
    def init_megatron(cls, workdir):
        '''Custom initialization of Megatron.'''

        # Load args.
        args_path = get_args_path(workdir)
        assert os.path.exists(args_path), "args.json not found in workdir."
        with open(args_path) as f:
            cls.args = types.SimpleNamespace(**json.load(f))
            cls.args.retro_workdir = workdir # just in case workdir moved
            cls.args.rank = 0 # override env
            cls.args.world_size = 1 # override env

        set_global_variables(cls.args)
        set_retro_args(cls.args)
        _initialize_distributed()
        _set_random_seed(cls.args.seed, cls.args.data_parallel_random_init)

    @classmethod
    def init(cls, workdir):
        '''Initialize Megatron, tokenizers, and datasets.'''

        # Load args.
        cls.init_megatron(workdir)

        cls.tokenizers = types.SimpleNamespace(
            gpt=get_gpt_tokenizer(),
            bert=get_bert_tokenizer(),
        )

        # Load data.
        cls.db_indexed_dataset_infos = get_db_indexed_dataset_infos()
        pt_train_ds, pt_valid_ds, _ = get_retro_datasets()
        cls.pt_datasets = types.SimpleNamespace(
            train=pt_train_ds,
            valid=pt_valid_ds,
        )

        # Print usage.
        cls.print_usage()

    ##############################################
    # utils.
    ##############################################

    @classmethod
    def gpt_to_text(cls, token_ids):
        '''GPT tokens to text.'''
        return cls.tokenizers.gpt.detokenize(token_ids)

    @classmethod
    def text_to_bert(cls, text):
        '''Text to Bert tokens.'''
        return cls.tokenizers.bert.tokenize(text)

    ##############################################
    # chunk db.
    ##############################################

    @classmethod
    def get_db_num_indexed_datasets(cls):
        '''Number of indexed datasets within blendable dataset.'''
        return len(cls.db_indexed_dataset_infos)

    @classmethod
    def get_db_indexed_dataset_infos(cls):
        '''Dataset infos, including number of training & sampled sets.'''
        return [(info["ratio"], info["name"])
                for info in cls.db_indexed_dataset_infos]

    @classmethod
    def get_db_dataset(cls):
        return cls.pt_datasets.train.db_dataset

    @classmethod
    def get_db_num_chunks(cls):
        '''Number of DB chunks.'''
        return len(cls.get_db_dataset())

    @classmethod
    def get_db_chunk_gpt(cls, idx):
        '''Get DB chunk as GPT token ids.'''
        return cls.get_db_dataset()[idx]["text"].tolist()

    @classmethod
    def get_db_chunk_bert(cls, idx):
        '''Get DB chunk as Bert token ids.'''
        return cls.text_to_bert(cls.get_db_chunk_text(idx))

    @classmethod
    def get_db_chunk_text(cls, idx):
        '''Get DB chunk as text.'''
        return cls.gpt_to_text(cls.get_db_chunk_gpt(idx))

    @classmethod
    def get_db_chunk_and_continuation_text(cls, idx):
        '''Get DB chunk along with continuation, as text.'''

        # Modulus used here to match original implementation (i.e., last
        # chunks continuation wraps around to first chunk).
        return [
            cls.get_db_chunk_text(idx),
            cls.get_db_chunk_text((idx + 1) % len(cls.get_db_dataset())),
        ]

    ##############################################
    # pretraining corpus.
    ##############################################

    @classmethod
    def get_pt_num_samples_and_chunks(cls, data_key):
        '''Number of samples & chunks (e.g., 32*n_samples) in corpus.'''
        assert hasattr(cls.pt_datasets, data_key), \
            "pretraining set '%s' not found (choices: %s)." % (
                data_key, ", ".join(vars(cls.pt_datasets).keys()))
        chunk_dataset = getattr(cls.pt_datasets, data_key).chunk_dataset
        return (
            len(chunk_dataset.sample_dataset),
            len(chunk_dataset),
        )

    @classmethod
    def get_pt_num_samples(cls, data_key):
        '''Number of pretraining samples.'''
        return cls.get_pt_num_samples_and_chunks(data_key)[0]

    @classmethod
    def get_pt_num_chunks(cls, data_key):
        '''Number of pretraining chunks (e.g., 32*n_samples).'''
        return cls.get_pt_num_samples_and_chunks(data_key)[1]

    @classmethod
    def get_pt_sample(cls, data_key, idx):
        return getattr(cls.pt_datasets, data_key)[idx]

    ##############################################
    # usage.
    ##############################################

    @classmethod
    def print_usage(cls):
        '''Print usage.'''

        print()
        print("+++++++++++++++++++++++++++++++++++++++++++++++++++")
        print("examples ... [ *note*: 'db' = chunk db; 'pt' = pretraining corpus. ]")
        print("+++++++++++++++++++++++++++++++++++++++++++++++++++")

        print()
        print("~~~~ indexed datasets ~~~~")
        print("retro.get_db_num_indexed_datasets() : %s" %
              cls.get_db_num_indexed_datasets())
        print("retro.get_db_indexed_dataset_infos() :")
        for i, (ratio,prefix) in enumerate(cls.get_db_indexed_dataset_infos()):
            print("  %s(%f, %s)%s" % (
                "[" if i == 0 else " ",
                ratio,
                prefix,
                "]" if i == len(cls.db_indexed_dataset_infos) - 1 else ",",
            ))

        print()
        print("~~~~ counts ~~~~")
        print("retro.get_db_num_chunks : %d." % cls.get_db_num_chunks())

        print()
        for sq_key in ("sample", "chunk"):
            for data_key in ("train", "valid"): # test?
                print("retro.get_pt_num_%ss('%s') : %d." % (
                    sq_key, data_key,
                    getattr(cls, f"get_pt_num_{sq_key}s")(data_key)))

        print()
        print("~~~~ tokens, text ~~~~")
        print("retro.get_db_chunk_gpt(chunk_id) : %s" %
              shorten_str(str(retro.get_db_chunk_gpt(0)), 50))
        print("retro.get_db_chunk_bert(chunk_id) : %s" %
              shorten_str(str(retro.get_db_chunk_bert(0)), 50))
        print("retro.get_db_chunk_text(chunk_id) : %s" %
              shorten_str(retro.get_db_chunk_text(0).strip(), 50))
        print("retro.get_db_chunk_and_continuation_text(chunk_id) :")
        for i, t in enumerate(retro.get_db_chunk_and_continuation_text(0)):
            print("  %s'%s'%s" % (
                "[" if i == 0 else " ",
                shorten_str(t.strip().replace("\n", " "), 50),
                "]" if i == 1 else ",",
            ))

        sample = cls.get_pt_sample("train", 0)
        print()
        print("retro.get_pt_sample('train', sample_id) :")
        print("  {")
        for k, v in sample.items():
            print("    '%s' : %s" % (k, shorten_str(str(v), 50)))
        print("  }")

        print()
        print("(e.g., sample = retro.get_pt_sample(...))")
        print()
        print("  sample['text'].shape : %s" % str(sample["text"].shape))
        print("  sample['neighbor_tokens'].shape : %s" % str(sample["neighbor_tokens"].shape))
        print("  sample['text'] : %s" % shorten_str(str(sample["text"]), 50))
        print("  sample['neighbor_tokens'][17][1] : %s" % shorten_str(str(sample["neighbor_tokens"][17][1]), 50))
        print("  retro.gpt_to_text(sample['text']) : %s" % shorten_str(cls.gpt_to_text(sample["text"]), 50))
        print("  retro.gpt_to_text(sample['neighbor_tokens']) : %s" % shorten_str(cls.gpt_to_text(sample["neighbor_tokens"][17][1]), 50))

        print("+++++++++++++++++++++++++++++++++++++++++++++++++++")
