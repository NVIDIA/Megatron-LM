# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.

import json
import numpy as np
import os
import typing as T
from types import SimpleNamespace

from megatron.training.arguments import load_retro_config, parse_args, validate_args
from megatron.core.datasets.retro.db.dataset import DBDataset
from megatron.core.datasets.retro.db.utils import (
    get_indexed_dataset_infos as get_db_indexed_dataset_infos,
    get_merged_train_dataset as get_db_dataset,
)
from megatron.core.datasets.retro.query.retro_dataset import get_retro_datasets, RetroDataset
from megatron.training.global_vars import set_global_variables
from megatron.training.training import build_train_valid_test_datasets, update_train_iters
from pretrain_retro import train_valid_test_datasets_provider
from tools.retro.preprocess_data import get_tokenizers


def shorten_str(s: str, n: int) -> str:
    s = "\\n".join(s.splitlines())
    return s if len(s) <= n else "%s ... %s" % (s[: n // 2], s[-n // 2 :])


class retro:

    config = None

    ##############################################
    # initialize.
    ##############################################

    @classmethod
    def init(cls, project_dir: str) -> None:
        '''Initialize Megatron, tokenizers, and datasets.'''

        # Megatron args.
        args = parse_args(extra_args_provider=None, ignore_unknown_args=False)
        args.retro_project_dir = project_dir
        args.micro_batch_size = 1
        args.num_layers = 1
        args.hidden_size = 1
        args.num_attention_heads = 1
        args.async_tensor_model_parallel_allreduce = False
        args.retro_add_retriever = True # for building RetroDataset
        validate_args(args)
        set_global_variables(args)
        update_train_iters(args)

        # Retro config.
        cls.config = load_retro_config(project_dir)
        cls.config.retro_project_dir = project_dir
        cls.config.retro_tokenizers = get_tokenizers(cls.config)

        # Chunk database dataset.
        cls.db_indexed_dataset_infos = get_db_indexed_dataset_infos(project_dir)
        cls.db_dataset = get_db_dataset(project_dir,
                                        cls.config.retro_gpt_chunk_length,
                                        cls.config.retro_tokenizers.gpt.eod)

        # Pretraining datasets.
        pt_train_ds, pt_valid_ds, pt_test_ds = build_train_valid_test_datasets(
            train_valid_test_datasets_provider)
        cls.pt_datasets = SimpleNamespace(
            train=pt_train_ds,
            valid=pt_valid_ds,
            test=pt_test_ds,
        )

        # Print usage.
        cls.print_usage()

    ##############################################
    # utils.
    ##############################################

    @classmethod
    def gpt_to_text(cls, token_ids: np.ndarray) -> str:
        '''GPT tokens to text.'''
        return cls.config.retro_tokenizers.gpt.detokenize(
            token_ids.tolist() if isinstance(token_ids, np.ndarray) else token_ids
        )

    @classmethod
    def text_to_bert(cls, text: str) -> np.ndarray:
        '''Text to Bert tokens.'''
        return cls.config.retro_tokenizers.bert.tokenize(text)

    ##############################################
    # chunk db.
    ##############################################

    @classmethod
    def get_db_num_indexed_datasets(cls) -> int:
        '''Number of indexed datasets within blended dataset.'''
        return len(cls.db_indexed_dataset_infos)

    @classmethod
    def get_db_indexed_dataset_infos(cls) -> T.List[T.Tuple[float, str]]:
        '''Dataset infos, including number of training & sampled sets.'''
        return [(info["ratio"], info["prefix"]) for info in cls.db_indexed_dataset_infos]

    @classmethod
    def get_db_dataset(cls) -> DBDataset:
        return cls.db_dataset

    @classmethod
    def get_db_num_chunks(cls) -> int:
        '''Number of DB chunks.'''
        return len(cls.get_db_dataset())

    @classmethod
    def get_db_chunk_gpt(cls, idx: int) -> T.List[int]:
        '''Get DB chunk as GPT token ids.'''
        return cls.get_db_dataset()[idx]["text"].tolist()

    @classmethod
    def get_db_chunk_bert(cls, idx: int) -> T.List[int]:
        '''Get DB chunk as Bert token ids.'''
        return cls.text_to_bert(cls.get_db_chunk_text(idx))

    @classmethod
    def get_db_chunk_text(cls, idx: int) -> str:
        '''Get DB chunk as text.'''
        return cls.gpt_to_text(cls.get_db_chunk_gpt(idx))

    @classmethod
    def get_db_chunk_and_continuation_text(cls, idx: int) -> T.List[str]:
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
    def get_pt_num_samples_and_chunks(cls, data_key: str) -> T.Tuple[int, int]:
        '''Number of samples & chunks (e.g., 32*n_samples) in corpus.'''
        assert hasattr(cls.pt_datasets, data_key), (
            "pretraining set '%s' not found (choices: %s)."
            % (data_key, ", ".join(vars(cls.pt_datasets).keys()))
        )
        chunk_dataset = getattr(cls.pt_datasets, data_key).chunk_dataset
        return (
            len(chunk_dataset.sample_dataset),
            len(chunk_dataset),
        )

    @classmethod
    def get_pt_num_samples(cls, data_key: str) -> int:
        '''Number of pretraining samples.'''
        return cls.get_pt_num_samples_and_chunks(data_key)[0]

    @classmethod
    def get_pt_num_chunks(cls, data_key: str) -> int:
        '''Number of pretraining chunks (e.g., 32*n_samples).'''
        return cls.get_pt_num_samples_and_chunks(data_key)[1]

    @classmethod
    def get_pt_dataset(cls, data_key: str) -> RetroDataset:
        return getattr(cls.pt_datasets, data_key)

    @classmethod
    def get_pt_sample(cls, data_key: str, idx: int) -> dict:
        return getattr(cls.pt_datasets, data_key)[idx]

    @classmethod
    def get_neighbor_tokens(cls, sample_id: int, chunk_id: int, data_key: str="train") -> T.Optional[dict]:
        try:
            sample = cls.get_pt_sample(data_key, sample_id)
            sample_token_ids = sample["text"]
            chunk_length = cls.args.retro_gpt_chunk_length
            chunk_start_idx = chunk_id * chunk_length
            chunk_end_idx = min(sample_token_ids.shape[0], chunk_start_idx + chunk_length)
            chunk_token_ids = sample_token_ids[chunk_start_idx:chunk_end_idx]
            neighbor_token_ids = sample["neighbor_tokens"][chunk_id]
            return {
                "chunk_tokens": chunk_token_ids,
                "neighbor_tokens": neighbor_token_ids,
            }
        except:
            return None

    @classmethod
    def print_neighbor_texts(cls, sample_id: int, chunk_id: int, data_key: str="train") -> None:
        tokens: dict = cls.get_neighbor_tokens(sample_id, chunk_id, data_key)
        print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
        try:
            print("PRETRAINING CHUNK:")
            print("  - %s" % shorten_str(cls.gpt_to_text(tokens["chunk_tokens"]), 150))
            print("NEIGHBOR_CHUNKS:")
            for token_ids in tokens["neighbor_tokens"]:
                print("  - %s" % shorten_str(cls.gpt_to_text(token_ids), 150))
        except:
            print("<no neighbors for sample %d>" % sample_id)

    ##############################################
    # usage.
    ##############################################

    @classmethod
    def print_usage(cls) -> None:
        '''Print usage.'''

        print()
        print("+++++++++++++++++++++++++++++++++++++++++++++++++++")
        print("examples ... [ *note*: 'db' = chunk db; 'pt' = pretraining corpus. ]")
        print("+++++++++++++++++++++++++++++++++++++++++++++++++++")

        print()
        print("~~~~ indexed datasets ~~~~")
        print("retro.get_db_num_indexed_datasets() : %s" % cls.get_db_num_indexed_datasets())
        print("retro.get_db_indexed_dataset_infos() :")
        for i, (ratio, prefix) in enumerate(cls.get_db_indexed_dataset_infos()):
            print(
                "  %s(%f, %s)%s"
                % (
                    "[" if i == 0 else " ",
                    ratio,
                    prefix,
                    "]" if i == len(cls.db_indexed_dataset_infos) - 1 else ",",
                )
            )

        print()
        print("~~~~ counts ~~~~")
        print("retro.get_db_num_chunks : %d." % cls.get_db_num_chunks())

        print()
        for sq_key in ("sample", "chunk"):
            for data_key in ("train", "valid"):  # test?
                print(
                    "retro.get_pt_num_%ss('%s') : %d."
                    % (sq_key, data_key, getattr(cls, f"get_pt_num_{sq_key}s")(data_key))
                )

        print()
        print("~~~~ tokens, text ~~~~")
        print(
            "retro.get_db_chunk_gpt(chunk_id) : %s"
            % shorten_str(str(retro.get_db_chunk_gpt(0)), 50)
        )
        print(
            "retro.get_db_chunk_bert(chunk_id) : %s"
            % shorten_str(str(retro.get_db_chunk_bert(0)), 50)
        )
        print(
            "retro.get_db_chunk_text(chunk_id) : %s"
            % shorten_str(retro.get_db_chunk_text(0).strip(), 50)
        )
        print("retro.get_db_chunk_and_continuation_text(chunk_id) :")
        for i, t in enumerate(retro.get_db_chunk_and_continuation_text(0)):
            print(
                "  %s'%s'%s"
                % (
                    "[" if i == 0 else " ",
                    shorten_str(t.strip().replace("\n", " "), 50),
                    "]" if i == 1 else ",",
                )
            )

        sample = cls.get_pt_sample("train", 0)
        sample_chunk_id = sample["neighbor_tokens"].shape[0] // 2
        sample_neighbor_id = 0
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
        print(
            "  sample['neighbor_tokens'][17][1] : %s"
            % shorten_str(str(sample["neighbor_tokens"][sample_chunk_id][sample_neighbor_id]), 50)
        )
        print(
            "  retro.gpt_to_text(sample['text']) : %s"
            % shorten_str(cls.gpt_to_text(sample["text"]), 50)
        )
        print(
            "  retro.gpt_to_text(sample['neighbor_tokens']) : %s"
            % shorten_str(
                cls.gpt_to_text(sample["neighbor_tokens"][sample_chunk_id][sample_neighbor_id]), 50
            )
        )

        print("+++++++++++++++++++++++++++++++++++++++++++++++++++")
