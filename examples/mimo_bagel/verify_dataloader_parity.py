# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Compare native BAGEL and MCore PackedDataset batches field by field."""

import argparse
import copy
import hashlib
import os
import random
from functools import partial
from types import SimpleNamespace

import numpy as np
import torch
from torch.utils.data import DataLoader


def _assert_equal(reference, actual, path="batch"):
    if isinstance(reference, torch.Tensor):
        assert isinstance(actual, torch.Tensor), f"{path}: expected tensor, got {type(actual)}"
        assert reference.dtype == actual.dtype, f"{path}: dtype {reference.dtype} != {actual.dtype}"
        assert (
            reference.shape == actual.shape
        ), f"{path}: shape {tuple(reference.shape)} != {tuple(actual.shape)}"
        if not torch.equal(reference, actual):
            mismatch = torch.ne(reference, actual)
            first = mismatch.reshape(-1).nonzero()[0].item()
            raise AssertionError(f"{path}: tensor values differ at flat index {first}")
        return

    assert type(reference) is type(actual), f"{path}: type {type(reference)} != {type(actual)}"
    if isinstance(reference, dict):
        assert (
            reference.keys() == actual.keys()
        ), f"{path}: keys {reference.keys()} != {actual.keys()}"
        for key in reference:
            _assert_equal(reference[key], actual[key], f"{path}.{key}")
    elif isinstance(reference, (list, tuple)):
        assert len(reference) == len(actual), f"{path}: length {len(reference)} != {len(actual)}"
        for index, (reference_item, actual_item) in enumerate(zip(reference, actual)):
            _assert_equal(reference_item, actual_item, f"{path}[{index}]")
    else:
        assert reference == actual, f"{path}: {reference!r} != {actual!r}"


def _seed_native_worker(worker_id: int, *, rank_seed: int) -> None:
    """Mirror the explicit worker seed contract in native BAGEL training."""

    worker_seed = rank_seed + worker_id
    random.seed(worker_seed)
    np.random.seed(worker_seed % (2**32))
    torch.manual_seed(worker_seed)


def _batch_fingerprint(batch) -> str:
    """Return a stable short digest for fields that define training order."""

    digest = hashlib.sha256()
    for key in (
        'batch_data_indexes',
        'packed_text_ids',
        'packed_label_ids',
        'ce_loss_indexes',
        'mse_loss_indexes',
        'packed_vae_token_indexes',
    ):
        value = batch.get(key)
        digest.update(key.encode('utf-8'))
        if isinstance(value, torch.Tensor):
            tensor = value.detach().cpu().contiguous()
            digest.update(str(tensor.dtype).encode('ascii'))
            digest.update(str(tuple(tensor.shape)).encode('ascii'))
            digest.update(tensor.numpy().tobytes())
        else:
            digest.update(repr(value).encode('utf-8'))
    return digest.hexdigest()[:16]


def _parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset-config-file', required=True)
    parser.add_argument('--bagel-example-path', required=True)
    parser.add_argument('--tokenizer-model', required=True)
    parser.add_argument('--num-batches', type=int, default=2)
    parser.add_argument('--num-workers', type=int, default=1)
    parser.add_argument('--rank', type=int, default=int(os.environ.get('RANK', 0)))
    parser.add_argument('--world-size', type=int, default=int(os.environ.get('WORLD_SIZE', 1)))
    parser.add_argument('--global-seed', type=int, default=4396)
    parser.add_argument('--data-seed', type=int, default=42)
    return parser.parse_args()


def main():
    cli_args = _parse_args()
    os.environ['BAGEL_EXAMPLE_PATH'] = cli_args.bagel_example_path

    import yaml
    from bagel.data.data_utils import add_special_tokens
    from bagel.data.dataset_base import DataConfig, PackedDataset, collate_wrapper
    from bagel.modeling.qwen2 import Qwen2Tokenizer

    from examples.mimo_bagel.configs.reference import get_reference_data_seed
    from examples.mimo_bagel.data.hf_dataloader import (
        _build_data_config_kwargs,
        _build_data_loader,
        _build_packed_dataset_kwargs,
    )

    with open(cli_args.dataset_config_file, 'r', encoding='utf-8') as stream:
        dataset_meta = yaml.safe_load(stream)
    tokenizer = Qwen2Tokenizer.from_pretrained(cli_args.tokenizer_model)
    tokenizer, special_tokens, _ = add_special_tokens(tokenizer)

    args = SimpleNamespace(
        text_cond_dropout_prob=0.1,
        vit_cond_dropout_prob=0.3,
        vae_cond_dropout_prob=0.3,
        max_latent_size=64,
        vit_patch_size=14,
        max_num_patch_per_side=70,
        expected_num_tokens=32768,
        max_num_tokens_per_sample=16384,
        max_num_tokens=36864,
        prefer_buffer_before=16384,
        packing_buffer_size=50,
        interpolate_pos=False,
        use_flex_attention=True,
        num_workers=cli_args.num_workers,
        prefetch_factor=2,
    )
    rank_seed = get_reference_data_seed(
        cli_args.global_seed,
        data_parallel_world_size=cli_args.world_size,
        data_parallel_rank=cli_args.rank,
    )

    # Native BAGEL construction, written independently of the MCore helpers.
    reference_config = DataConfig(
        grouped_datasets=copy.deepcopy(dataset_meta),
        text_cond_dropout_prob=0.1,
        vit_cond_dropout_prob=0.3,
        vae_cond_dropout_prob=0.3,
        vae_image_downsample=16,
        max_latent_size=64,
        vit_patch_size=14,
        max_num_patch_per_side=70,
    )
    reference_dataset = PackedDataset(
        reference_config,
        tokenizer=tokenizer,
        special_tokens=special_tokens,
        local_rank=cli_args.rank,
        world_size=cli_args.world_size,
        num_workers=cli_args.num_workers,
        expected_num_tokens=32768,
        max_num_tokens_per_sample=16384,
        max_num_tokens=36864,
        prefer_buffer_before=16384,
        max_buffer_size=50,
        interpolate_pos=False,
        use_flex=True,
    )
    reference_dataset.set_epoch(cli_args.data_seed)

    # MCore provider construction.
    mcore_config = DataConfig(
        grouped_datasets=copy.deepcopy(dataset_meta),
        **_build_data_config_kwargs(args, vae_image_downsample=16),
    )
    mcore_dataset = PackedDataset(
        **_build_packed_dataset_kwargs(
            args,
            data_config=mcore_config,
            tokenizer=tokenizer,
            special_tokens=special_tokens,
            data_parallel_rank=cli_args.rank,
            data_parallel_world_size=cli_args.world_size,
        )
    )
    mcore_dataset.set_epoch(cli_args.data_seed)

    reference_loader_kwargs = {
        'dataset': reference_dataset,
        'batch_size': 1,
        'num_workers': cli_args.num_workers,
        'pin_memory': True,
        'collate_fn': collate_wrapper(),
        'drop_last': True,
    }
    if cli_args.num_workers > 0:
        reference_generator = torch.Generator()
        reference_generator.manual_seed(rank_seed)
        reference_loader_kwargs.update(
            prefetch_factor=2,
            generator=reference_generator,
            worker_init_fn=partial(_seed_native_worker, rank_seed=rank_seed),
        )
    else:
        _seed_native_worker(0, rank_seed=rank_seed)
    reference_iterator = iter(DataLoader(**reference_loader_kwargs))
    mcore_iterator = iter(_build_data_loader(mcore_dataset, args, rank_seed))
    for batch_index in range(cli_args.num_batches):
        reference_batch = next(reference_iterator).to_dict()
        mcore_batch = next(mcore_iterator).to_dict()
        _assert_equal(reference_batch, mcore_batch, path=f"batch[{batch_index}]")
        ce_tokens = int(reference_batch['ce_loss_indexes'].numel())
        mse_positions = int(reference_batch['mse_loss_indexes'].numel())
        print(
            f"PASS batch={batch_index} sequence_length={reference_batch['sequence_length']} "
            f"samples={len(reference_batch['batch_data_indexes'])} "
            f"ce_tokens={ce_tokens} mse_positions={mse_positions} "
            f"fingerprint={_batch_fingerprint(reference_batch)}",
            flush=True,
        )

    print(
        f"PASS: {cli_args.num_batches} raw PackedDataset batches are bit-exact "
        f"for rank {cli_args.rank}/{cli_args.world_size}",
        flush=True,
    )


if __name__ == '__main__':
    main()
