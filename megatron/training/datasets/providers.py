# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""The GPT dataset provider shared by the GPT entry points.

``build_gpt_datasets`` is the body of ``train_valid_test_datasets_provider`` in ``pretrain_gpt.py``,
``pretrain_hybrid.py`` and ``megatron/elastification/pretrain_hybrid_flex.py``: it takes ``cfg.data``
(the dataset config bound to the run's tokenizer, the loader settings and the dataset class)
and hands the dataset config to ``BlendedMegatronDatasetBuilder`` as is. Each script keeps a
``train_valid_test_datasets_provider`` of the legacy shape that calls it with its rank predicate:
``is_dataset_built_on_rank`` below is the GPT and hybrid rule (a packed dataset is built on every
pipeline stage, the others on the first and last stages and on the multi-token-prediction ranks),
hybrid-flex passes its own.
"""

from functools import partial
from typing import Any, Callable, Optional

from megatron.core import mpu
from megatron.core.datasets.blended_megatron_dataset_builder import BlendedMegatronDatasetBuilder
from megatron.core.transformer.multi_token_prediction import mtp_on_this_rank


def is_dataset_built_on_rank(
    vp_stage: Optional[int] = None, is_packed_sequence: bool = False
) -> bool:
    """Whether the dataset should be built on the current rank (the GPT and hybrid rule)."""
    from megatron.training.arguments import core_transformer_config_from_args
    from megatron.training.global_vars import get_args
    from megatron.training.utils import is_first_or_last_pipeline_stage

    args = get_args()
    config = core_transformer_config_from_args(args)
    if mpu.get_tensor_model_parallel_rank() != 0:
        return False
    elif args.sequence_packing_scheduler is not None:
        # The scheduler sends packing metadata to middle PP stages itself and requires that
        # only the first and last stages hold a data iterator.
        return is_first_or_last_pipeline_stage(vp_stage)
    elif is_packed_sequence:
        return True
    return is_first_or_last_pipeline_stage(vp_stage) or mtp_on_this_rank(
        layout=config.pipeline_model_parallel_layout,
        mtp_num_layers=config.mtp_num_layers,
        ignore_virtual=False,
        vp_stage=vp_stage,
    )


def build_gpt_datasets(
    train_val_test_num_samples,
    vp_stage: Optional[int] = None,
    *,
    data_config: Any = None,
    built_on_rank: Optional[Callable[..., bool]] = None,
):
    """Build the train, validation and test datasets from ``cfg.data``.

    Args:
        train_val_test_num_samples: the number of samples for the train, validation and test
            splits.
        vp_stage: the virtual pipeline stage the datasets are built for.
        data_config: ``cfg.data``; the registered run config's when omitted (the scripts'
            providers), given explicitly by tests and direct callers.
        built_on_rank: the rank predicate, called with ``vp_stage`` and ``is_packed_sequence``;
            ``is_dataset_built_on_rank`` (the GPT and hybrid rule) when omitted.
    """
    from megatron.training.global_vars import get_run_config
    from megatron.training.utils import print_rank_0

    if data_config is None:
        data_config = get_run_config().data
    if built_on_rank is None:
        built_on_rank = is_dataset_built_on_rank
    dataset_type, config, is_packed_sequence = data_config.dataset_build()

    print_rank_0("> building train, validation, and test datasets for GPT ...")

    train_ds, valid_ds, test_ds = BlendedMegatronDatasetBuilder(
        dataset_type,
        train_val_test_num_samples,
        partial(built_on_rank, vp_stage=vp_stage, is_packed_sequence=is_packed_sequence),
        config,
    ).build()

    print_rank_0("> finished creating GPT datasets ...")

    return train_ds, valid_ds, test_ds
