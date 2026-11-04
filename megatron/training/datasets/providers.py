# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""The GPT dataset provider shared by the GPT entry points.

``build_gpt_datasets`` is the body of ``train_valid_test_datasets_provider`` in ``pretrain_gpt.py``,
``pretrain_hybrid.py`` and ``megatron/elastification/pretrain_hybrid_flex.py``: it takes ``cfg.data``
(the dataset config bound to the run's tokenizer, the loader settings and the dataset class)
and hands the dataset config to ``BlendedMegatronDatasetBuilder`` as is. Each script keeps a
``train_valid_test_datasets_provider`` of the legacy shape that calls it with its rank predicate:
``is_dataset_built_on_rank_explicit`` below is the GPT and hybrid rule with its inputs spelled out
(a packed dataset is built on every pipeline stage, the others on the first and last stages and on
the multi-token-prediction ranks); ``is_dataset_built_on_rank`` applies it to the registered
arguments and is the predicate ``pretrain_gpt.py`` and ``pretrain_hybrid.py`` pass; hybrid-flex
passes its own.
"""

from functools import partial
from typing import Any, Callable, Optional

from megatron.core import mpu
from megatron.core.datasets.blended_megatron_dataset_builder import BlendedMegatronDatasetBuilder
from megatron.core.process_groups_config import (
    ProcessGroupCollection,
    global_process_group_fallback,
)
from megatron.core.transformer.multi_token_prediction import mtp_on_this_rank
from megatron.core.utils import get_pg_rank


def is_dataset_built_on_rank_explicit(
    vp_stage: Optional[int],
    is_packed_sequence: bool,
    *,
    sequence_packing_scheduler: Optional[str],
    pipeline_model_parallel_layout: Any,
    mtp_num_layers: Optional[int],
    vp_size: Optional[int] = None,
    pg_collection: Optional[ProcessGroupCollection] = None,
) -> bool:
    """Whether the dataset should be built on the current rank: the GPT and hybrid rule, with the
    run's inputs given explicitly (``is_dataset_built_on_rank`` reads them from the arguments).

    Only the first rank of the tensor-parallel group builds. A sequence-packing scheduler or
    packed dataset builds on every pipeline stage; the others on the first and last stages of the
    pipeline group and on the multi-token-prediction ranks. ``vp_size`` is the model's number of
    virtual pipeline stages, 1 without a virtual pipeline; omitting it falls back to the global
    size, with a ``ProcessGroupFallbackWarning``. ``pg_collection`` holds the process groups of
    the run, whose tensor-parallel and pipeline groups decide the ranks; omitting it is
    deprecated: the groups are built from the global parallel state, with a
    ``ProcessGroupFallbackWarning``.
    """
    from megatron.training.utils import is_first_or_last_pipeline_stage

    if pg_collection is None:
        with global_process_group_fallback(
            "is_dataset_built_on_rank", deprecated_in="0.21", removed_in="0.23"
        ):
            pg_collection = ProcessGroupCollection.use_mpu_process_groups(required_pgs=["tp", "pp"])
    for group_name in ("tp", "pp"):
        if vars(pg_collection).get(group_name) is None:
            raise ValueError(f"is_dataset_built_on_rank requires pg_collection.{group_name}")
    if get_pg_rank(pg_collection.tp) != 0:
        return False
    if is_packed_sequence or sequence_packing_scheduler is not None:
        # The sequence-packing scheduler runs on every PP stage and reads its input there.
        return True
    if vp_size is None and vp_stage is not None:
        # Compatibility for direct callers; is_dataset_built_on_rank passes the model's size.
        with global_process_group_fallback(
            "is_dataset_built_on_rank", "vp_size", deprecated_in="0.21", removed_in="0.23"
        ):
            vp_size = mpu.get_virtual_pipeline_model_parallel_world_size() or 1
    return is_first_or_last_pipeline_stage(
        vp_stage, pp_group=pg_collection.pp, vp_size=vp_size
    ) or mtp_on_this_rank(
        layout=pipeline_model_parallel_layout,
        mtp_num_layers=mtp_num_layers,
        ignore_virtual=False,
        vp_stage=vp_stage,
        pp_group=pg_collection.pp,
        vp_size=vp_size,
    )


def is_dataset_built_on_rank(
    vp_stage: Optional[int] = None,
    is_packed_sequence: bool = False,
    pg_collection: Optional[ProcessGroupCollection] = None,
) -> bool:
    """``is_dataset_built_on_rank_explicit`` applied to the registered arguments: the predicate
    ``pretrain_gpt.py`` and ``pretrain_hybrid.py`` pass, and the default of ``build_gpt_datasets``.
    The arguments and the transformer config are read at call time from
    ``megatron.training.global_vars`` and ``megatron.training.arguments``, which is where tests
    patch them. ``pg_collection`` holds the process groups of the run (see
    ``is_dataset_built_on_rank_explicit``); omitting it is deprecated."""
    from megatron.training.arguments import core_transformer_config_from_args
    from megatron.training.global_vars import get_args

    args = get_args()
    config = core_transformer_config_from_args(args)
    return is_dataset_built_on_rank_explicit(
        vp_stage,
        is_packed_sequence,
        sequence_packing_scheduler=args.sequence_packing_scheduler,
        pipeline_model_parallel_layout=config.pipeline_model_parallel_layout,
        mtp_num_layers=config.mtp_num_layers,
        vp_size=config.virtual_pipeline_model_parallel_size or 1,
        pg_collection=pg_collection,
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
    dataset_type, config, is_packed_sequence = data_config.builder_inputs()

    print_rank_0("> building train, validation, and test datasets for GPT ...")

    train_ds, valid_ds, test_ds = BlendedMegatronDatasetBuilder(
        dataset_type,
        train_val_test_num_samples,
        partial(built_on_rank, vp_stage=vp_stage, is_packed_sequence=is_packed_sequence),
        config,
    ).build()

    print_rank_0("> finished creating GPT datasets ...")

    return train_ds, valid_ds, test_ds
