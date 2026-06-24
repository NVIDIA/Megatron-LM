# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Globals-free train/valid/test data-loader and data-iterator construction.

This module hosts the data-loader orchestration that was historically tied to
``get_args()`` / ``mpu`` globals (see ``megatron.training.training`` and
``megatron.training.datasets.data_samplers``). Every function here is driven
purely by:

  * a configuration container (``cfg``) exposing ``cfg.train``,
    ``cfg.validation`` and ``cfg.dataset`` -- e.g.
    ``megatron.training.config.container.PretrainConfigContainer``, or any
    structurally-compatible container (Megatron-Bridge's ``ConfigContainer``
    works by duck typing, which is what lets Bridge import and reuse these
    functions instead of maintaining its own copy);
  * an explicit data-parallel ``torch.distributed.ProcessGroup`` (``dp_group``),
    instead of reading ``mpu.get_data_parallel_*``;
  * explicit ``consumed_*_samples`` integers for resumption, instead of reading
    ``args.consumed_*``.

The functions **return** the ``(do_train, do_valid, do_test)`` flags rather than
writing them onto a global ``args`` object, so the caller owns all mutable
training state. This keeps the module free of any dependency on a specific
``TrainState`` type (Megatron-LM has none; Megatron-Bridge has its own).
"""

from typing import Any, Callable, Iterable, Iterator, Optional

import torch
from torch.utils.data import DataLoader, Dataset

from megatron.core.rerun_state_machine import RerunDataIterator
from megatron.training.datasets.data_samplers import (
    MegatronGlobalBatchSampler,
    MegatronPretrainingRandomSampler,
    MegatronPretrainingSampler,
)
from megatron.training.dist_signal_handler import DistributedSignalHandler
from megatron.training.utils import print_rank_0

# Single source of truth for blend parsing lives in common_utils; re-export it so
# callers (incl. Megatron-Bridge) can import it from here.
from megatron.training.utils.common_utils import get_blend_and_blend_per_split


def cyclic_iter(iterable: Iterable) -> Iterator:
    """Create an infinite iterator from a finite iterable.

    Re-creates the iterator each pass so the iterable is replayed. Raises if the
    iterable yields nothing on a pass (e.g. an empty validation dataloader), which
    would otherwise spin forever.
    """
    while True:
        iterator = iter(iterable)
        count = 0
        for x in iterator:
            count += 1
            yield x
        if count == 0:
            raise RuntimeError(
                "cyclic_iter: iterable produced no data. "
                "This may indicate the validation dataloader is empty or eval_iters is "
                "incorrectly set. Check that your validation dataset has data and that the "
                "dataloader is properly configured."
            )


def build_pretraining_data_loader(
    dataset: Optional[Dataset],
    consumed_samples: int,
    dataloader_type: str,
    micro_batch_size: int,
    num_workers: int,
    data_sharding: bool,
    *,
    data_parallel_rank: int = 0,
    data_parallel_size: int = 1,
    worker_init_fn: Optional[Callable] = None,
    collate_fn: Optional[Callable] = None,
    pin_memory: bool = True,
    persistent_workers: bool = False,
    drop_last: Optional[bool] = True,
    global_batch_size: Optional[int] = None,
) -> Optional[DataLoader]:
    """Build a torch DataLoader for pretraining, free of ``get_args()``/``mpu``.

    Selects the batch sampler from ``dataloader_type`` and wraps ``dataset`` in a
    ``torch.utils.data.DataLoader``. The data-parallel rank/size are passed
    explicitly so the same code serves any caller regardless of how it tracks
    its parallel state.

    Args:
        dataset: Dataset to load from. If None, returns None.
        consumed_samples: Samples already consumed (for resumption).
        dataloader_type: One of 'single', 'cyclic', 'batch', 'external'.
        micro_batch_size: Per-GPU batch size.
        num_workers: DataLoader worker subprocesses.
        data_sharding: Whether the random sampler shards before shuffling.
        data_parallel_rank: This rank within the data-parallel group.
        data_parallel_size: Size of the data-parallel group.
        worker_init_fn: Optional DataLoader worker init callable.
        collate_fn: Optional custom collate function.
        pin_memory: Whether to pin host memory.
        persistent_workers: Whether to keep workers alive across epochs.
        drop_last: Whether to drop the last incomplete batch.
        global_batch_size: Required for 'batch'; total batch across DP ranks.

    Returns:
        A DataLoader, the dataset itself when ``dataloader_type == 'external'``,
        or None when ``dataset`` is None.
    """
    if dataset is None:
        return None

    if dataloader_type == "single":
        batch_sampler = MegatronPretrainingSampler(
            total_samples=len(dataset),
            consumed_samples=consumed_samples,
            micro_batch_size=micro_batch_size,
            data_parallel_rank=data_parallel_rank,
            data_parallel_size=data_parallel_size,
            drop_last=drop_last,
        )
    elif dataloader_type == "cyclic":
        batch_sampler = MegatronPretrainingRandomSampler(
            dataset,
            total_samples=len(dataset),
            consumed_samples=consumed_samples,
            micro_batch_size=micro_batch_size,
            data_parallel_rank=data_parallel_rank,
            data_parallel_size=data_parallel_size,
            data_sharding=data_sharding,
        )
    elif dataloader_type == "batch":
        if global_batch_size is None:
            raise RuntimeError(
                "global_batch_size must be provided when using dataloader_type='batch'."
            )
        batch_sampler = MegatronGlobalBatchSampler(
            total_samples=len(dataset),
            consumed_samples=consumed_samples,
            micro_batch_size=micro_batch_size,
            global_batch_size=global_batch_size,
            data_parallel_rank=data_parallel_rank,
            data_parallel_size=data_parallel_size,
            drop_last=drop_last,
            pad_samples_to_global_batch_size=not drop_last,
        )
    elif dataloader_type == "external":
        # External dataloaders are passed through untouched.
        return dataset
    else:
        raise Exception("{} dataloader type is not supported.".format(dataloader_type))

    return DataLoader(
        dataset,
        batch_sampler=batch_sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_fn,
        persistent_workers=persistent_workers,
        worker_init_fn=worker_init_fn,
    )


def get_train_valid_test_num_samples(cfg: Any) -> tuple:
    """Compute (train, valid, test) target sample counts from the container.

    Uses ``cfg.train`` (train_samples or train_iters x global_batch_size) and
    ``cfg.validation`` (eval_interval / eval_iters / eval_global_batch_size) to
    derive the minimum number of samples each split must provide.

    Args:
        cfg: Configuration container exposing ``cfg.train`` and ``cfg.validation``.

    Returns:
        Tuple ``(train_samples, valid_samples, test_samples)``.
    """
    if cfg.train.train_samples is not None:
        train_samples = cfg.train.train_samples
    else:
        train_samples = cfg.train.train_iters * cfg.train.global_batch_size

    if cfg.validation.eval_interval:
        eval_iters = (cfg.train.train_iters // cfg.validation.eval_interval + 1) * cfg.validation.eval_iters
    else:
        eval_iters = 0
    test_iters = cfg.validation.eval_iters

    eval_gbs = (
        cfg.validation.eval_global_batch_size
        if cfg.validation.eval_global_batch_size is not None
        else cfg.train.global_batch_size
    )
    return (train_samples, eval_iters * eval_gbs, test_iters * eval_gbs)


def build_train_valid_test_datasets(cfg: Any, build_train_valid_test_datasets_provider: Callable):
    """Build train/valid/test datasets via a provider function.

    Args:
        cfg: Configuration container (for sizing and ``cfg.dataset``).
        build_train_valid_test_datasets_provider: Callable taking
            ``(train_val_test_num_samples, dataset_config)`` and returning the
            three datasets.

    Returns:
        Tuple ``(train_ds, valid_ds, test_ds)``.
    """
    train_valid_test_num_samples = get_train_valid_test_num_samples(cfg)
    print_rank_0(" > datasets target sizes (minimum size):")
    print_rank_0("    train:      {}".format(train_valid_test_num_samples[0]))
    print_rank_0("    validation: {}".format(train_valid_test_num_samples[1]))
    print_rank_0("    test:       {}".format(train_valid_test_num_samples[2]))
    return build_train_valid_test_datasets_provider(train_valid_test_num_samples, cfg.dataset)


def _build_dataloader_worker_init_fn(cfg: Any) -> Optional[Callable]:
    """Build the DataLoader ``worker_init_fn`` from the container, or None.

    Installs the distributed exit-signal handler in each worker process when
    ``cfg.train.exit_signal_handler_for_dataloader`` is set, so workers shut down
    cleanly on the configured signal. Returns None when not requested. Sourcing
    this from the container (instead of having each caller build it) keeps the
    signal-handling logic in one place.
    """
    if not getattr(cfg.train, "exit_signal_handler_for_dataloader", False):
        return None
    exit_signal = cfg.train.exit_signal

    def worker_init_fn(_):
        DistributedSignalHandler(exit_signal).__enter__()

    return worker_init_fn


def build_train_valid_test_data_loaders(
    cfg: Any,
    build_train_valid_test_datasets_provider: Callable,
    dp_group: torch.distributed.ProcessGroup,
    consumed_train_samples: int = 0,
    consumed_valid_samples: int = 0,
) -> tuple:
    """Build train/valid/test DataLoaders, free of ``get_args()``/``mpu``.

    Builds the datasets via the provider, then constructs DataLoaders with the
    appropriate samplers. The data-parallel rank/size are resolved from the
    explicit ``dp_group``. Resumption offsets come from the explicit
    ``consumed_*_samples`` arguments. The DataLoader ``worker_init_fn`` (exit
    signal handler) is built from ``cfg.train``. The ``do_train/valid/test`` flags
    are computed, broadcast from rank 0, and returned (not written to any global).

    Args:
        cfg: Configuration container (``cfg.train``, ``cfg.validation``, ``cfg.dataset``).
        build_train_valid_test_datasets_provider: Callable building the datasets.
        dp_group: The data-parallel process group.
        consumed_train_samples: Train samples already consumed (resumption).
        consumed_valid_samples: Valid samples already consumed (resumption).

    Returns:
        Tuple ``(train_dl, valid_dl, test_dl, do_train, do_valid, do_test)``.
    """
    (train_dataloader, valid_dataloader, test_dataloader) = (None, None, None)

    worker_init_fn = _build_dataloader_worker_init_fn(cfg)

    print_rank_0("> building train, validation, and test datasets ...")

    train_ds, valid_ds, test_ds = build_train_valid_test_datasets(
        cfg=cfg, build_train_valid_test_datasets_provider=build_train_valid_test_datasets_provider
    )

    # Guard: need at least one global batch of train samples.
    if (
        train_ds is not None
        and cfg.dataset.dataloader_type != "external"
        and len(train_ds) < cfg.train.global_batch_size
    ):
        raise RuntimeError(
            f"Not enough train samples for a single global batch: "
            f"train dataset size ({len(train_ds)}) < global batch size "
            f"({cfg.train.global_batch_size})."
        )

    dp_rank = torch.distributed.get_rank(group=dp_group)
    dp_size = torch.distributed.get_world_size(group=dp_group)

    def _collate(ds):
        return ds.collate_fn if hasattr(ds, "collate_fn") else None

    train_dataloader = build_pretraining_data_loader(
        train_ds,
        consumed_train_samples,
        cfg.dataset.dataloader_type,
        cfg.train.micro_batch_size,
        cfg.dataset.num_workers,
        cfg.dataset.data_sharding,
        data_parallel_rank=dp_rank,
        data_parallel_size=dp_size,
        worker_init_fn=worker_init_fn,
        collate_fn=_collate(train_ds),
        pin_memory=cfg.dataset.pin_memory,
        persistent_workers=cfg.dataset.persistent_workers,
        global_batch_size=cfg.train.global_batch_size,
    )

    eval_gbs = (
        cfg.validation.eval_global_batch_size
        if cfg.validation.eval_global_batch_size is not None
        else cfg.train.global_batch_size
    )
    eval_mbs = (
        cfg.validation.eval_micro_batch_size
        if cfg.validation.eval_micro_batch_size is not None
        else cfg.train.micro_batch_size
    )

    if cfg.validation.skip_train and cfg.validation.eval_iters > 0:
        valid_dataloader = build_pretraining_data_loader(
            valid_ds,
            0,
            cfg.dataset.dataloader_type,
            eval_mbs,
            cfg.dataset.num_workers,
            cfg.dataset.data_sharding,
            data_parallel_rank=dp_rank,
            data_parallel_size=dp_size,
            worker_init_fn=worker_init_fn,
            collate_fn=_collate(valid_ds),
            pin_memory=cfg.dataset.pin_memory,
            persistent_workers=cfg.dataset.persistent_workers,
            global_batch_size=eval_gbs,
        )
    elif cfg.validation.eval_iters > 0:
        # GPT datasets validate cyclically so eval_iters can exceed the valid set
        # size; build the loader with the matching (random/cyclic) sampler.
        val_dataloader_type = (
            "cyclic" if _is_gpt_dataset_config(cfg.dataset) else cfg.dataset.dataloader_type
        )
        valid_dataloader = build_pretraining_data_loader(
            valid_ds,
            consumed_valid_samples,
            val_dataloader_type,
            eval_mbs,
            cfg.dataset.num_workers,
            cfg.dataset.data_sharding,
            data_parallel_rank=dp_rank,
            data_parallel_size=dp_size,
            worker_init_fn=worker_init_fn,
            collate_fn=_collate(valid_ds),
            pin_memory=cfg.dataset.pin_memory,
            persistent_workers=cfg.dataset.persistent_workers,
            global_batch_size=eval_gbs,
        )

    if cfg.validation.eval_iters > 0:
        test_dataloader = build_pretraining_data_loader(
            test_ds,
            0,
            cfg.dataset.dataloader_type,
            eval_mbs,
            cfg.dataset.num_workers,
            cfg.dataset.data_sharding,
            data_parallel_rank=dp_rank,
            data_parallel_size=dp_size,
            worker_init_fn=worker_init_fn,
            collate_fn=_collate(test_ds),
            pin_memory=cfg.dataset.pin_memory,
            persistent_workers=cfg.dataset.persistent_workers,
            global_batch_size=eval_gbs,
        )

    do_train = train_dataloader is not None and cfg.train.train_iters > 0
    do_valid = valid_dataloader is not None and cfg.validation.eval_iters > 0
    do_test = test_dataloader is not None and cfg.validation.eval_iters > 0
    flags = torch.tensor([int(do_train), int(do_valid), int(do_test)], dtype=torch.long, device="cuda")
    torch.distributed.broadcast(flags, 0)

    return (
        train_dataloader,
        valid_dataloader,
        test_dataloader,
        bool(flags[0].item()),
        bool(flags[1].item()),
        bool(flags[2].item()),
    )


def wrap_loaders_in_iterators(
    cfg: Any,
    train_dataloader,
    valid_dataloader,
    test_dataloader,
) -> tuple:
    """Wrap already-built DataLoaders into ``RerunDataIterator`` iterators.

    Single source of truth for the iterator-wrapping rules, shared by Megatron-LM
    and Megatron-Bridge: cycle 'cyclic'/'batch' loaders via ``cyclic_iter``; cycle
    validation for GPT datasets (so ``eval_iters`` may exceed the valid set size);
    pass through 'external'. Takes already-built loaders so it works regardless of
    how they were produced (e.g. Bridge's MegatronMIMO path).

    Args:
        cfg: Configuration container (reads ``cfg.dataset.dataloader_type``).
        train_dataloader: Built train loader (or None).
        valid_dataloader: Built valid loader (or None).
        test_dataloader: Built test loader (or None).

    Returns:
        Tuple ``(train_it, valid_it, test_it)``.
    """
    dl_type = cfg.dataset.dataloader_type
    assert dl_type in ["single", "cyclic", "batch", "external"]

    def _get_iterator(dataloader_type, dataloader):
        """Return dataset iterator wrapped for rerun support."""
        if dataloader_type == "single":
            return RerunDataIterator(iter(dataloader))
        elif dataloader_type in ("cyclic", "batch"):
            return RerunDataIterator(iter(cyclic_iter(dataloader)))
        elif dataloader_type == "external":
            if isinstance(dataloader, list):
                return [RerunDataIterator(d) for d in dataloader]
            return RerunDataIterator(dataloader)
        else:
            raise RuntimeError("unexpected dataloader type")

    # GPT datasets always cycle validation so eval_iters may exceed valid size.
    is_gpt_dataset = _is_gpt_dataset_config(cfg.dataset)

    train_it = _get_iterator(dl_type, train_dataloader) if train_dataloader is not None else None
    if valid_dataloader is not None:
        valid_it = _get_iterator("cyclic" if is_gpt_dataset else dl_type, valid_dataloader)
    else:
        valid_it = None
    test_it = _get_iterator(dl_type, test_dataloader) if test_dataloader is not None else None

    return train_it, valid_it, test_it


def build_train_valid_test_data_iterators(
    cfg: Any,
    build_train_valid_test_datasets_provider: Callable,
    dp_group: torch.distributed.ProcessGroup,
    consumed_train_samples: int = 0,
    consumed_valid_samples: int = 0,
) -> tuple:
    """Build train/valid/test data iterators, free of ``get_args()``/``mpu``.

    Builds the DataLoaders, then wraps them via :func:`wrap_loaders_in_iterators`.

    Args:
        cfg: Configuration container.
        build_train_valid_test_datasets_provider: Callable building the datasets.
        dp_group: The data-parallel process group.
        consumed_train_samples: Train samples already consumed (resumption).
        consumed_valid_samples: Valid samples already consumed (resumption).

    Returns:
        Tuple ``(train_it, valid_it, test_it, do_train, do_valid, do_test)``.
    """
    (
        train_dataloader,
        valid_dataloader,
        test_dataloader,
        do_train,
        do_valid,
        do_test,
    ) = build_train_valid_test_data_loaders(
        cfg=cfg,
        build_train_valid_test_datasets_provider=build_train_valid_test_datasets_provider,
        dp_group=dp_group,
        consumed_train_samples=consumed_train_samples,
        consumed_valid_samples=consumed_valid_samples,
    )

    train_it, valid_it, test_it = wrap_loaders_in_iterators(
        cfg, train_dataloader, valid_dataloader, test_dataloader
    )
    return train_it, valid_it, test_it, do_train, do_valid, do_test


def _is_gpt_dataset_config(dataset_config: Any) -> bool:
    """Whether ``dataset_config`` is (a subclass of) the core GPT dataset config.

    Used to decide whether validation should iterate cyclically. Imported lazily
    so this module does not hard-depend on the GPT dataset definition.
    """
    try:
        from megatron.core.datasets.gpt_dataset import GPTDatasetConfig

        return isinstance(dataset_config, GPTDatasetConfig)
    except Exception:
        return False
