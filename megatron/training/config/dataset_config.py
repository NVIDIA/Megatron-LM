# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""The ``data`` section of ``PretrainConfigContainer``: Megatron Core's GPT dataset config, the loader
settings and the dataset class.

``DataConfig.dataset`` is ``megatron.core.datasets.gpt_dataset.GPTDatasetConfig`` itself (or its FIM
subclass), the object ``BlendedMegatronDatasetBuilder`` consumes, constructed from the arguments or
a YAML file before the tokenizer exists: Core checks the blend and split at construction and keeps
the checks that need a tokenizer in ``validate()``, which ``DataConfig.bind_tokenizer`` runs once the
runtime services bind the run's tokenizer (the builder runs it again before building).
``random_seed`` is Core's required constructor argument, filled from ``--seed`` by the factory and
saved with the section. Six Core fields are owned by other container sections (the parallel layout,
the validation flags): the container writes them at ``finalize()`` (``CONTAINER_WRITTEN_FIELDS``)
and they are not serialized with the section.

``DataConfig`` groups the dataset config, the ``DataloaderConfig`` and ``dataset_type``, the dataset
class the builder instantiates (``GPTDataset``, ``MockGPTDataset``, ``GPTFIMDataset``, ...). One rule
serves every GPT entry point: the args-to-config factory forwards every data argument, the class
follows the family flags, and the provider hands ``cfg.data.dataset`` to the builder unchanged.
Entry points with their own data pipeline carry a loader-only section (``dataset is None``) and
nothing here runs for them.
"""

from dataclasses import dataclass, field, fields
from typing import Any

from megatron.core.datasets.gpt_dataset import GPTDatasetConfig
from megatron.core.datasets.gpt_fim_dataset import GPTFIMDatasetConfig

DATALOADER_TYPES = ("single", "cyclic", "external")
DATASET_FAMILIES = ("gpt", "fim", "varlen", "sft")

# Fields of the dataset config that other container sections own: the parallel layout (the model
# config) and the validation flags (the validation config). PretrainConfigContainer writes them at
# finalize(), the container-oriented factory leaves them unset and the section serializes without
# them: they are recorded once, in their owning sections.
CONTAINER_WRITTEN_FIELDS = (
    "data_parallel_size",
    "context_parallel_size",
    "sequence_parallel_size",
    "hybrid_context_parallel",
    "multiple_validation_sets",
    "full_validation",
)


@dataclass(kw_only=True)
class DataloaderConfig:
    """Settings for building torch DataLoaders on top of a dataset."""

    dataloader_type: str | None = None
    """Single pass vs multiple pass data loader. 'external' passes the dataset through unchanged.
    Unset selects 'single'."""

    num_workers: int = 2
    """Dataloader number of workers."""

    data_sharding: bool = True
    """Shard the dataset per data-parallel rank before shuffling (cyclic sampler only)."""

    pin_memory: bool = True
    """Pin host memory in the torch DataLoader. Declared only for now: the training loop keeps its
    built-in value until the loader consumers read this config."""

    persistent_workers: bool = True
    """Keep DataLoader workers alive between epochs. Forced to False when num_workers == 0. Declared
    only for now: the training loop keeps its built-in value until the loader consumers read it."""

    drop_last: bool = True
    """Drop the last incomplete micro-batch. Declared only for now: the training loop keeps its
    built-in value until the loader consumers read this config."""

    def validate(self) -> None:
        """Check the loader settings without mutation."""
        if self.dataloader_type is not None and self.dataloader_type not in DATALOADER_TYPES:
            raise ValueError(
                f"dataloader_type must be one of {DATALOADER_TYPES}, got {self.dataloader_type!r}."
            )
        if self.num_workers < 0:
            raise ValueError(f"num_workers must be non-negative, got {self.num_workers}.")

    def finalize(self) -> None:
        """Apply the loader defaults (idempotent): unset type -> 'single', no workers -> no
        persistent workers."""
        if self.dataloader_type is None:
            self.dataloader_type = "single"
        if self.num_workers == 0 and self.persistent_workers:
            self.persistent_workers = False


def _dataset_class_table() -> dict[str, tuple[type, type | None]]:
    """Family -> (dataset class, mock dataset class). Imported lazily: the dataset modules are not
    part of the config import path."""
    from megatron.core.datasets.gpt_dataset import GPTDataset, MockGPTDataset
    from megatron.core.datasets.gpt_fim_dataset import GPTFIMDataset
    from megatron.training.datasets.sft_dataset import MockSFTDataset, SFTDataset
    from megatron.training.datasets.varlen_dataset import MockVarlenDataset, VarlenDataset

    return {
        "gpt": (GPTDataset, MockGPTDataset),
        "fim": (GPTFIMDataset, None),
        "varlen": (VarlenDataset, MockVarlenDataset),
        "sft": (SFTDataset, MockSFTDataset),
    }


def dataset_class_for(family: str, mock: bool) -> type | None:
    """The dataset class of a family, or its mock class (``None`` when the family has none)."""
    if family not in DATASET_FAMILIES:
        raise ValueError(f"dataset family must be one of {DATASET_FAMILIES}, got {family!r}.")
    real, mock_class = _dataset_class_table()[family]
    return mock_class if mock else real


def classify_dataset_class(dataset_type: Any) -> tuple[str, bool]:
    """``(family, is_mock)`` of one of the seven supported dataset classes."""
    for family, (real, mock_class) in _dataset_class_table().items():
        if dataset_type is real:
            return family, False
        if mock_class is not None and dataset_type is mock_class:
            return family, True
    known = sorted(
        cls.__name__ for pair in _dataset_class_table().values() for cls in pair if cls is not None
    )
    raise ValueError(
        f"dataset_type must be one of the dataset classes {known} (in YAML: "
        "{_target_: <module>.<class>, _call_: false}), got "
        f"{getattr(dataset_type, '__name__', dataset_type)!r}."
    )


def has_data_source(dataset: GPTDatasetConfig) -> bool:
    """Whether a blend is declared: the inverse of Megatron Core's ``mock``."""
    return not dataset.mock


def is_packed_sequence(dataset_type: type, dataset: GPTDatasetConfig) -> bool:
    """Packed (THD) samples: built on every pipeline stage, and get_batch expects cu_seqlens."""
    family, _mock = classify_dataset_class(dataset_type)
    if family == "sft":
        return True
    if family == "varlen":
        return not dataset.varlen_sbhd_validation
    return False


def resolve_dataset_class(
    dataset: GPTDatasetConfig | None, dataset_type: type | None
) -> type | None:
    """The dataset class for a section, checked against the config.

    Read-only: ``DataConfig.validate()`` calls it and ``DataConfig.finalize()`` commits its result.
    With no class given, a ``GPTFIMDatasetConfig`` with a source resolves to ``GPTFIMDataset``, a
    plain config with a source to ``GPTDataset`` and a plain config without one to
    ``MockGPTDataset``; sft and varlen share the plain config class and are always explicit. An
    explicit class must match the config (mock without a source, real with one, the FIM class with
    the FIM config); nothing is silently rewritten.
    """
    if dataset is None:
        if dataset_type is not None:
            raise ValueError("dataset_type needs a dataset: cfg.data.dataset is None.")
        return None
    source = has_data_source(dataset)
    config_family = "fim" if isinstance(dataset, GPTFIMDatasetConfig) else None
    if dataset_type is None:
        family = config_family or "gpt"
        if source:
            return dataset_class_for(family, mock=False)
        mock_class = dataset_class_for(family, mock=True)
        if mock_class is None:
            raise ValueError(
                f"No data source is declared and there is no mock dataset for the {family} "
                "family; declare blend or blend_per_split."
            )
        return mock_class
    family, mock = classify_dataset_class(dataset_type)
    real, mock_class = _dataset_class_table()[family]
    if mock and source:
        raise ValueError(
            f"{dataset_type.__name__} is a mock dataset and takes no data source; drop blend / "
            f"blend_per_split or select {real.__name__}."
        )
    if not mock and not source:
        if mock_class is not None:
            hint = f"select {mock_class.__name__} for mock data"
        else:
            hint = f"there is no mock dataset for the {family} family"
        raise ValueError(
            f"{dataset_type.__name__} needs a data source (blend or blend_per_split); {hint}."
        )
    if (family == "fim") != (config_family == "fim"):
        raise ValueError(
            "GPTFIMDataset and GPTFIMDatasetConfig go together: the FIM fields live on that config "
            f"class and only GPTFIMDataset reads them (got {dataset_type.__name__} with "
            f"{type(dataset).__name__})."
        )
    if dataset.varlen_sbhd_validation and family == "varlen" and mock:
        raise ValueError("varlen_sbhd_validation requires real data (VarlenDataset, not its mock).")
    return dataset_type


@dataclass(kw_only=True)
class DataConfig:
    """``PretrainConfigContainer.data``: the dataset config, the loader settings, the dataset class.

    ``dataset is None`` is the loader-only section of entry points with their own data pipeline;
    nothing dataset-related is validated, bound or built for it.
    """

    dataset: GPTDatasetConfig | None = None
    """Megatron Core's GPT dataset config (or ``GPTFIMDatasetConfig``), handed to
    ``BlendedMegatronDatasetBuilder`` as is."""

    dataloader: DataloaderConfig = field(default_factory=DataloaderConfig)
    """Loader settings."""

    dataset_type: type | None = None
    """The dataset class the builder instantiates (``GPTDataset``, ``MockGPTDataset``,
    ``GPTFIMDataset``, ``SFTDataset``, ``MockSFTDataset``, ``VarlenDataset``, ``MockVarlenDataset``).
    Resolved at ``finalize()`` when unset. In YAML: ``{_target_: <module>.<class>, _call_: false}``."""

    def validate(self) -> None:
        """Check the loader settings and the class / config agreement; no mutation.

        Megatron Core checks the dataset config itself: the blend and split rules at construction,
        the rest in ``validate()``, which needs the tokenizer and runs in ``bind_tokenizer()``.
        """
        self.dataloader.validate()
        resolve_dataset_class(self.dataset, self.dataset_type)

    def finalize(self) -> None:
        """Apply the loader defaults and commit the resolved dataset class (idempotent)."""
        self.dataloader.finalize()
        if self.dataset is not None:
            self.dataset_type = resolve_dataset_class(self.dataset, self.dataset_type)

    def bind_tokenizer(self, tokenizer: Any) -> None:
        """Bind the run's tokenizer to the dataset config and run Megatron Core's ``validate()``.

        The step between ``finalize()`` and the first build: ``initialize_runtime_services`` does it
        once the tokenizer exists. Idempotent for the same tokenizer; another tokenizer is an error
        (the datasets built from this config share it by reference: build a new config for a
        differently configured run). Nothing happens for a loader-only section.
        """
        if self.dataset is None:
            return
        if tokenizer is None:
            raise ValueError("bind_tokenizer() needs a tokenizer.")
        if self.dataset.tokenizer is None:
            self.dataset.tokenizer = tokenizer
        elif self.dataset.tokenizer is not tokenizer:
            raise RuntimeError(
                "cfg.data.dataset is already bound to another tokenizer; build a new config for a "
                "differently configured run."
            )
        self.dataset.validate()

    def dataset_build(self) -> tuple[type, GPTDatasetConfig, bool]:
        """``(dataset class, dataset config with its tokenizer, is_packed_sequence)`` for the
        builder."""
        if self.dataset is None or self.dataset_type is None:
            raise ValueError(
                "cfg.data carries no dataset to build: construct the container with "
                "data_cfg=gpt_data_config_from_args(args) and finalize it."
            )
        if self.dataset.tokenizer is None:
            raise RuntimeError(
                "cfg.data.dataset has no tokenizer bound: initialize_megatron() with "
                "build_tokenizer=True binds the run's tokenizer, or call "
                "cfg.data.bind_tokenizer(tokenizer)."
            )
        return self.dataset_type, self.dataset, is_packed_sequence(self.dataset_type, self.dataset)

    def to_cfg_dict(self) -> dict[str, Any]:
        """Serialize the section.

        The dataset config is written with every constructor field except the tokenizer and the
        fields other sections own (``CONTAINER_WRITTEN_FIELDS``, recorded once, in their owning
        sections); Megatron Core's derived values (``mock``, ``split_matrix``, ``token_dtype_code``)
        are read-only properties, not fields, and are never written.
        """
        from megatron.training.config.container import ConfigContainerBase

        convert = ConfigContainerBase._convert_value_to_dict
        dataset = None
        if self.dataset is not None:
            cls = type(self.dataset)
            dataset = {"_target_": f"{cls.__module__}.{cls.__qualname__}"}
            for f in fields(self.dataset):
                if not f.init or f.name == "tokenizer" or f.name in CONTAINER_WRITTEN_FIELDS:
                    continue
                dataset[f.name] = convert(getattr(self.dataset, f.name))
        return {
            "_target_": f"{type(self).__module__}.{type(self).__qualname__}",
            "dataset": dataset,
            "dataloader": convert(self.dataloader),
            "dataset_type": convert(self.dataset_type),
        }
