# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""The ``data`` section of ``PretrainConfigContainer``: Megatron Core's GPT dataset config, the loader
settings and the dataset class.

``DataConfig.dataset`` is ``megatron.core.datasets.gpt_dataset.GPTDatasetConfig`` itself (or its FIM
subclass), the object ``BlendedMegatronDatasetBuilder`` consumes, constructed from the arguments or
a YAML file before the tokenizer exists: nothing runs at construction, Core's checks that need no
tokenizer (``validate_without_tokenizer()``) run at ``cfg.validate()``, and the complete
``validate()``, which needs the tokenizer, runs in ``DataConfig.bind_tokenizer`` once the runtime
services bind the run's tokenizer (the builder runs it again before building).
``random_seed`` is Core's required constructor argument, filled from ``--seed`` by the factory and
saved with the section. Seven Core fields are owned by other container sections (the parallel
layout with its deprecated ``hybrid_context_parallel`` alias, the validation flags): the container
writes them at ``finalize()`` (``CONTAINER_WRITTEN_FIELDS``) and they are not serialized with the
section.

``DataConfig`` groups the dataset config, the ``DataloaderConfig`` and ``dataset_type``, the dataset
class the builder instantiates (``GPTDataset``, ``MockGPTDataset``, ``GPTFIMDataset``, ...). One rule
serves every GPT entry point: the args-to-config factory forwards every data argument, the class
follows the family flags, and the provider hands ``cfg.data.dataset`` to the builder unchanged.
The loop reads the section through ``batch_flags()`` and ``yields_unpacked_samples()`` (and the
data length through ``megatron.training.config.compat``), never through ``args``.
Entry points with their own data pipeline carry a loader-only section (``dataset is None``) and
nothing here runs for them.
"""

from dataclasses import dataclass, field, fields
from typing import Any

from megatron.core.datasets.gpt_dataset import GPTDataset, GPTDatasetConfig, MockGPTDataset
from megatron.core.datasets.gpt_fim_dataset import GPTFIMDataset, GPTFIMDatasetConfig
from megatron.training.datasets.sft_dataset import MockSFTDataset, SFTDataset
from megatron.training.datasets.varlen_dataset import MockVarlenDataset, VarlenDataset

DATALOADER_TYPES = ("single", "cyclic", "external")
# Family -> (dataset class, mock dataset class or None): the legacy family flags select the class
# and DataConfig.validate() checks the class against the config.
DATASET_CLASSES: dict[str, tuple[type, type | None]] = {
    "gpt": (GPTDataset, MockGPTDataset),
    "fim": (GPTFIMDataset, None),
    "varlen": (VarlenDataset, MockVarlenDataset),
    "sft": (SFTDataset, MockSFTDataset),
}
DATASET_FAMILIES = tuple(DATASET_CLASSES)

# Fields of the dataset config that other container sections own: the parallel layout (the model
# config) and the validation flags (the validation config). PretrainConfigContainer writes them at
# finalize(), the container-oriented factory leaves them unset and the section serializes without
# them: they are recorded once, in their owning sections. hybrid_context_parallel is the deprecated
# alias of dynamic_context_parallel (#7047): the container folds it into the canonical field and
# clears it, so it is neither serialized nor left set.
CONTAINER_WRITTEN_FIELDS = (
    "data_parallel_size",
    "context_parallel_size",
    "sequence_parallel_size",
    "dynamic_context_parallel",
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


def dataset_type_from_family(family: str, mock: bool) -> type | None:
    """The dataset class of a family, or its mock class (``None`` when the family has none); the
    inverse of ``dataset_family_from_type``."""
    if family not in DATASET_CLASSES:
        raise ValueError(f"dataset family must be one of {DATASET_FAMILIES}, got {family!r}.")
    real, mock_class = DATASET_CLASSES[family]
    return mock_class if mock else real


def dataset_family_from_type(dataset_type: Any) -> tuple[str, bool]:
    """``(family, is_mock)`` of one of the seven supported dataset classes; the inverse of
    ``dataset_type_from_family``."""
    for family, (real, mock_class) in DATASET_CLASSES.items():
        if dataset_type is real:
            return family, False
        if mock_class is not None and dataset_type is mock_class:
            return family, True
    known = sorted(
        cls.__name__ for pair in DATASET_CLASSES.values() for cls in pair if cls is not None
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
    family, _mock = dataset_family_from_type(dataset_type)
    if family == "sft":
        return True
    if family == "varlen":
        return not dataset.varlen_sbhd_validation
    return False


@dataclass(frozen=True)
class BatchFlags:
    """What ``get_batch``, ``dummy_train_step``, the loss reductions and the distillation hooks need
    to know about the samples: computed by ``DataConfig.batch_flags()`` from the dataset class and
    two fields of the dataset config, never stored (the section stays the single source)."""

    is_sft: bool
    """The SFT family exactly (``SFTDataset`` / ``MockSFTDataset``); the loss reductions key on it."""

    is_packed_sequence: bool
    """THD metadata (the SFT family, or varlen without SBHD validation): built on every pipeline
    stage, ``get_batch`` expects ``cu_seqlens``. Not the loader's collate decision, which is
    ``DataConfig.yields_unpacked_samples()``: SFT samples are packed but fixed-width."""

    inter_document_masking: bool
    """``GPTDatasetConfig.inter_document_masking`` (``--dataloader-inter-document-masking``)."""

    create_attention_mask: bool
    """``GPTDatasetConfig.create_attention_mask``: the dataset builds the attention mask (the
    default; ``--no-create-attention-mask-in-dataloader`` turns it off)."""

    @property
    def has_cu_seqlens(self) -> bool:
        """``get_batch``'s formula: SFT samples and inter-document masking carry ``cu_seqlens``."""
        return self.is_sft or self.inter_document_masking

    @property
    def use_per_sequence_balancing(self) -> bool:
        """``get_batch``'s formula: inter-document masking balances the loss per sequence, SFT
        does not."""
        return self.inter_document_masking and not self.is_sft


LOADER_ONLY_BATCH_FLAGS = BatchFlags(
    is_sft=False, is_packed_sequence=False, inter_document_masking=False, create_attention_mask=True
)
"""The plain-GPT values a section without a dataset reports (the parser defaults)."""


def resolve_dataset_class(
    dataset: GPTDatasetConfig | None, dataset_type: type | None
) -> type | None:
    """The dataset class the config entails, or the given one checked against the config.

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
            return dataset_type_from_family(family, mock=False)
        mock_class = dataset_type_from_family(family, mock=True)
        if mock_class is None:
            raise ValueError(
                f"No data source is declared and there is no mock dataset for the {family} "
                "family; declare blend or blend_per_split."
            )
        return mock_class
    family, mock = dataset_family_from_type(dataset_type)
    real, mock_class = DATASET_CLASSES[family]
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

        Megatron Core's checks that need no tokenizer (``validate_without_tokenizer()``: the blend /
        split rules, the GPT booleans, the FIM rates and tokens) run here too, so a YAML or Python
        section fails at ``cfg.validate()``, before runtime initialization; the checks that need the
        tokenizer run in ``bind_tokenizer()`` and in the builder. Inter-document masking with the
        dataloader attention mask is rejected here (the CLI normalizes the pair away).
        """
        self.dataloader.validate()
        if self.dataset is not None:
            self.dataset.validate_without_tokenizer()
            if self.dataset.inter_document_masking and self.dataset.create_attention_mask:
                raise ValueError(
                    "data.dataset.inter_document_masking=True needs create_attention_mask=False: "
                    "the dataset omits the attention mask under inter-document masking, and a "
                    "batch path told to expect one waits for a broadcast that never comes "
                    "(validate_args clears the flag for --dataloader-inter-document-masking; a "
                    "Python or YAML section sets it explicitly)."
                )
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

    def builder_inputs(self) -> tuple[type, GPTDatasetConfig, bool]:
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

    def batch_flags(self) -> BatchFlags:
        """The sample properties the batch path reads (``BatchFlags``), from the dataset class and
        the dataset config; read-only, valid before the tokenizer is bound, the same for a mock
        dataset and its real counterpart. A loader-only section gets the plain-GPT values
        (``LOADER_ONLY_BATCH_FLAGS``), so ``get_batch`` runs unchanged for an entry point whose
        datasets come from its own callback."""
        if self.dataset is None:
            return LOADER_ONLY_BATCH_FLAGS
        dataset_type = resolve_dataset_class(self.dataset, self.dataset_type)
        family, _mock = dataset_family_from_type(dataset_type)
        return BatchFlags(
            is_sft=family == "sft",
            is_packed_sequence=is_packed_sequence(dataset_type, self.dataset),
            inter_document_masking=bool(self.dataset.inter_document_masking),
            create_attention_mask=bool(self.dataset.create_attention_mask),
        )

    def yields_unpacked_samples(self) -> bool:
        """The dataset emits one variable-length sample dict per item for an upstream packing
        scheduler (the varlen family without SBHD validation), so the loader keeps the list as is
        (the identity collate). ``False`` for every other family and for a loader-only section:
        SFT samples are packed (``is_packed_sequence``) but fixed-width, and go through the
        default collate."""
        if self.dataset is None:
            return False
        family, _mock = dataset_family_from_type(
            resolve_dataset_class(self.dataset, self.dataset_type)
        )
        return family == "varlen" and not bool(self.dataset.varlen_sbhd_validation)

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
