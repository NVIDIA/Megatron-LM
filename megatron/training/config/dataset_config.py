# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""The ``data`` section of ``PretrainConfigContainer``: one dataset config, the loader settings and
the dataset class.

``GPTDatasetConfig`` here *is* Megatron Core's GPT dataset config: the same fields, consumed by
``BlendedMegatronDatasetBuilder`` as is. It defers Core's ``__post_init__``, which needs a tokenizer,
until ``prepare(tokenizer)`` runs it in full, once; the container is built from the arguments before
the tokenizer exists (Megatron-Bridge defers it the same way). Its one redeclared field is
``random_seed``, which Core requires at construction and which the container supplies later. Seven
fields are *mirrors* of values other container sections own (the seed, the parallel layout, the
validation flags): the container writes them at ``finalize()`` and they are not serialized with the
section.

``DataConfig`` groups the dataset config, the ``DataloaderConfig`` and ``dataset_type``, the dataset
class the builder instantiates (``GPTDataset``, ``MockGPTDataset``, ``GPTFIMDataset``, ...). One rule
serves every GPT entry point: the args-to-config factory forwards every data argument, the class
follows the family flags, and the provider hands ``cfg.data.dataset`` to the builder unchanged.
Entry points with their own data pipeline carry a loader-only section (``dataset is None``) and
nothing here runs for them.
"""

from dataclasses import MISSING, dataclass, field, fields
from typing import Any

from megatron.core.datasets.gpt_dataset import GPTDatasetConfig as MCoreGPTDatasetConfig
from megatron.training.datasets.fim_dataset import GPTFIMDatasetConfig as RuntimeGPTFIMDatasetConfig

DATALOADER_TYPES = ("single", "cyclic", "external")
DATASET_FAMILIES = ("gpt", "fim", "varlen", "sft")

# Core fields whose value another container section owns. The container writes them at finalize()
# (see PretrainConfigContainer), they are left unset by the container-oriented factory, and they
# are not serialized with the dataset section.
MIRROR_FIELDS = (
    "random_seed",
    "data_parallel_size",
    "context_parallel_size",
    "sequence_parallel_size",
    "hybrid_context_parallel",
    "multiple_validation_sets",
    "full_validation",
)
# The mirrors derived from the model layout: the container writes them only with a model config.
LAYOUT_MIRRORS = (
    "data_parallel_size",
    "context_parallel_size",
    "sequence_parallel_size",
    "hybrid_context_parallel",
)
# Megatron Core's own defaults for the mirrors. A layout mirror at its Core default (1, 0, False,
# None for context_parallel_size) carries no explicit value: the container writes it when it can
# derive it, and a run without a model config keeps it. random_seed has no Core default.
_CORE_MIRROR_DEFAULTS = {
    f.name: f.default
    for f in fields(MCoreGPTDatasetConfig)
    if f.name in MIRROR_FIELDS and f.default is not MISSING
}
_FIM_TOKEN_KEYS = ("prefix", "middle", "suffix", "pad", "eod")


def mirror_is_unset(dataset: MCoreGPTDatasetConfig, name: str) -> bool:
    """Whether mirror ``name`` carries no explicit value on ``dataset``: ``None``, or Megatron
    Core's default for a layout mirror (what the field holds when nothing writes it)."""
    value = getattr(dataset, name)
    return value is None or value == _CORE_MIRROR_DEFAULTS.get(name)


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


@dataclass(kw_only=True)
class GPTDatasetConfig(MCoreGPTDatasetConfig):
    """Megatron Core's GPT dataset config, constructed before the tokenizer exists.

    Same fields as ``megatron.core.datasets.gpt_dataset.GPTDatasetConfig`` (imported as
    ``MCoreGPTDatasetConfig`` where both are visible), consumed by ``BlendedMegatronDatasetBuilder``
    as is. Core's ``__post_init__`` asserts a tokenizer and derives ``split_matrix``, ``mock`` and
    ``token_dtype_code``; here it is deferred: ``prepare(tokenizer)`` binds the tokenizer and runs
    it in full, once. ``validate()`` gives earlier, tokenizer-free errors for the same rules.

    Core's invariant holds: an object holding a tokenizer has run Core's post-init (``is_prepared``).
    A tokenizer given to the constructor is bound through ``prepare()`` right away, so a config
    built the Core way behaves as Core's; without one the object waits for ``prepare()``.

    The one redeclared field is ``random_seed``, which Core requires at construction: ``None``
    means "not set", so the section can be built, and reloaded, before the container writes the
    seed at ``finalize()``. The other mirrors keep Core's fields and defaults; a layout mirror at
    its Core default carries no explicit value (``mirror_is_unset``). A prepared object is shared
    by reference with every dataset built from it and is not edited afterwards: build a new object
    for a differently configured run.
    """

    random_seed: int | None = None
    """Seed for dataset shuffling. Mirror of ``cfg.rng.seed``, written by the container."""

    def __post_init__(self) -> None:
        """Deferred: Megatron Core's post-init runs in ``prepare()``. A tokenizer passed to the
        constructor is bound through it now."""
        tokenizer, self.tokenizer = self.tokenizer, None
        if tokenizer is not None:
            self.prepare(tokenizer)

    @property
    def is_prepared(self) -> bool:
        """Whether ``prepare()`` ran: the tokenizer is bound and Core's derived fields are set."""
        return self.tokenizer is not None

    def validate(self) -> None:
        """Tokenizer-free checks, as ``ValueError``s; no files, no mutation.

        These restate the rules Megatron Core asserts in ``__post_init__`` (which ``prepare()``
        runs, and which stays authoritative) so a bad section fails at container validation.
        """
        if not isinstance(self.sequence_length, int) or self.sequence_length <= 0:
            raise ValueError(
                f"sequence_length must be a positive integer, got {self.sequence_length!r}."
            )
        for name in ("reset_position_ids", "reset_attention_mask", "eod_mask_loss"):
            if getattr(self, name) is None:
                raise ValueError(f"{name} must be set (True or False).")
        if self.num_dataset_builder_threads <= 0:
            raise ValueError("num_dataset_builder_threads must be positive.")
        if self.fast_cache_load and self.path_to_cache is None:
            raise ValueError("fast_cache_load requires path_to_cache (--data-cache-path).")
        if self.fast_cache_load and self.blend is not None:
            raise ValueError(
                "fast_cache_load cannot be used with blend (--data-path); declare blend_per_split."
            )
        if self.defer_npy_index_mmap and self.path_to_cache is None:
            raise ValueError("defer_npy_index_mmap requires path_to_cache (--data-cache-path).")
        if self.varlen_sbhd_validation and self.hybrid_context_parallel:
            raise ValueError("varlen_sbhd_validation is incompatible with hybrid_context_parallel.")
        self._validate_sources()

    def _validate_sources(self) -> None:
        """Megatron Core's blend / split rules, as ``ValueError``s."""
        if self.blend_per_split is not None and any(self.blend_per_split):
            if self.blend is not None:
                raise ValueError("blend and blend_per_split are incompatible.")
            if self.split is not None:
                raise ValueError("split and blend_per_split are incompatible.")
            if len(self.blend_per_split) != 3:
                raise ValueError("blend_per_split must contain 3 blends (train, valid, test).")
            for item in self.blend_per_split:
                _validate_blend(item, "blend_per_split")
        elif self.blend is not None:
            _validate_blend(self.blend, "blend")
            if self.split is None:
                raise ValueError("split must be provided when blend is set.")

    def prepare(self, tokenizer: Any) -> None:
        """Bind the tokenizer and run Megatron Core's ``__post_init__`` in full, once.

        Idempotent for the same tokenizer; a different tokenizer is an error (build a new object
        for a different run). ``random_seed`` must be set (the container writes it from
        ``cfg.rng.seed`` at ``finalize()``; standalone callers set it explicitly). When Core's
        post-init fails the tokenizer stays unbound: the object is not prepared.
        """
        if tokenizer is None:
            raise ValueError("prepare() needs a tokenizer.")
        if self.tokenizer is not None:
            if tokenizer is self.tokenizer:
                return
            raise RuntimeError(
                "This dataset config is already prepared with another tokenizer; build a new "
                "config for a differently configured run."
            )
        if self.random_seed is None:
            raise ValueError(
                "random_seed is not set: the container writes it from cfg.rng.seed at finalize(); "
                "a standalone config sets it explicitly."
            )
        self.tokenizer = tokenizer
        try:
            MCoreGPTDatasetConfig.__post_init__(self)
        except Exception:
            self.tokenizer = None
            raise

    def to_cfg_dict(self) -> dict[str, Any]:
        """Serialize the section: every constructor field except the tokenizer and the mirrors.

        ``init=False`` fields (``split_matrix``, ``mock``, ``token_dtype_code``) are derived by
        ``prepare()`` and never written; the mirrors are recorded once, in their owning sections.
        """
        from megatron.training.config.container import ConfigContainerBase

        out: dict[str, Any] = {"_target_": f"{type(self).__module__}.{type(self).__qualname__}"}
        for f in fields(self):
            if not f.init or f.name == "tokenizer" or f.name in MIRROR_FIELDS:
                continue
            out[f.name] = ConfigContainerBase._convert_value_to_dict(getattr(self, f.name))
        return out


def _validate_blend(blend: Any, name: str) -> None:
    if blend is None:
        return
    if len(blend) != 2:
        raise ValueError(f"{name} must be (prefixes, weights), got {blend!r}.")
    prefixes, weights = blend
    if weights is not None and len(prefixes) != len(weights):
        raise ValueError(f"{name} prefixes and weights must be equal in number.")


@dataclass(kw_only=True)
class GPTFIMDatasetConfig(GPTDatasetConfig, RuntimeGPTFIMDatasetConfig):
    """``GPTDatasetConfig`` with the six fill-in-the-middle fields of the runtime
    ``megatron.training.datasets.fim_dataset.GPTFIMDatasetConfig`` (``fim_rate``, ``fim_spm_rate``,
    ``fim_extra_tokens``, ``fim_split_sample``, ``fim_fragment_rate``, ``fim_no_prefix``), which
    ``GPTFIMDataset`` reads. On the CLI the five ``--fim-*-token`` flags are folded into
    ``fim_extra_tokens``."""

    def validate(self) -> None:
        """The shared checks, then the FIM checks ``validate_args`` applies to ``--fim-data``."""
        super().validate()
        if not self.fim_rate:
            raise ValueError("fim_rate should be specified (--fim-rate).")
        if not self.fim_spm_rate:
            raise ValueError("fim_spm_rate should be specified (--fim-spm-rate).")
        tokens = self.fim_extra_tokens
        if tokens is None or any(tokens.get(key) is None for key in _FIM_TOKEN_KEYS):
            raise ValueError(
                "FIM extra tokens should be specified: fim_extra_tokens needs the keys "
                f"{_FIM_TOKEN_KEYS}."
            )


def _dataset_class_table() -> dict[str, tuple[type, type | None]]:
    """Family -> (dataset class, mock dataset class). Imported lazily: the dataset modules are not
    part of the config import path."""
    from megatron.core.datasets.gpt_dataset import GPTDataset, MockGPTDataset
    from megatron.training.datasets.fim_dataset import GPTFIMDataset
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


def has_data_source(dataset: MCoreGPTDatasetConfig) -> bool:
    """Whether a blend is declared (Megatron Core's rule for ``mock``, inverted)."""
    if dataset.blend is not None:
        return True
    return dataset.blend_per_split is not None and any(dataset.blend_per_split)


def is_packed_sequence(dataset_type: type, dataset: MCoreGPTDatasetConfig) -> bool:
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
    """The one dataset config, handed to ``BlendedMegatronDatasetBuilder`` as is."""

    dataloader: DataloaderConfig = field(default_factory=DataloaderConfig)
    """Loader settings."""

    dataset_type: type | None = None
    """The dataset class the builder instantiates (``GPTDataset``, ``MockGPTDataset``,
    ``GPTFIMDataset``, ``SFTDataset``, ``MockSFTDataset``, ``VarlenDataset``, ``MockVarlenDataset``).
    Resolved at ``finalize()`` when unset. In YAML: ``{_target_: <module>.<class>, _call_: false}``."""

    def validate(self) -> None:
        """Check the loader settings, the dataset config and the class / config agreement; no
        mutation."""
        self.dataloader.validate()
        if self.dataset is not None:
            self.dataset.validate()
        resolve_dataset_class(self.dataset, self.dataset_type)

    def finalize(self) -> None:
        """Apply the loader defaults and commit the resolved dataset class (idempotent)."""
        self.dataloader.finalize()
        if self.dataset is not None:
            self.dataset_type = resolve_dataset_class(self.dataset, self.dataset_type)

    def dataset_build(self) -> tuple[type, GPTDatasetConfig, bool]:
        """``(dataset class, prepared dataset config, is_packed_sequence)`` for the builder."""
        if self.dataset is None or self.dataset_type is None:
            raise ValueError(
                "cfg.data carries no dataset to build: construct the container with "
                "data_cfg=gpt_data_config_from_args(args) and finalize it."
            )
        if not self.dataset.is_prepared:
            raise RuntimeError(
                "cfg.data.dataset is not prepared (no tokenizer bound): initialize_megatron() with "
                "build_tokenizer=True prepares it, or call cfg.data.dataset.prepare(tokenizer)."
            )
        return self.dataset_type, self.dataset, is_packed_sequence(self.dataset_type, self.dataset)
