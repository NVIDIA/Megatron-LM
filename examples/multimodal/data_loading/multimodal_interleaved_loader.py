# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Shared Energon construction and external-iterator state delegation for interleaved data."""

import hashlib
import importlib.metadata
import json
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import torch
from PIL import Image

from megatron.core.models.vision.encoder_registry import EncoderSpec
from megatron.energon import (
    Lazy,
    NoCachePool,
    StreamingPackingDataset,
    WorkerConfig,
    get_savable_loader,
    get_train_dataset,
    register_aux_filestore_protocol,
)
from megatron.energon.errors import reraise_exception

from .binidx_token_mask import IndexedTokenMaskReference
from .conversation_sample import ImageMedia
from .document_image_store import register_document_image_store
from .image_processing import DynamicResolutionImageTilingStrategy
from .indexed_jsonl_metadata import IndexedJsonlMetadataReference
from .multimodal_interleaved_checkpoint import (
    InterleavedRescaleProfile,
    build_rescale_profile,
    validate_rescale_restore,
)


def _tokenizer_identity(tokenizer) -> dict:
    """Fingerprint the actual fast tokenizer, including merges/normalization."""
    inner = getattr(tokenizer, "_tokenizer", tokenizer)
    backend = getattr(inner, "_hf_tokenizer", getattr(inner, "tokenizer", inner))
    fast = getattr(backend, "backend_tokenizer", None)
    if fast is None:
        raise ValueError("interleaved exact resume requires a serializable fast tokenizer backend")
    return {
        "wrapper": f"{type(tokenizer).__module__}.{type(tokenizer).__qualname__}",
        "backend_sha256": hashlib.sha256(fast.to_str().encode("utf-8")).hexdigest(),
        "tokenizers_version": importlib.metadata.version("tokenizers"),
        "transformers_version": importlib.metadata.version("transformers"),
        "gigatoken": bool(getattr(inner, "use_gigatoken", False)),
    }


@dataclass(frozen=True)
class InterleavedImageConfig:
    """Deterministic image policy, usable without constructing a vision encoder."""

    min_patches: int = 4
    max_patches: int = 4096
    patch_size: int | None = None
    encoder_spec: EncoderSpec | None = None

    def __post_init__(self) -> None:
        patch = self.patch_size
        if patch is None:
            if self.encoder_spec is None:
                raise ValueError(
                    "patch size is required: supply encoder metadata or an explicit patch_size"
                )
            patch = self.encoder_spec.patch_dim
            object.__setattr__(self, "patch_size", patch)
        if type(patch) is not int or patch <= 0:
            raise ValueError("patch_size must be a positive integer")
        if self.encoder_spec is None:
            raise ValueError(
                "encoder metadata is required: patch size alone does not define reduction or normalization"
            )
        if patch != self.encoder_spec.patch_dim:
            raise ValueError("patch_size disagrees with encoder metadata")
        reduction = self.encoder_spec.spatial_reduction
        if any(type(value) is not int for value in (self.min_patches, self.max_patches)) or not (
            reduction**2 <= self.min_patches <= self.max_patches
        ):
            raise ValueError(
                "image patch budgets must satisfy reduction^2 <= min_patches <= max_patches"
            )

    def checkpoint_contract(self):
        # Preserve previously qualified RADIO checkpoints, including their exact
        # three-key image contract. New image policies record encoding semantics.
        result = {
            "min_patches": self.min_patches,
            "max_patches": self.max_patches,
            "patch_size": self.patch_size,
        }
        spec = self.encoder_spec
        if spec is not None:
            from megatron.core.models.vision.encoder_registry import get_spec

            legacy = get_spec("radio")
            if not (
                spec.name == "radio"
                and spec.patch_dim == 16
                and spec.pixel_shuffle
                and not spec.has_native_spatial_merge
                and spec.pixel_mean == legacy.pixel_mean
                and spec.pixel_std == legacy.pixel_std
            ):
                result["encoder"] = {
                    "policy_version": 1,
                    "name": spec.name,
                    "spatial_merge_size": spec.spatial_merge_size,
                    "pixel_shuffle": spec.pixel_shuffle,
                    "pixel_mean": list(spec.pixel_mean),
                    "pixel_std": list(spec.pixel_std),
                    "image_bounds": [spec.default_img_h, spec.default_img_w],
                }
        return result

    def build_strategy(self):
        spec = self.encoder_spec
        return DynamicResolutionImageTilingStrategy(
            vision_model_type=spec.name,
            min_num_patches=self.min_patches,
            max_num_patches=self.max_patches,
            patch_size=self.patch_size,
            pixel_shuffle=spec.pixel_shuffle,
            spatial_merge_size=spec.spatial_merge_size,
            image_bounds=None if spec.name == "radio" else (spec.default_img_h, spec.default_img_w),
            pixel_mean=spec.pixel_mean,
            pixel_std=spec.pixel_std,
            get_num_embeddings=spec.num_spatial_embeddings,
        )


class InterleavedDataIterator:
    """Retain one live iterator and expose state to the existing checkpoint writer.

    Normal restore requires an exact provider contract. Explicit worker-preserving
    migration additionally validates the actual destination before native restore;
    it does not convert old Energon schemas or changed data/encoding policies.
    """

    def __init__(
        self,
        loader,
        contract: dict,
        *,
        checkpoint_enabled: bool = True,
        rescale_profile: InterleavedRescaleProfile | None = None,
    ) -> None:
        self.loader = loader
        self.contract = contract
        self.checkpoint_enabled = checkpoint_enabled
        self.rescale_profile = rescale_profile
        self._iterator = None
        self._closed = False

    def __iter__(self):
        return self

    def __next__(self):
        if self._closed:
            raise RuntimeError("interleaved dataloader is closed")
        if self._iterator is None:
            self._iterator = iter(self.loader)
        batch = next(self._iterator)
        batch.pop("__restore_key__", None)
        return batch

    def save_state(self):
        if not self.checkpoint_enabled:
            return None
        if self._closed:
            raise RuntimeError("cannot checkpoint a closed interleaved dataloader")
        state = self.loader.save_state_rank()
        if state is None:
            raise RuntimeError("interleaved dataloader returned no savable state")
        payload = {"format": 1, "contract": self.contract, "energon": state}
        if self.rescale_profile is not None:
            payload.update(format=2, rescale=asdict(self.rescale_profile))
        return payload

    def restore_state(self, state) -> None:
        if self._iterator is not None or self._closed:
            raise RuntimeError("restore interleaved state before creating its iterator")
        try:
            if (
                not isinstance(state, dict)
                or type(state.get("format")) is not int
                or state["format"] not in (1, 2)
            ):
                raise ValueError("unsupported interleaved dataloader checkpoint format")
            if state["format"] == 2:
                validate_rescale_restore(state, self.contract, self.rescale_profile)
            elif state.get("contract") != self.contract or "migration" in state:
                raise ValueError("interleaved dataloader recipe/token/worker/lane contract changed")
            if state.get("energon") is None:
                raise ValueError("interleaved dataloader checkpoint has no Energon state")
            self.loader.restore_state_rank(state["energon"])
        except BaseException:
            # A rejected resume must not later yield a silently fresh stream.
            self.close()
            raise

    def close(self) -> None:
        if not self._closed:
            self._closed = True
            try:
                close = getattr(self._iterator, "close", None)
                if close is not None:
                    close()
            finally:
                self._iterator = None
                close_loader = getattr(self.loader, "close", None)
                if close_loader is not None:
                    close_loader()


def materialize_document_images(pack, image_strategy):
    """Realize selected images without mutating the replayable metadata-only pack."""
    pixels = []
    for plan in pack.image_params:
        value = plan.media.value
        if isinstance(value, Lazy):
            value = value.get(sample=pack)
        if not isinstance(value, Image.Image):
            raise TypeError("interleaved image decoder must produce a PIL image")
        media = ImageMedia(value=value, metadata=plan.media.metadata)
        pixels.extend(image_strategy.apply_params(replace(plan, media=media), data_augment=False))
    return image_strategy.stack(pixels)


def restore_interleaved_iterator(iterator, args, *, lane: int, pipeline_parallel: bool = False):
    """Restore a new interleaved iterator strictly, closing it if resume is rejected."""
    from examples.multimodal.dataloader_resume import (
        is_initial_checkpoint_load_without_dataloader_state,
    )
    from megatron.training import get_train_state
    from megatron.training.checkpointing import get_checkpoint_name

    try:
        if args.load is None:
            return iterator
        iteration = get_train_state().iteration
        state_path = get_checkpoint_name(
            args.multimodal_interleaved_dataloader_checkpoint_root or args.load,
            iteration,
            tensor_rank=0,
            pipeline_rank=0,
            expert_parallel=False,
            expert_rank=0,
            pipeline_parallel=pipeline_parallel,
            basename=f"train_dataloader_dprank{lane:03d}.pt",
        )
        if Path(state_path).is_file():
            state = torch.load(state_path, map_location="cpu", weights_only=False)
            iterator.restore_state(state["dataloader_state_dict"])
        elif not is_initial_checkpoint_load_without_dataloader_state(args, iteration):
            raise FileNotFoundError(f"missing interleaved dataloader state: {state_path}")
    except BaseException:
        iterator.close()
        raise
    return iterator


def build_interleaved_iterator(
    recipe,
    encoder,
    image_config: InterleavedImageConfig,
    *,
    lane: int,
    lanes: int,
    seed: int,
    shuffle_buffer_size: int = 0,
    num_workers: int = 0,
    global_batch_size: int | None = None,
    checkpoint_enabled: bool = True,
    repeat: bool = True,
):
    """Construct an interleaved stream; distributed placement is the caller's job."""
    if (
        not 0 <= lane < lanes
        or shuffle_buffer_size < 0
        or type(num_workers) is not int
        or num_workers < 0
    ):
        raise ValueError("invalid interleaved data lane or shuffle buffer size")
    # Finite streams require the sampler exhaustion state to survive a checkpoint.
    if not repeat and "_source_exhausted" not in StreamingPackingDataset._savable_fields:
        raise RuntimeError(
            "finite interleaved loading requires Energon streaming exhaustion-state support"
        )
    register_document_image_store()
    register_aux_filestore_protocol(
        "indexed-jsonl", lambda path: IndexedJsonlMetadataReference(fs_path=path), override=True
    )
    register_aux_filestore_protocol(
        "indexed-token-mask", lambda path: IndexedTokenMaskReference(fs_path=path), override=True
    )
    recipe_path = Path(recipe).resolve(strict=True)
    # Hash bounded recipe text, never scan or hash the interleaved corpus. Referenced
    # datasets/manifests must remain immutable for exact fixed-config resume.
    recipe_hash = None
    if recipe_path.is_file() and recipe_path.suffix in (".yaml", ".yml"):
        if recipe_path.stat().st_size > 10 * 1024 * 1024:
            raise ValueError("interleaved recipe exceeds the bounded configuration size")
        recipe_hash = hashlib.sha256(recipe_path.read_bytes()).hexdigest()
    contract = {
        "recipe": str(recipe_path),
        "recipe_sha256": recipe_hash,
        "packing": asdict(encoder.config),
        "images": image_config.checkpoint_contract(),
        "tokenizer": _tokenizer_identity(encoder.tokenizer),
        "energon_version": importlib.metadata.version("megatron-energon"),
        "lane": lane,
        "lanes": lanes,
        "seed": seed,
        "workers": num_workers,
        "shuffle": shuffle_buffer_size,
        "repeat": repeat,
    }
    # Round-trip to primitives so state comparisons do not depend on live objects.
    contract = json.loads(json.dumps(contract, sort_keys=True))
    workers = WorkerConfig(
        rank=lane,
        world_size=lanes,
        num_workers=num_workers,
        seed_offset=seed,
        data_parallel_group=None,
        global_error_handler=reraise_exception,
        restore_error_handler=reraise_exception,
    )
    dataset = get_train_dataset(
        str(recipe_path),
        worker_config=workers,
        task_encoder=encoder,
        batch_size=1,
        packing_buffer_size="stream",
        shuffle_buffer_size=shuffle_buffer_size or None,
        virtual_epoch_length=0,
        max_samples_per_sequence=100,
        shuffle_over_epochs_multiplier=1,
        parallel_shard_iters=1,
        repeat=repeat,
    )
    loader = get_savable_loader(
        dataset,
        cache_pool=NoCachePool(),
        watchdog_timeout_seconds=300,
        watchdog_initial_timeout_seconds=300,
    )
    # Energon already resolves nested recipes, blend weights, tags and wrappers.
    # Reuse that identity rather than walking recipes or scanning their corpus.
    # Auxiliary contents and prepared manifests must still remain immutable.
    try:
        resolved = json.dumps(
            loader.config(), sort_keys=True, separators=(",", ":"), allow_nan=False
        )
        contract["resolved_pipeline_sha256"] = hashlib.sha256(resolved.encode("utf-8")).hexdigest()
        profile = (
            build_rescale_profile(contract, json.loads(resolved), global_batch_size)
            if num_workers > 0 and global_batch_size is not None
            else None
        )
        return InterleavedDataIterator(
            loader, contract, checkpoint_enabled=checkpoint_enabled, rescale_profile=profile
        )
    except BaseException:
        loader.close()
        raise
