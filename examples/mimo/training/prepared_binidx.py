# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Prepared BinIdx causal batches and worker-prepared vision inputs for MIMO."""

import hashlib
import json
from copy import copy
from dataclasses import replace
from functools import partial

import torch
from PIL import Image

from examples.mimo.training.energon_provider import EnergonDataConfig
from examples.mimo.vision import resolve_encoder_spec
from examples.multimodal.data_loading.binidx_token_mask import IndexedTokenMaskReference
from examples.multimodal.data_loading.image_processing import (
    DynamicResolutionImageTilingStrategy,
    pixel_statistics,
)
from examples.multimodal.data_loading.prepared_streaming import PreparedStreamingTaskEncoder
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.energon import Lazy, NoCachePool, register_aux_filestore_protocol, stateless
from megatron.training.global_vars import get_run_config


class PreparedMimoTaskEncoder(PreparedStreamingTaskEncoder):
    """Batch fixed packs, optionally preparing encoder pixels without shifting targets.

    Positions run from zero across each packed row. Attention is ordinary causal
    attention across the row, including document boundaries. Source fragments
    record provenance; they do not request attention isolation or position resets.

    ``prepare_images`` defaults to False for language replicas and direct users.
    Encoder replicas set it to True and supply the matching encoder ``geometry``.
    Only the stateless terminal batch differs; source plans and packing state
    remain lazy and interchangeable between the two roles.
    """

    def __init__(self, image_strategy, *, prepare_images=False, geometry=None, **options) -> None:
        super().__init__(image_strategy, **options)
        if prepare_images and geometry is None:
            raise ValueError("worker image preparation requires encoder geometry")
        self.prepare_images = prepare_images
        self.geometry = geometry

    # Native fanout replays the logical stream for outputs owned by siblings.
    # This terminal callback has no persistent state and is safe to omit there;
    # image reads/transforms must not run for those discarded replicas.
    @stateless(skip_safe=True)
    def batch(self, samples):
        ids = torch.stack([sample.input_ids for sample in samples])
        batch = {
            "input_ids": ids,
            "labels": torch.stack([sample.labels for sample in samples]),
            "loss_mask": torch.stack([sample.loss_mask for sample in samples]),
            "position_ids": torch.arange(self.sequence_length).expand_as(ids).contiguous(),
            "modality_inputs": {
                "prepared_images": [
                    {"plans": sample.image_params, "spans": sample.image_spans}
                    for sample in samples
                ]
            },
        }
        if self.prepare_images:
            for row in batch["modality_inputs"]["prepared_images"]:
                row["tiles"] = _realize_image_row(
                    row, image_strategy=self.image_strategy, geometry=self.geometry
                )
        return batch


def _realize_image_row(row, *, image_strategy, geometry):
    """Prepare ordered CPU tiles without changing lazy source/checkpoint objects."""
    if len(row["plans"]) != len(row["spans"]):
        raise ValueError("prepared image plans must match visual spans")
    images = []
    for plan, (start, stop) in zip(row["plans"], row["spans"]):
        pixels = plan.media.value
        if isinstance(pixels, Lazy):
            pixels = copy(pixels).get()
        if not isinstance(pixels, Image.Image):
            raise TypeError("prepared vision images must decode to PIL images")
        if pixels.size != (plan.media.width, plan.media.height):
            raise ValueError("decoded image size disagrees with original width/height metadata")
        realized = replace(plan, media=replace(plan.media, value=pixels))
        tiles = image_strategy.apply_params(realized)
        actual = sum(
            geometry.num_spatial_embeddings(int(tile.shape[-2]), int(tile.shape[-1]))
            for tile in tiles
        )
        if len(tiles) != plan.num_tiles or actual != plan.num_embeddings or actual != stop - start:
            raise ValueError("realized encoder geometry disagrees with prepared visual spans")
        images.extend(tiles)
    return images


def merge_prepared_encoder_batches(batches, *, lane_ids, module_name, image_strategy, geometry):
    """Assemble worker tiles in lane/row order on encoder ranks only.

    Direct users may still supply lazy batches; they use the same realization
    helper. Worker-prepared batches do not repeat image reads or transforms.
    """
    if len(batches) != len(lane_ids) or not batches:
        raise ValueError("one prepared batch is required per ordered sample lane")
    images = []
    for batch in batches:
        rows = batch["modality_inputs"]["prepared_images"]
        if len(rows) != batch["input_ids"].shape[0]:
            raise ValueError("prepared image rows must match the native batch rows")
        for row in rows:
            images.extend(
                row["tiles"]
                if "tiles" in row
                else _realize_image_row(row, image_strategy=image_strategy, geometry=geometry)
            )
    result = {"input_ids": torch.cat([batch["input_ids"] for batch in batches])}
    if images:
        x, sizes, cu, maximum = image_strategy.stack(images)
        maximum = int(maximum)
        # RADIO inserts its class tokens into these pre-shuffle patch boundaries.
        packed = PackedSeqParams(
            qkv_format="thd",
            cu_seqlens_q=cu,
            cu_seqlens_kv=cu.clone(),
            max_seqlen_q=maximum,
            max_seqlen_kv=maximum,
        )
        result["modality_inputs"] = {
            module_name: {module_name: {"x": x, "imgs_sizes": sizes, "packed_seq_params": packed}}
        }
    return result


def _count_spatial_embeddings(width, height, *, geometry, **kwargs):
    return geometry.num_spatial_embeddings(height, width)


def _make_encoder(*, split, strategy, options, geometry, prepare_images=False):
    if split != "train":
        raise ValueError("prepared MIMO application currently supports only training")
    return PreparedMimoTaskEncoder(
        strategy, geometry=geometry, prepare_images=prepare_images, **options
    )


def make_config(args):
    """Build the opt-in prepared vision application using explicit prepared-token IDs.

    Select with ``--energon-data-config examples.mimo.training.prepared_binidx:make_config``.
    This application supports dynamic RADIO with optional pixel shuffle and
    native Pixtral with its preset spatial merger. Both drop class tokens. Image budgets count emitted
    visual embeddings, never source end/break tokens. Recipe artifacts must stay
    immutable for checkpoint replay.
    """
    if not args.dynamic_resolution or not args.disable_vision_class_token:
        raise ValueError(
            "prepared vision data requires dynamic resolution and dropped class tokens"
        )
    geometry = resolve_encoder_spec(args)
    native = geometry.name != "radio"
    vision_model_type = geometry.name if native else "radio"
    # These CLI values also configure the actual encoder. Registry defaults must
    # not silently replace explicitly selected patch size or pixel shuffle.
    if type(args.patch_dim) is not int or args.patch_dim <= 0:
        raise ValueError("patch_dim must be a positive integer")
    budget = args.energon_image_token_budget
    if type(budget) is not int or not 0 < budget <= args.seq_length:
        raise ValueError("--energon-image-token-budget must be positive and at most --seq-length")
    for name in ("energon_eos_token_id", "energon_pad_token_id"):
        if type(getattr(args, name)) is not int:
            raise ValueError(f"--{name.replace('_', '-')} must be explicitly configured")
    strategy = DynamicResolutionImageTilingStrategy(
        vision_model_type=vision_model_type,
        min_num_patches=geometry.spatial_reduction**2,
        max_num_patches=budget * geometry.spatial_reduction**2,
        patch_size=args.patch_dim,
        get_num_embeddings=partial(_count_spatial_embeddings, geometry=geometry),
        pixel_shuffle=args.pixel_shuffle,
        use_thumbnail=False,
        spatial_merge_size=geometry.spatial_merge_size,
        max_image_size=(geometry.default_img_h, geometry.default_img_w) if native else None,
    )
    tokenizer_config = get_run_config().tokenizer
    options = dict(
        image_token_id=args.image_token_id,
        eos_token_id=args.energon_eos_token_id,
        pad_token_id=args.energon_pad_token_id,
        vocab_size=tokenizer_config.padded_vocab_size or tokenizer_config.vocab_size,
        image_token_budget=budget,
        sequence_length=args.seq_length,
    )
    # Validate selected policy before any source or worker is opened.
    PreparedMimoTaskEncoder(strategy, **options)
    register_aux_filestore_protocol(
        "indexed-token-mask", lambda path: IndexedTokenMaskReference(fs_path=path), override=True
    )
    identity = {
        "format": 1,
        "task": "prepared-causal-stream-radio",
        **options,
        "vision_model_type": "radio",
        "patch_dim": args.patch_dim,
        "pixel_shuffle": args.pixel_shuffle,
        "drop_class_tokens": True,
        "image_min_raw_patches": 4 if args.pixel_shuffle else 1,
        "image_max_raw_patches": budget * geometry.spatial_reduction**2,
        "thumbnail": False,
        "attention": "causal-across-documents",
        "position_reset": False,
        "target_ownership": "next-token-v1",
        "targetless_pack": "error",
        "image_normalization": pixel_statistics["radio"][0],
        "image_std": pixel_statistics["radio"][1],
    }
    if native:
        identity.update(
            task="prepared-causal-stream-native-pixtral",
            vision_model_type=vision_model_type,
            spatial_merge_size=geometry.spatial_merge_size,
            image_min_raw_patches=geometry.spatial_reduction**2,
            image_max_height=geometry.default_img_h,
            image_max_width=geometry.default_img_w,
            image_resize_policy="aligned-native-axis-bounds-v1",
            image_normalization=pixel_statistics[vision_model_type][0],
            image_std=pixel_statistics[vision_model_type][1],
        )
    return EnergonDataConfig(
        create_task_encoder=partial(
            _make_encoder, strategy=strategy, options=options, geometry=geometry
        ),
        create_encoder_task_encoder=partial(
            _make_encoder,
            strategy=strategy,
            options=options,
            geometry=geometry,
            prepare_images=True,
        ),
        checkpoint_identity=hashlib.sha256(
            json.dumps(identity, sort_keys=True, allow_nan=False).encode()
        ).hexdigest(),
        merge_encoder_batches=partial(
            merge_prepared_encoder_batches, image_strategy=strategy, geometry=geometry
        ),
        dataset_kwargs={"packing_buffer_size": "stream"},
        loader_kwargs={"cache_pool": NoCachePool()},
    )
