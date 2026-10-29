# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Prepared BinIdx Energon bridge to heterogeneous MIMO batches and data lanes."""

from contextlib import ExitStack
from dataclasses import replace

import torch
from einops import rearrange

from examples.mimo.training.batch import collate_encoder_inputs
from examples.mimo.utils.hetero import get_data_lane_rank, get_language_batch_parallel_size
from examples.multimodal.data_loading.cookers.multimodal_interleaved_binidx import (
    cook_multimodal_interleaved_binidx,
)
from examples.multimodal.data_loading.multimodal_interleaved_encoder import (
    InterleavedPackingConfig,
    MultimodalInterleavedTaskEncoder,
)
from examples.multimodal.data_loading.multimodal_interleaved_loader import (
    InterleavedImageConfig,
    build_interleaved_iterator,
    materialize_document_images,
    restore_interleaved_iterator,
)
from megatron.core.models.mimo.config.role import MIMO_LANGUAGE_MODULE_KEY
from megatron.core.models.vision.encoder_registry import get_spec
from megatron.core.utils import get_pg_rank
from megatron.energon import Cooker, stateless


def configure_energon_data(args) -> None:
    """Validate the prepared-data route before initialization; leave mock args untouched."""
    if args.dataset_provider != "energon":
        return
    for name in ("mimo_llm_tp", "mimo_llm_pp", "mimo_llm_cp", "mimo_encoder_tp"):
        if getattr(args, name) != 1:
            raise ValueError(f"initial interleaved Energon support requires {name}=1")
    if getattr(args, "gtp_weight_remat_size", 1) != 1:
        raise ValueError("interleaved Energon GTP rematerialization is not validated")
    if (
        args.micro_batch_size != 1
        or type(args.num_workers) is not int
        or args.num_workers < 0
        or args.eval_iters != 0
    ):
        raise ValueError("interleaved Energon requires MBS1, nonnegative workers and eval_iters0")
    if args.mimo_encoder_dp <= 0 or args.mimo_llm_dp <= 0 or args.mimo_llm_only:
        raise ValueError("interleaved multimodal Energon requires an active encoder grid")
    if args.mimo_llm_dp % args.mimo_encoder_dp:
        raise ValueError("interleaved language sample lanes must be divisible by encoder DP")
    if not args.dynamic_resolution:
        raise ValueError("prepared Energon data requires --dynamic-resolution")
    if not args.disable_vision_class_token:
        raise ValueError("interleaved data requires no emitted class tokens")
    if args.patch_dim != 16 or not args.pixel_shuffle:
        raise ValueError("interleaved RADIO requires patch16 and pixel shuffle")
    if args.mimo_encoder_prefetch:
        raise ValueError("interleaved data does not support encoder-feature prefetch")
    if args.tokenizer_type != "MultimodalTokenizer":
        raise ValueError("interleaved data requires MultimodalTokenizer")
    if not args.data_path or len(args.data_path) != 1:
        raise ValueError(
            "interleaved Energon requires one --data-path recipe (blending lives in the recipe)"
        )

    # These attributes select MultiModalTokenizerConfig in the public config
    # builder. Install them only for the real provider, never the mock path.
    args.image_tag_type = ""
    args.force_system_message = False
    args.dataloader_save = args.dataloader_save or args.save


def initialize_energon_tokenizer(cfg, args) -> None:
    """Resolve prepared vocabulary metadata through the public config/tokenizer lifecycle."""
    if args.dataset_provider != "energon":
        return
    from megatron.training.global_vars import rebuild_tokenizer

    tokenizer = rebuild_tokenizer(cfg, args)
    image_id = tokenizer.vocab.get("<image>")
    if image_id is None:
        raise ValueError("the configured tokenizer must expose the structural <image> token")
    args.image_token_id = int(image_id)
    args.pad_token_id = int(tokenizer.pad)
    if args.vocab_size is not None and args.vocab_size != tokenizer.vocab_size:
        raise ValueError("--vocab-size disagrees with the loaded tokenizer")
    args.vocab_size = tokenizer.vocab_size


class MimoInterleavedTaskEncoder(MultimodalInterleavedTaskEncoder):
    """Translate prepared packs into canonical MIMO batches; no on-the-fly text encoding."""

    cookers = [
        Cooker(
            cook_multimodal_interleaved_binidx, has_tags={"cook": "multimodal_interleaved_binidx"}
        )
    ]

    def __init__(self, *args, encoder_name: str, role: str, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        if role not in ("language", "encoder"):
            raise ValueError(f"unknown MIMO data role {role!r}")
        self.encoder_name = encoder_name
        self.role = role

    @stateless
    def batch(self, samples):
        """Expand structural image slots and assemble the rank role's canonical batch."""
        if len(samples) != 1:
            raise ValueError("interleaved MIMO batches require one selected pack")
        pack = samples[0]
        cfg = self.config
        counts = iter(int(params.num_embeddings) for params in pack.image_params)
        tokens, labels = [], []
        for token, label in zip(pack.tokens.tolist(), pack.labels.tolist(), strict=True):
            if token == cfg.image_token_id:
                try:
                    count = next(counts)
                except StopIteration as error:
                    raise ValueError(
                        "interleaved pack has more image slots than image plans"
                    ) from error
                tokens.extend([token] * count)
                labels.extend([cfg.ignore_index] * count)
            else:
                tokens.append(token)
                labels.append(label)
        if next(counts, None) is not None or len(tokens) != cfg.sequence_length:
            raise ValueError("interleaved MIMO image expansion disagrees with the selected pack")
        batch = {
            "input_ids": rearrange(torch.tensor(tokens, dtype=torch.long), "sequence -> 1 sequence")
        }
        if self.role == "language":
            target = rearrange(torch.tensor(labels, dtype=torch.long), "sequence -> 1 sequence")
            sequence_lengths = pack.cu_seqlens.diff()
            positions = torch.cat([torch.arange(int(length)) for length in sequence_lengths])
            cu_seqlens = rearrange(pack.cu_seqlens, "boundary -> 1 boundary")
            batch.update(
                labels=target,
                loss_mask=(target != cfg.ignore_index).float(),
                position_ids=rearrange(positions, "sequence -> 1 sequence"),
                cu_seqlens=cu_seqlens,
                cu_seqlens_padded=cu_seqlens.clone(),
                max_seqlen=rearrange(sequence_lengths.max(), "-> 1"),
            )
        elif pack.image_params:
            values, sizes, cu, _ = materialize_document_images(pack, self.image_strategy)
            reduction = self.image_strategy.spatial_reduction
            patch = self.image_strategy._patch_size
            if torch.any((sizes // patch) % reduction):
                raise ValueError(
                    "each image patch-grid dimension must divide by the spatial reduction"
                )
            if int(cu[-1]) // reduction**2 != sum(p.num_embeddings for p in pack.image_params):
                raise ValueError("encoder spatial output disagrees with image token budget")
            inputs = collate_encoder_inputs([{"x": values, "imgs_sizes": sizes.to(torch.int32)}])
            batch["modality_inputs"] = {self.encoder_name: {self.encoder_name: inputs}}
        batch["__restore_key__"] = pack.__restore_key__
        return batch


class EncoderLaneIterator:
    """Consume the encoder's ordered language lanes; language state is authoritative."""

    def __init__(self, lanes, encoder_name: str) -> None:
        self.lanes = lanes
        self.encoder_name = encoder_name

    def __iter__(self):
        return self

    def __next__(self):
        batches = [next(lane) for lane in self.lanes]
        batch = {"input_ids": torch.cat([item["input_ids"] for item in batches], dim=0)}
        inputs = [
            item["modality_inputs"][self.encoder_name][self.encoder_name]
            for item in batches
            if item.get("modality_inputs")
        ]
        if inputs:
            combined = collate_encoder_inputs(inputs)
            batch["modality_inputs"] = {self.encoder_name: {self.encoder_name: combined}}
        return batch

    def save_state(self):
        """Leave checkpoint authority with the corresponding language iterators."""
        return None

    def close(self):
        """Attempt to close every lane even if a lane cleanup raises."""
        with ExitStack() as cleanup:
            for lane in self.lanes:
                cleanup.callback(lane.close)


def build_energon_data_loaders(args, topology):
    """Compose the shared data pipeline on the current explicit sample-lane topology."""
    from megatron.training import get_tokenizer

    names = [name for name in topology.grids if name != MIMO_LANGUAGE_MODULE_KEY]
    if len(names) != 1:
        raise ValueError("interleaved MIMO requires exactly one encoder module")
    name = names[0]
    language = topology.grids[MIMO_LANGUAGE_MODULE_KEY].is_current_rank_in_grid()
    vision = topology.grids[name].is_current_rank_in_grid()
    if language and vision:
        raise ValueError("interleaved MIMO requires non-colocated module grids")
    if not language and not vision:
        return None, None, None
    tokenizer = get_tokenizer()
    images = InterleavedImageConfig(
        args.multimodal_interleaved_min_image_patches,
        args.multimodal_interleaved_max_image_patches,
        args.patch_dim,
        encoder_spec=replace(get_spec("radio"), pixel_shuffle=True),
    )
    packing = InterleavedPackingConfig(
        args.seq_length, args.image_token_id, int(tokenizer.eod), int(tokenizer.pad)
    )
    lanes = get_language_batch_parallel_size(args)

    def build_lane(lane, role):
        encoder = MimoInterleavedTaskEncoder(
            tokenizer, images.build_strategy(), packing, encoder_name=name, role=role
        )
        iterator = build_interleaved_iterator(
            args.data_path[0],
            encoder,
            images,
            lane=lane,
            lanes=lanes,
            seed=args.seed,
            shuffle_buffer_size=args.multimodal_interleaved_shuffle_buffer_documents,
            num_workers=args.num_workers,
            global_batch_size=getattr(args, "global_batch_size", None),
            checkpoint_enabled=role == "language",
        )
        return restore_interleaved_iterator(
            iterator, args, lane=lane, pipeline_parallel=args.mimo_llm_pp > 1
        )

    if language:
        groups = topology.module_pgs[MIMO_LANGUAGE_MODULE_KEY]
        return build_lane(get_data_lane_rank(groups), "language"), None, None
    rank = get_pg_rank(topology.module_pgs[name].dp)
    fanout = lanes // args.mimo_encoder_dp
    children = []
    with ExitStack() as cleanup:
        for lane in range(rank * fanout, (rank + 1) * fanout):
            child = build_lane(lane, "encoder")
            children.append(child)
            cleanup.callback(child.close)
        iterator = EncoderLaneIterator(children, name)
        cleanup.pop_all()
    return iterator, None, None
