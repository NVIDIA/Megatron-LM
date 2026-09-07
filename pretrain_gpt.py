# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Pretrain and SFT GPT."""

# Capture the true program start time BEFORE any heavy imports.
import time

_PROGRAM_START_TIME = time.time()

import json

# Suppress warnings on all ranks but rank 0.
import os
import warnings

rank = int(os.environ.get('RANK', 0))
if rank != 0:
    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)

from functools import lru_cache, partial
from typing import Any, List, NamedTuple, Optional, Tuple

import torch

from gpt_builders import gpt_builder
from megatron.core import mpu
from megatron.core.context_parallel_layout import finalize_packed_seq_params
from megatron.core.datasets.blended_megatron_dataset_builder import BlendedMegatronDatasetBuilder
from megatron.core.datasets.data_schedule import get_batch_on_this_rank_for_sequence_packing
from megatron.core.datasets.gpt_dataset import GPTDataset, GPTDatasetConfig, MockGPTDataset
from megatron.core.enums import ModelType
from megatron.core.full_cuda_graph import FullCudaGraphPreparedIterator
from megatron.core.models.gpt import GPTModel
from megatron.core.packed_seq_params import (
    PackedSeqParams,
    get_thd_padding_kwargs,
    pad_sequence_for_thd,
    resolve_thd_tail_padding_policy,
)
from megatron.core.rerun_state_machine import get_rerun_state_machine
from megatron.core.tokenizers.utils.build_tokenizer import build_tokenizer
from megatron.core.transformer.cuda_graph_config import cuda_graph_captures_attention
from megatron.core.transformer.experimental_attention_variant.cp_balanced_indexer import (
    GraphDynamicRouteBuffers,
    GraphDynamicRouteSpec,
    attach_graph_dynamic_route,
    get_graph_dynamic_plan_buffers,
    prebuild_balanced_layouts,
    validate_graph_dynamic_plan_contract,
    validate_graph_dynamic_route,
)
from megatron.core.transformer.multi_token_prediction import get_mtp_ranks, mtp_on_this_rank
from megatron.core.utils import (
    StragglerDetector,
    get_attr_wrapped_model,
    get_thd_batch_on_this_cp_rank,
)
from megatron.training import (
    get_args,
    get_timers,
    inprocess_restart,
    pretrain,
    print_rank_0,
    set_startup_timestamps,
)
from megatron.training.argument_utils import pretrain_cfg_container_from_args
from megatron.training.arguments import core_transformer_config_from_args, parse_and_validate_args
from megatron.training.datasets.fim_dataset import GPTFIMDataset, GPTFIMDatasetConfig
from megatron.training.datasets.sft_dataset import MockSFTDataset, SFTDataset
from megatron.training.datasets.varlen_dataset import MockVarlenDataset, VarlenDataset
from megatron.training.utils import (
    get_batch_on_this_cp_rank,
    get_batch_on_this_tp_rank,
    get_blend_and_blend_per_split,
    is_first_or_last_pipeline_stage,
)
from model_provider import model_provider

try:
    from megatron.post_training.arguments import add_modelopt_args
    from megatron.post_training.loss_func import loss_func as loss_func_modelopt

    has_nvidia_modelopt = True
except ImportError:
    has_nvidia_modelopt = False

stimer = StragglerDetector()

_FULL_CUDA_GRAPH_PACKED_BASE_BATCH_KEYS = (
    "tokens",
    "labels",
    "loss_mask",
    "position_ids",
    "padding_mask",
    "cu_seqlens",
    "cu_seqlens_padded",
)
_FULL_CUDA_GRAPH_PACKED_ROUTE_KEYS = (
    "dsa_cp_graph_layout_buffer",
    "dsa_cp_graph_route_buffer",
)
_FULL_CUDA_GRAPH_PACKED_BATCH_KEYS = (
    _FULL_CUDA_GRAPH_PACKED_BASE_BATCH_KEYS + _FULL_CUDA_GRAPH_PACKED_ROUTE_KEYS
)


class _FullCudaGraphPackedBatchSpec(NamedTuple):
    """Run-static schema shared by base and balanced prepared packed inputs."""

    cp_size: int
    cp_rank: int
    l_local: int
    cu_entries: int

    @property
    def capacity(self):
        return self.cp_size * self.l_local


def _validate_full_cuda_graph_prepared_packed_config(config):
    """Validate the common fixed-capacity packed-input contract."""
    if getattr(config, "cuda_graph_impl", None) != "full_iteration":
        raise ValueError("prepared packed batches require cuda_graph_impl='full_iteration'")
    if getattr(config, "experimental_attention_variant", None) != "dsv4_hybrid":
        raise ValueError(
            "prepared packed batches currently require "
            "experimental_attention_variant='dsv4_hybrid'"
        )
    if getattr(config, "sequence_packing_scheduler", None) != "dp_balanced":
        raise ValueError(
            "prepared packed batches require sequence_packing_scheduler='dp_balanced'"
        )
    if getattr(config, "dynamic_context_parallel", False):
        raise ValueError("prepared packed batches do not support dynamic context parallelism")
    if getattr(config, "cp_partition_mode", None) != "contiguous":
        raise ValueError("prepared packed batches require cp_partition_mode='contiguous'")
    if not getattr(config, "calculate_per_token_loss", False):
        raise ValueError("prepared packed batches require calculate_per_token_loss=True")
    if (
        getattr(config, "pipeline_model_parallel_size", 1) != 1
        or (getattr(config, "virtual_pipeline_model_parallel_size", None) or 1) != 1
    ):
        raise ValueError("prepared packed batches currently require PP=1 and no VPP")

    balance_indexer = bool(getattr(config, "dsa_cp_balance_indexer", False))
    dynamic_route = bool(
        getattr(config, "dsa_cp_balance_indexer_graph_dynamic_packs", False)
    )
    if balance_indexer != dynamic_route:
        raise ValueError(
            "full-iteration prepared packed config has inconsistent balanced-route state: "
            f"dsa_cp_balance_indexer={balance_indexer}, "
            f"derived dynamic route={dynamic_route}"
        )

    cp_size = getattr(config, "context_parallel_size", None)
    l_local = getattr(config, "max_seqlen_per_dp_cp_rank", None)
    max_sequences = getattr(config, "thd_max_packed_sequences", None)
    for name, value in (
        ("context_parallel_size", cp_size),
        ("max_seqlen_per_dp_cp_rank", l_local),
        ("thd_max_packed_sequences", max_sequences),
    ):
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise ValueError(f"prepared packed batches require positive integer {name}")
    if cp_size <= 1:
        raise ValueError("prepared packed batches require context_parallel_size > 1")
    return cp_size, l_local, max_sequences + 1


def _validate_full_cuda_graph_dynamic_packed_config(config):
    """Validate the balanced-route extension to the common packed-input contract."""
    if not getattr(config, "dsa_cp_balance_indexer_graph_dynamic_packs", False):
        raise ValueError(
            "balanced full-iteration prepared batches require the config to infer "
            "fixed-capacity dynamic routing"
        )
    if not getattr(config, "dsa_cp_balance_indexer", False):
        raise ValueError("prepared dynamic packed batches require dsa_cp_balance_indexer=True")
    return _validate_full_cuda_graph_prepared_packed_config(config)


def _full_cuda_graph_packed_context(config, pg_collection=None):
    """Resolve the run-static CP group and common packed-input schema."""
    cp_size, l_local, cu_entries = _validate_full_cuda_graph_prepared_packed_config(config)
    cp_group = (
        mpu.get_context_parallel_group() if pg_collection is None else pg_collection.cp
    )
    if cp_group is None or cp_group.size() != cp_size:
        actual_size = None if cp_group is None else cp_group.size()
        raise RuntimeError(
            "prepared packed batch CP group does not match the configured topology: "
            f"expected {cp_size}, got {actual_size}"
        )
    spec = _FullCudaGraphPackedBatchSpec(
        cp_size=cp_size, cp_rank=cp_group.rank(), l_local=l_local, cu_entries=cu_entries
    )
    return cp_group, spec


def _full_cuda_graph_route_context(config, pg_collection=None):
    """Resolve the run-static CP group and typed route schema."""
    cp_size, l_local, cu_entries = _validate_full_cuda_graph_dynamic_packed_config(config)
    cp_group = (
        mpu.get_context_parallel_group() if pg_collection is None else pg_collection.cp
    )
    if cp_group is None or cp_group.size() != cp_size:
        actual_size = None if cp_group is None else cp_group.size()
        raise RuntimeError(
            "prepared dynamic packed batch CP group does not match the configured topology: "
            f"expected {cp_size}, got {actual_size}"
        )
    spec = GraphDynamicRouteSpec(
        cp_size=cp_size, cp_rank=cp_group.rank(), l_local=l_local, cu_entries=cu_entries
    )
    return cp_group, spec


def _validate_full_cuda_graph_tensor_owners(batch, spec, *, include_route=True):
    """Validate the fixed prepared-owner schema using host tensor metadata only."""
    if not isinstance(batch, dict):
        raise TypeError("prepared full-iteration CUDA graph batch must be a flat dictionary")
    actual_keys = set(batch)
    ordered_keys = (
        _FULL_CUDA_GRAPH_PACKED_BATCH_KEYS
        if include_route
        else _FULL_CUDA_GRAPH_PACKED_BASE_BATCH_KEYS
    )
    expected_keys = set(ordered_keys)
    if actual_keys != expected_keys:
        missing = sorted(expected_keys - actual_keys)
        extra = sorted(actual_keys - expected_keys)
        raise ValueError(
            "prepared full-iteration CUDA graph batch schema mismatch: "
            f"missing={missing}, extra={extra}"
        )

    # This prepared ABI currently supports one THD pack per microbatch (and PP1), so
    # every token-like owner deliberately retains the scheduler's [1, L] shape.
    # Accepting an arbitrary leading shape here would turn it into an accidental
    # graph ABI and could silently reinterpret a future scheduler layout.
    expected = {
        "tokens": (torch.int64, (1, spec.l_local)),
        "labels": (torch.int64, (1, spec.l_local)),
        "loss_mask": (torch.float32, (1, spec.l_local)),
        "position_ids": (torch.int64, (1, spec.l_local)),
        "padding_mask": (torch.bool, (1, spec.l_local)),
        "cu_seqlens": (torch.int32, (spec.cu_entries,)),
        "cu_seqlens_padded": (torch.int32, (spec.cu_entries,)),
    }
    owner_device = None
    for name in ordered_keys:
        tensor = batch[name]
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"prepared full-iteration CUDA graph owner {name!r} must be a tensor")
        if not tensor.is_contiguous():
            raise ValueError(
                f"prepared full-iteration CUDA graph owner {name!r} must be contiguous"
            )
        if owner_device is None:
            owner_device = tensor.device
        elif tensor.device != owner_device:
            raise ValueError(
                "prepared full-iteration CUDA graph owners must be on one device: "
                f"{name!r} is on {tensor.device}, expected {owner_device}"
            )
        if name in expected:
            expected_dtype, expected_shape = expected[name]
            if tensor.dtype != expected_dtype or tuple(tensor.shape) != expected_shape:
                raise ValueError(
                    f"prepared full-iteration CUDA graph owner {name!r} must have "
                    f"dtype={expected_dtype} and shape={expected_shape}, got "
                    f"dtype={tensor.dtype} and shape={tuple(tensor.shape)}"
                )

    if not include_route:
        return None
    return GraphDynamicRouteBuffers(
        batch["dsa_cp_graph_layout_buffer"], batch["dsa_cp_graph_route_buffer"], spec
    )


def _serialize_full_cuda_graph_dynamic_packed_batch(batch, config, pg_collection=None):
    """Flatten an eager, fully finalized GPT batch into fixed tensor owners."""
    if not isinstance(batch, tuple) or len(batch) != 7:
        raise TypeError("full-iteration packed GPT prepare must receive the seven-value batch")
    tokens, labels, loss_mask, attention_mask, position_ids, packed, padding_mask = batch
    if attention_mask is not None:
        raise ValueError("packed THD full-iteration batches require attention_mask=None")
    if not isinstance(packed, PackedSeqParams) or packed.qkv_format != "thd":
        raise TypeError("full-iteration dynamic packed batch requires THD PackedSeqParams")

    include_route = bool(
        getattr(config, "dsa_cp_balance_indexer_graph_dynamic_packs", False)
    )
    cp_group, spec = (
        _full_cuda_graph_route_context(config, pg_collection)
        if include_route
        else _full_cuda_graph_packed_context(config, pg_collection)
    )
    if packed.cp_group is not cp_group:
        raise ValueError("eager packed batch carries the wrong fixed context-parallel group")
    if packed.local_cp_size is not None:
        raise ValueError("full-iteration dynamic packed batch does not support local_cp_size")
    if packed.cp_partition_mode != "contiguous":
        raise ValueError("full-iteration dynamic packed batch must be CP-contiguous")
    if packed.pad_between_seqs is not True:
        raise ValueError("full-iteration dynamic packed batch requires pad_between_seqs=True")
    if packed.total_tokens is not None:
        raise ValueError("full-iteration dynamic packed batch requires total_tokens=None")

    global_capacity = spec.capacity
    if (packed.max_seqlen_q, packed.max_seqlen_kv) != (global_capacity, global_capacity):
        raise ValueError(
            "full-iteration dynamic packed max_seqlen must equal the fixed global capacity: "
            f"expected {global_capacity}, got q={packed.max_seqlen_q}, "
            f"kv={packed.max_seqlen_kv}"
        )
    if packed.cu_seqlens_q is None or packed.cu_seqlens_q_padded is None:
        raise ValueError("full-iteration dynamic packed batch requires real and padded cu_seqlens")
    if packed.cu_seqlens_kv is None or not torch.equal(packed.cu_seqlens_q, packed.cu_seqlens_kv):
        raise ValueError("full-iteration dynamic packed q/kv real cu_seqlens must match")
    if packed.cu_seqlens_kv_padded is None or not torch.equal(
        packed.cu_seqlens_q_padded, packed.cu_seqlens_kv_padded
    ):
        raise ValueError("full-iteration dynamic packed q/kv padded cu_seqlens must match")

    owners = {
        "tokens": tokens,
        "labels": labels,
        "loss_mask": loss_mask,
        "position_ids": position_ids,
        "padding_mask": padding_mask,
        "cu_seqlens": packed.cu_seqlens_q,
        "cu_seqlens_padded": packed.cu_seqlens_q_padded,
    }
    if include_route:
        validate_graph_dynamic_plan_contract(packed, spec.cp_size, spec.cp_rank, spec.l_local)
        layout_i32, route_i64 = get_graph_dynamic_plan_buffers(packed)
        owners.update(
            {
                "dsa_cp_graph_layout_buffer": layout_i32,
                "dsa_cp_graph_route_buffer": route_i64,
            }
        )
    buffers = _validate_full_cuda_graph_tensor_owners(
        owners, spec, include_route=include_route
    )
    if buffers is not None:
        validate_graph_dynamic_route(owners["cu_seqlens_padded"], buffers)
    return owners


def _reconstruct_full_cuda_graph_dynamic_packed_batch(batch, config, pg_collection=None):
    """Reconstruct the THD object during capture without eager data preparation."""
    include_route = bool(
        getattr(config, "dsa_cp_balance_indexer_graph_dynamic_packs", False)
    )
    cp_group, spec = (
        _full_cuda_graph_route_context(config, pg_collection)
        if include_route
        else _full_cuda_graph_packed_context(config, pg_collection)
    )
    buffers = _validate_full_cuda_graph_tensor_owners(
        batch, spec, include_route=include_route
    )
    if buffers is not None:
        validate_graph_dynamic_route(batch["cu_seqlens_padded"], buffers)

    real_cu = batch["cu_seqlens"]
    padded_cu = batch["cu_seqlens_padded"]
    packed = PackedSeqParams(
        qkv_format="thd",
        cu_seqlens_q=real_cu,
        cu_seqlens_kv=real_cu,
        cu_seqlens_q_padded=padded_cu,
        cu_seqlens_kv_padded=padded_cu,
        max_seqlen_q=spec.capacity,
        max_seqlen_kv=spec.capacity,
        local_cp_size=None,
        cp_group=cp_group,
        total_tokens=None,
        pad_between_seqs=True,
        cp_partition_mode="contiguous",
        cp_partition_route=None,
    )
    # Validation above emits only capture-safe device assertions. Balanced runs
    # additionally attach the typed route owners; the base indexer consumes only
    # the replay-updated token and packed-metadata owners.
    if buffers is not None:
        attach_graph_dynamic_route(packed, buffers)
    return (
        batch["tokens"],
        batch["labels"],
        batch["loss_mask"],
        None,
        batch["position_ids"],
        packed,
        batch["padding_mask"],
    )


def prepare_full_cuda_graph_dynamic_packed_batch(
    *, data_iterator, model, stage, microbatch_index, model_chunk_index, pg_collection=None
):
    """Eager packed-input prologue invoked by ``FullCudaGraphWrapper`` on every rank."""
    if isinstance(data_iterator, FullCudaGraphPreparedIterator):
        raise TypeError("full-iteration batch prepare expects a raw data iterator")
    if stage not in ("training", "validation"):
        raise ValueError(f"invalid full-iteration CUDA graph stage {stage!r}")
    for name, value in (
        ("microbatch_index", microbatch_index),
        ("model_chunk_index", model_chunk_index),
    ):
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise ValueError(f"{name} must be a non-negative integer, got {value!r}")
    # Tie the prepared schema to the exact model chunk that the wrapper will
    # capture. ``get_batch`` still owns its normal args-derived eager path, but
    # the prologue must not rebuild a second TransformerConfig for every slot.
    config = get_attr_wrapped_model(model, "config", allow_none=False)
    _validate_full_cuda_graph_prepared_packed_config(config)
    vp_stage = get_attr_wrapped_model(model, "vp_stage")
    eager_batch = get_batch(
        data_iterator, vp_stage, config=config, pg_collection=pg_collection
    )
    return _serialize_full_cuda_graph_dynamic_packed_batch(
        eager_batch, config, pg_collection
    )


def get_batch(
    data_iterator,
    vp_stage: Optional[int] = None,
    *,
    config=None,
    pg_collection=None,
):
    """Generate a batch.

    Packed sequence support (SFT / ``--sft`` flag):
        When ``args.sft`` is True, the dataset emits THD-format batches where
        multiple sequences are concatenated into a single flat token tensor.
        The batch includes ``cu_seqlens`` (cumulative sequence lengths, shape
        ``[1, S+1]``) and ``max_seqlen`` (shape ``[1]``) that describe the
        individual sequence boundaries.

        This function validates and squeezes those fields:
          - ``cu_seqlens``:  asserted to have shape ``[1, S+1]`` (micro-batch
            size must be 1 for packing), then squeezed to ``[S+1]``.
          - ``max_seqlen``:  asserted to be 1-D; kept as a tensor and passed
            to ``get_thd_batch_on_this_cp_rank`` which performs the final
            scalar conversion internally.

        Pipeline stage handling:
          - First/last PP stages: fetch the full batch (tokens + labels) and
            route through ``get_thd_batch_on_this_cp_rank`` to produce a
            ``PackedSeqParams`` object that carries ``cu_seqlens`` and
            ``max_seqlen`` to the attention kernel.
          - Middle PP stages: only ``cu_seqlens`` and ``max_seqlen`` are
            needed for attention masking; all other fields are returned as
            ``None`` with a ``PackedSeqParams`` built directly here.
          - MTP ranks (``mtp_on_this_rank``) also receive the full batch,
            regardless of pipeline stage.

        Difference from ``pretrain_hybrid.py``:
          - Return format: GPT returns a 6-tuple
            ``(tokens, labels, loss_mask, attention_mask, position_ids,
            packed_seq_params)`` where ``packed_seq_params`` is a
            ``PackedSeqParams`` dataclass.  Mamba returns 7 values via
            ``batch.values()`` with ``cu_seqlens`` and ``max_seqlen`` as
            separate dict entries (no ``PackedSeqParams`` wrapper).
          - Middle-stage return: GPT returns ``(None×5, PackedSeqParams)``;
            Mamba returns an ``empty_batch`` dict with ``cu_seqlens`` and
            ``max_seqlen`` set.
          - CP with packed sequences: GPT delegates to
            ``get_thd_batch_on_this_cp_rank`` (MCore utility); Mamba
            implements the ``tex.thd_get_partitioned_indices`` CP slicing
            inline and does not call that helper.
          - MTP: GPT passes ``mtp_on_this_rank`` to ``get_batch_on_this_tp_rank``
            and uses it to gate the early-return; Mamba has no MTP support.
          - ``max_seqlen`` conversion: Mamba converts to a Python int scalar
            before returning (``int(max_seqlen[0].item())``); GPT keeps it as
            a tensor and lets ``get_thd_batch_on_this_cp_rank`` convert it,
            except for the middle-stage ``PackedSeqParams`` where conversion
            is done inline.
    """
    args = get_args()
    if isinstance(data_iterator, FullCudaGraphPreparedIterator):
        if config is None:
            if data_iterator.model_chunk is None:
                raise RuntimeError(
                    "prepared full-iteration batch is missing its owning model chunk"
                )
            config = get_attr_wrapped_model(
                data_iterator.model_chunk, "config", allow_none=False
            )
        if pg_collection is None:
            pg_collection = data_iterator.pg_collection
        return _reconstruct_full_cuda_graph_dynamic_packed_batch(
            next(data_iterator), config, pg_collection
        )

    if config is None:
        config = core_transformer_config_from_args(args)

    balance_indexer = getattr(config, "dsa_cp_balance_indexer", False)
    graph_dynamic_packs = getattr(config, "dsa_cp_balance_indexer_graph_dynamic_packs", False)

    if config.sequence_packing_scheduler is not None:
        # `get_batch_on_this_rank_for_sequence_packing` owns scheduler THD metadata
        # and returns a 7-tuple including `padding_mask`.
        batch = get_batch_on_this_rank_for_sequence_packing(
            data_iterator,
            vpp_size=config.virtual_pipeline_model_parallel_size,
            mtp_on_this_rank=mtp_on_this_rank(config, ignore_virtual=False, vp_stage=vp_stage),
            vp_stage=vp_stage,
            dynamic_cp=config.dynamic_context_parallel,
            pg_collection=pg_collection,
            config=config,
        )
        finalize_packed_seq_params(batch[5])
        if balance_indexer:
            prebuild_balanced_layouts(
                batch[5],
                pad_alignment=config.pad_packed_seq_alignment,
                capacity=(
                    config.max_seqlen_per_dp_cp_rank * config.context_parallel_size
                    if graph_dynamic_packs
                    else None
                ),
                graphs_enabled=cuda_graph_captures_attention(config),
                graph_dynamic_packs=graph_dynamic_packs,
            )
        return batch

    # TODO: this is pretty hacky, find a better way
    is_packed_sequence = args.sft or (args.use_varlen_dataset and not args.varlen_sbhd_validation)
    needs_padding_mask = args.use_varlen_dataset and args.varlen_sbhd_validation
    if (
        not is_first_or_last_pipeline_stage(vp_stage)
        and not is_packed_sequence
        and not needs_padding_mask
        and ((not mtp_on_this_rank(config, ignore_virtual=False, vp_stage=vp_stage)))
    ):
        return None, None, None, None, None, None, None

    # get batches based on the TP rank you are on
    batch = get_batch_on_this_tp_rank(
        data_iterator,
        mtp_on_this_rank=mtp_on_this_rank(config, ignore_virtual=False, vp_stage=vp_stage),
        needs_padding_mask=needs_padding_mask,
    )

    cu_seqlens = batch.pop('cu_seqlens', None)
    cu_seqlens_padded = batch.pop('cu_seqlens_padded', None)
    max_seqlen = batch.pop('max_seqlen', None)
    batch.pop('local_cp_size', None)

    if cu_seqlens is not None:
        assert (
            cu_seqlens.dim() == 2 and cu_seqlens.shape[0] == 1
        ), "micro-batch-size must be 1 for packing"
        cu_seqlens = cu_seqlens[0]
        assert max_seqlen.dim() == 1

    # For middle pipeline stages with packed sequences, only cu_seqlens and
    # max_seqlen are needed (for attention masking); skip the full batch.
    if not is_first_or_last_pipeline_stage(vp_stage) and is_packed_sequence:
        packed_seq_params = PackedSeqParams(
            cu_seqlens_q=cu_seqlens,
            cu_seqlens_kv=cu_seqlens,
            max_seqlen_q=int(max_seqlen[0].item()),
            max_seqlen_kv=int(max_seqlen[0].item()),
            qkv_format='thd',
        )
        finalize_packed_seq_params(packed_seq_params)
        if balance_indexer:
            # Middle-stage PackedSeqParams carry the raw cu; the hidden states are
            # padded, so probe/build at the physical capacity. Eligibility still
            # comes from the actual sequence boundaries plus that capacity tail;
            # an unrepresentable pack records False and uses the reference path.
            prebuild_balanced_layouts(
                packed_seq_params,
                pad_alignment=config.pad_packed_seq_alignment,
                capacity=args.seq_length,
                graphs_enabled=cuda_graph_captures_attention(config),
                graph_dynamic_packs=graph_dynamic_packs,
            )
        return (None, None, None, None, None, packed_seq_params, None)

    thd_tail_padding_policy = resolve_thd_tail_padding_policy(config)
    if cu_seqlens is None:
        # slice batch along sequence dimension for context parallelism
        batch = get_batch_on_this_cp_rank(batch)  # The implementation of this function is in MCore
        packed_seq_params = None
    else:  # Packed THD format
        batch, packed_seq_params = get_thd_batch_on_this_cp_rank(
            batch, cu_seqlens, cu_seqlens_padded, max_seqlen
        )

    # Pad the already-packed THD tensors at the end when requested. A configured
    # thd_max_packed_sequences also pads cu_seqlens to a fixed capacity in eager or graph mode.
    # SBHD validation samples carry physical right-padding metadata. CP has
    # already partitioned it with the other sequence-dimension tensors.
    padding_mask = batch.pop('padding_mask', None)
    if config.pad_packed_seq_alignment is not None and packed_seq_params is not None:
        tokens = batch.get('tokens', None)
        labels = batch.get('labels', None)
        loss_mask = batch.get('loss_mask', None)
        position_ids = batch.get('position_ids', None)
        alignment, target_len, max_num_seqs = get_thd_padding_kwargs(
            config.pad_packed_seq_alignment,
            config.max_seqlen_per_dp_cp_rank,
            config.thd_max_packed_sequences,
            config.cuda_graph_impl != "none",
        )
        tokens, labels, loss_mask, position_ids, packed_seq_params, padding_mask = (
            pad_sequence_for_thd(
                tokens,
                labels,
                loss_mask,
                position_ids,
                packed_seq_params,
                alignment=alignment,
                target_len=target_len,
                max_num_seqs=max_num_seqs,
                tail_padding_policy=thd_tail_padding_policy,
                padding_mask=padding_mask,
            )
        )
        if 'tokens' in batch:
            batch['tokens'] = tokens
        if 'labels' in batch:
            batch['labels'] = labels
        if 'loss_mask' in batch:
            batch['loss_mask'] = loss_mask
        if 'position_ids' in batch:
            batch['position_ids'] = position_ids

    finalize_packed_seq_params(packed_seq_params)
    if balance_indexer:
        prebuild_balanced_layouts(
            packed_seq_params,
            pad_alignment=config.pad_packed_seq_alignment,
            graphs_enabled=cuda_graph_captures_attention(config),
            graph_dynamic_packs=graph_dynamic_packs,
        )

    # Unpack explicitly to avoid relying on dict insertion order.
    return (
        batch.get('tokens'),
        batch.get('labels'),
        batch.get('loss_mask'),
        batch.get('attention_mask'),
        batch.get('position_ids'),
        packed_seq_params,
        padding_mask,
    )


# define spiky loss as a loss that's 10x the max loss observed
SPIKY_LOSS_FACTOR = 10


@lru_cache(maxsize=1)
def _build_cached_logits_loss_func(
    logprobs_dir, decode_threads, prefetch_factor, msc_prefetch_depth, kd_loss_alpha, ignore_errors
):
    """Build (once) the offline knowledge-distillation loss callable for cached logits.

    Memoized so the teacher log-probability reader is constructed a single time per
    process, replacing the previous module-level mutable global.
    """
    from megatron.training.distillation import LossFuncCallable

    return LossFuncCallable(
        logprobs_dir=logprobs_dir,
        decode_threads=decode_threads,
        prefetch_factor=prefetch_factor,
        msc_prefetch_depth=msc_prefetch_depth,
        kd_loss_alpha=kd_loss_alpha,
        ignore_errors=ignore_errors,
    )


def loss_func(
    loss_mask: torch.Tensor, output_tensor: torch.Tensor, model: Optional[GPTModel] = None
):
    """Loss function.

    Args:
        loss_mask (torch.Tensor): Used to mask out some portions of the loss
        output_tensor (torch.Tensor): The tensor with the losses
        model (GPTModel, optional): The model (can be wrapped)

    Returns:
        the loss scalar for this micro-batch
        the number of non-padded tokens in this microbatch
        a dict containing reporting metrics on the loss and number of tokens across
            the data parallel ranks
    """
    args = get_args()

    if args.logits_load_dir is not None:
        # Offline knowledge distillation loss using cached teacher log-probabilities.
        loss_func_cached_logits = _build_cached_logits_loss_func(
            logprobs_dir=args.logits_load_dir,
            decode_threads=args.logits_load_decode_threads,
            prefetch_factor=args.logits_load_prefetch_factor,
            msc_prefetch_depth=args.logits_load_msc_prefetch_depth,
            kd_loss_alpha=args.logits_load_kd_loss_alpha,
            ignore_errors=args.logits_load_ignore_errors,
        )
        loss, num_tokens, report = loss_func_cached_logits(loss_mask, output_tensor, model=model)
    elif has_nvidia_modelopt and getattr(args, 'modelopt_enabled', False):  # [ModelOpt]
        loss, num_tokens, report = loss_func_modelopt(loss_mask, output_tensor, model=model)
    else:
        losses = output_tensor.view(-1).float()
        loss_mask = loss_mask.view(-1).float()
        loss = torch.sum(losses * loss_mask)

        num_tokens = loss_mask.sum().clone().detach().to(torch.int)
        report = {'lm loss': torch.cat([loss.clone().detach().view(1), num_tokens.view(1)])}

    # Check individual rank losses are not NaN prior to DP all-reduce.
    rerun_state_machine = get_rerun_state_machine()
    if args.check_for_nan_in_loss_and_grad:
        rerun_state_machine.validate_result(
            result=loss,
            rejection_func=torch.isnan,
            message="found NaN in local forward loss calculation",
            tolerance=0.0,  # forward pass calculations are determinisic
            fatal=True,
        )
        rerun_state_machine.validate_result(
            result=loss,
            rejection_func=torch.isinf,
            message="found Inf in local forward loss calculation",
            tolerance=0.0,  # forward pass calculations are determinisic
            fatal=True,
        )
    # Check for spiky loss
    if args.check_for_spiky_loss:
        rerun_state_machine.validate_result(
            result=loss,
            rejection_func=partial(
                rerun_state_machine.is_unexpectedly_large,
                threshold=SPIKY_LOSS_FACTOR,
                context="loss",
            ),
            message="Spiky loss",
            tolerance=0.0,  # forward pass calculations are determinisic
            fatal=False,
        )

    return loss, num_tokens, report


def forward_step(data_iterator, model: GPTModel, return_schedule_plan: bool = False):
    """Forward training step.

    Args:
        data_iterator : Input data iterator
        model (GPTModel): The GPT Model
        return_schedule_plan (bool): Whether to return the schedule plan instead of the output tensor
    """
    args = get_args()
    timers = get_timers()

    # Get the batch.
    timers('batch-generator', log_level=2).start()
    global stimer
    with stimer(bdata=True):
        vp_stage = get_attr_wrapped_model(model, "vp_stage")
        tokens, labels, loss_mask, attention_mask, position_ids, packed_seq_params, padding_mask = (
            get_batch(data_iterator, vp_stage)
        )
    timers('batch-generator').stop()

    with stimer:
        if return_schedule_plan:
            assert (
                args.overlap_moe_expert_parallel_comm
            ), "overlap_moe_expert_parallel_comm must be enabled to return the schedule plan"
            schedule_plan = model.build_schedule_plan(
                tokens,
                position_ids,
                attention_mask,
                labels=labels,
                loss_mask=loss_mask,
                packed_seq_params=packed_seq_params,
                padding_mask=padding_mask,
            )
            return schedule_plan, partial(loss_func, loss_mask, model=model)
        else:
            output_tensor = model(
                tokens,
                position_ids,
                attention_mask,
                labels=labels,
                loss_mask=loss_mask,
                packed_seq_params=packed_seq_params,
                padding_mask=padding_mask,
            )

    # [ModelOpt]: model is needed to access ModelOpt distillation losses
    return output_tensor, partial(loss_func, loss_mask, model=model)


# FullCudaGraphWrapper discovers this eager prologue from the existing
# forward-step callable. It is deliberately an attribute rather than another
# training API argument so non-full-iteration schedules remain unchanged.
forward_step.full_cuda_graph_batch_prepare_func = prepare_full_cuda_graph_dynamic_packed_batch


def is_dataset_built_on_rank(vp_stage=None, is_packed_sequence=False):
    args = get_args()
    config = core_transformer_config_from_args(args)
    if mpu.get_tensor_model_parallel_rank() != 0:
        return False
    elif is_packed_sequence or (
        getattr(args, 'use_varlen_dataset', False)
        and getattr(args, 'varlen_sbhd_validation', False)
    ):
        # Packed THD and SBHD validation both need padding metadata on every
        # pipeline stage so each MoE layer excludes physical padding.
        return True
    return is_first_or_last_pipeline_stage(vp_stage) or mtp_on_this_rank(
        config, ignore_virtual=False, vp_stage=vp_stage
    )


def core_gpt_dataset_config_from_args(args: Any) -> GPTDatasetConfig:
    tokenizer = build_tokenizer(args)

    # Sometimes --data-path is too long, instead we parse it from a file.
    blend: Optional[Tuple[List[str], Optional[List[float]]]]
    blend_per_split: Optional[List[Optional[Tuple[List[str], Optional[List[float]]]]]]
    blend, blend_per_split = get_blend_and_blend_per_split(args)

    sequences_per_dataset = None
    if args.per_dataset_sequences_path is not None:
        with open(args.per_dataset_sequences_path, "r") as f:
            sequences_per_dataset = json.load(f)

    data_args = {
        "random_seed": args.seed,
        "sequence_length": args.seq_length,
        "blend": blend,
        "blend_per_split": blend_per_split,
        "split": args.split,
        "multiple_validation_sets": args.multiple_validation_sets,
        "full_validation": args.full_validation,
        "num_dataset_builder_threads": args.num_dataset_builder_threads,
        "path_to_cache": args.data_cache_path,
        "mmap_bin_files": args.mmap_bin_files,
        "tokenizer": tokenizer,
        "reset_position_ids": args.reset_position_ids,
        "reset_attention_mask": args.reset_attention_mask,
        "eod_mask_loss": args.eod_mask_loss,
        "create_attention_mask": args.create_attention_mask_in_dataloader,
        "object_storage_cache_path": args.object_storage_cache_path,
        "mid_level_dataset_surplus": args.mid_level_dataset_surplus,
        "allow_ambiguous_pad_tokens": args.allow_ambiguous_pad_tokens,
        "fast_cache_load": args.dataloader_fast_cache_load,
        "sequences_per_dataset": sequences_per_dataset,
        "defer_npy_index_mmap": args.dataloader_defer_npy_index_mmap,
        "context_parallel_size": args.context_parallel_size,
        "data_parallel_size": args.data_parallel_size,
        "sequence_parallel_size": args.tensor_model_parallel_size * args.sequence_parallel,
        "dynamic_context_parallel": args.dynamic_context_parallel,
        "sft_mock_dataset_config_json": args.sft_mock_dataset_config_json,
        "varlen_mock_dataset_config_json": args.varlen_mock_dataset_config_json,
        "varlen_sbhd_validation": args.varlen_sbhd_validation,
    }

    # add FIM args to the config
    if args.fim_data:
        extra_tokens = {
            "prefix": args.fim_prefix_token,
            "middle": args.fim_middle_token,
            "suffix": args.fim_suffix_token,
            "pad": args.fim_pad_token,
            "eod": args.fim_eod_token,
        }
        data_args.update(
            {
                "fim_rate": args.fim_rate,
                "fim_spm_rate": args.fim_spm_rate,
                "fim_extra_tokens": extra_tokens,
                "fim_split_sample": args.fim_split_sample,
                "fim_fragment_rate": args.fim_fragment_rate,
                "fim_no_prefix": args.fim_no_prefix,
            }
        )
        return GPTFIMDatasetConfig(**data_args)

    return GPTDatasetConfig(**data_args)


def train_valid_test_datasets_provider(train_val_test_num_samples, vp_stage=None):
    """Build the train test and validation datasets.

    Args:
        train_val_test_num_samples : A list containing the number of samples in train test and validation.
    """
    args = get_args()

    config = core_gpt_dataset_config_from_args(args)

    is_packed_sequence = False
    if args.sft:
        if args.mock_data:
            dataset_type = MockSFTDataset
        else:
            dataset_type = SFTDataset
        is_packed_sequence = True  # SFT always uses packed sequence
    elif args.use_varlen_dataset:
        # Variable-length packed (THD) dataset, independent of --sft.
        # Reuses SFTDataset's THD/dynamic-cp packing internally but is gated
        # by its own top-level flag.
        if args.mock_data:
            dataset_type = MockVarlenDataset
        else:
            dataset_type = VarlenDataset
        # SBHD validation mode runs the non-packed pipeline; THD mode
        # is the packed-sequence path.
        is_packed_sequence = not args.varlen_sbhd_validation
    else:
        if args.mock_data:
            dataset_type = MockGPTDataset
        elif args.fim_data:
            dataset_type = GPTFIMDataset
        else:
            dataset_type = GPTDataset

    print_rank_0("> building train, validation, and test datasets for GPT ...")

    is_dataset_built = partial(
        is_dataset_built_on_rank, vp_stage=vp_stage, is_packed_sequence=is_packed_sequence
    )
    train_ds, valid_ds, test_ds = BlendedMegatronDatasetBuilder(
        dataset_type, train_val_test_num_samples, is_dataset_built, config
    ).build()

    print_rank_0("> finished creating GPT datasets ...")

    return train_ds, valid_ds, test_ds


def get_embedding_ranks(pp_ranks: List[int]):
    """Get the embedding ranks."""
    embedding_ranks = [pp_ranks[0]]
    if len(pp_ranks) > 1:
        args = get_args()
        if not args.untie_embeddings_and_output_weights:
            embedding_ranks.append(pp_ranks[-1])
        config = core_transformer_config_from_args(args)
        mtp_ranks = get_mtp_ranks(pp_ranks, config)
        embedding_ranks.extend(mtp_ranks)
    embedding_ranks = list(set(embedding_ranks))
    embedding_ranks = sorted(embedding_ranks)
    return embedding_ranks


if __name__ == "__main__":
    # Timestamp right after entering __main__ block (after all imports/library setup)
    _MAIN_ENTRY_TIME = time.time()

    # Register startup timestamps for timing report in pretrain()
    set_startup_timestamps(program_start=_PROGRAM_START_TIME, main_entry=_MAIN_ENTRY_TIME)

    # Temporary for transition to core datasets
    setattr(train_valid_test_datasets_provider, "is_distributed", True)

    # Optionally enable inprocess restart on pretrain
    pretrain, store = inprocess_restart.maybe_wrap_for_inprocess_restart(pretrain)

    args = parse_and_validate_args(
        extra_args_provider=add_modelopt_args if has_nvidia_modelopt else None,
        args_defaults={'tokenizer_type': 'GPT2BPETokenizer'},
    )
    full_config = pretrain_cfg_container_from_args(args)
    pretrain(
        full_config,
        train_valid_test_datasets_provider,
        ModelType.encoder_or_decoder,
        forward_step,
        model_provider=partial(model_provider, gpt_builder),
        store=store,
        get_embedding_ranks=get_embedding_ranks,
    )
