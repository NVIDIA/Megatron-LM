# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

"""Pretrain using an HLModel config with optional CLI overrides.

Usage:
    # Use defaults
    torchrun --nproc_per_node=8 pretrain_hl.py \
        --hlmodel-path examples/ablations.py

    # Override a single knob
    torchrun --nproc_per_node=8 pretrain_hl.py \
        --hlmodel-path examples/ablations.py \
        --override hidden_size=4096

    # Shell-level sweep
    for hs in 1024 2048 4096; do
        torchrun --nproc_per_node=8 pretrain_hl.py \
            --hlmodel-path examples/ablations.py \
            --override hidden_size=$hs
    done

The HLModel config contract:
    - The module MUST define a `build_model(**overrides)` function that returns
      an HLModelConfig.
    - It SHOULD define a `DEFAULTS` dict so `--list-overrides` can show
      available knobs and their defaults.
"""

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

from functools import partial
from typing import List, Optional, Tuple

import torch

from megatron.core import mpu
from megatron.core.datasets.blended_megatron_dataset_builder import BlendedMegatronDatasetBuilder
from megatron.core.datasets.gpt_dataset import GPTDataset, GPTDatasetConfig, MockGPTDataset
from megatron.core.enums import ModelType
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.rerun_state_machine import get_rerun_state_machine
from megatron.core.tokenizers.text.utils.build_tokenizer import build_tokenizer
from megatron.core.models.hl import HLModelConfig
from megatron.core.utils import get_attr_wrapped_model, is_te_min_version, StragglerDetector
from megatron.training import (
    get_args,
    get_timers,
    get_tokenizer,
    inprocess_restart,
    pretrain,
    print_rank_0,
    set_startup_timestamps,
)
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

try:
    import transformer_engine  # pylint: disable=unused-import
    import transformer_engine_torch as tex
except ImportError:
    tex = None

stimer = StragglerDetector()


# =============================================================================
# HLMODEL CONFIG LOADING
# =============================================================================


def _coerce_value(value_str: str):
    """Best-effort conversion of a CLI string to a Python value.

    Tries (in order): int, float, bool literals, then falls back to str.
    """
    try:
        return int(value_str)
    except ValueError:
        pass

    try:
        return float(value_str)
    except ValueError:
        pass

    if value_str.lower() in ("true", "yes"):
        return True
    if value_str.lower() in ("false", "no"):
        return False

    return value_str


def parse_overrides(override_args: list[str]) -> dict:
    """Parse a list of 'key=value' strings into a dict with coerced types."""
    overrides = {}
    for item in override_args:
        if "=" not in item:
            raise ValueError(
                f"Invalid override '{item}'. Expected format: key=value"
            )
        key, value_str = item.split("=", 1)
        overrides[key] = _coerce_value(value_str)
    return overrides


def load_hlmodel_config(config_path: str):
    """Dynamically import a Python HLModel config file as a module."""
    import importlib.util
    from pathlib import Path

    path = Path(config_path).resolve()
    if not path.exists():
        raise FileNotFoundError(f"HLModel config file not found: {path}")

    return HLModelConfig.from_path(path)


# =============================================================================
# HLMODEL BUILDER
# =============================================================================


def hlmodel_builder(hlmodel_config, args, pre_process, post_process, vp_stage=None, config=None, pg_collection=None):
    """Build a model from an HLModelConfig.

    Same signature as mamba_builder / gpt_builder so it can be passed to
    model_provider from model_provider.py.
    """
    print_rank_0('building HLModel ...')
    model = hlmodel_config.build()
    print_rank_0(f'HLModel built: {model}')
    return model


# =============================================================================
# BATCH / LOSS / FORWARD (same as pretrain_mamba.py)
# =============================================================================


def get_batch(data_iterator, vp_stage=None):
    """Generate a batch."""

    empty_batch = {
        'tokens': None,
        'labels': None,
        'loss_mask': None,
        'attention_mask': None,
        'position_ids': None,
        'cu_seqlens': None,
        'max_seqlen': None,
    }

    is_packed_sequence = get_args().sft
    if not is_first_or_last_pipeline_stage(vp_stage) and not is_packed_sequence:
        return empty_batch.values()

    batch = get_batch_on_this_tp_rank(data_iterator)

    cu_seqlens = batch['cu_seqlens']
    cu_seqlens_padded = batch.pop('cu_seqlens_padded', None)
    local_cp_size = batch.pop('local_cp_size', None)

    if cu_seqlens is not None:
        assert (
            cu_seqlens.dim() == 2 and cu_seqlens.shape[0] == 1
        ), "micro-batch-size must be 1 for packing"
        cu_seqlens = cu_seqlens[0]
        batch['cu_seqlens'] = cu_seqlens

        max_seqlen = batch['max_seqlen']
        assert max_seqlen.dim() == 1
        batch['max_seqlen'] = int(max_seqlen[0].item())

    if mpu.is_pipeline_first_stage(ignore_virtual=(vp_stage is None), vp_stage=vp_stage):
        total_tokens = batch['tokens'].size(1)
    elif mpu.is_pipeline_last_stage(ignore_virtual=(vp_stage is None), vp_stage=vp_stage):
        total_tokens = batch['labels'].size(1)
    else:  # packed sequence
        empty_batch['cu_seqlens'] = cu_seqlens
        empty_batch['max_seqlen'] = max_seqlen
        return empty_batch.values()

    if cu_seqlens is None:
        batch = get_batch_on_this_cp_rank(batch)
    else:  # Packed THD format
        cp_size = mpu.get_context_parallel_world_size()
        if cp_size > 1:
            assert tex is not None and is_te_min_version("1.10.0"), (
                "Please update Transformer Engine to >= 1.10 to use "
                "Context Parallel with THD format data"
            )
            cp_rank = mpu.get_context_parallel_rank()
            index = tex.thd_get_partitioned_indices(
                cu_seqlens,
                total_tokens,
                cp_size,
                cp_rank,
            )
            for key, data in batch.items():
                if key in {'attention_mask', 'cu_seqlens', 'max_seqlen'}:
                    continue
                if data is not None:
                    batch[key] = data.index_select(1, index)

    return batch.values()


SPIKY_LOSS_FACTOR = 10


def loss_func(loss_mask: torch.Tensor, output_tensor: torch.Tensor, model=None):
    """Loss function."""
    args = get_args()
    if has_nvidia_modelopt and getattr(args, 'modelopt_enabled', False):
        loss, num_tokens, report = loss_func_modelopt(loss_mask, output_tensor, model=model)
    else:
        losses = output_tensor.view(-1).float()
        loss_mask = loss_mask.view(-1).float()
        loss = torch.sum(losses * loss_mask)

        num_tokens = loss_mask.sum().clone().detach().to(torch.int)
        report = {'lm loss': torch.cat([loss.clone().detach().view(1), num_tokens.view(1)])}

    rerun_state_machine = get_rerun_state_machine()
    if args.check_for_nan_in_loss_and_grad:
        rerun_state_machine.validate_result(
            result=loss,
            rejection_func=torch.isnan,
            message="found NaN in local forward loss calculation",
            tolerance=0.0,
            fatal=True,
        )
        rerun_state_machine.validate_result(
            result=loss,
            rejection_func=torch.isinf,
            message="found Inf in local forward loss calculation",
            tolerance=0.0,
            fatal=True,
        )
    if args.check_for_spiky_loss:
        rerun_state_machine.validate_result(
            result=loss,
            rejection_func=partial(
                rerun_state_machine.is_unexpectedly_large,
                threshold=SPIKY_LOSS_FACTOR,
                context="loss",
            ),
            message="Spiky loss",
            tolerance=0.0,
            fatal=False,
        )

    return loss, num_tokens, report


def forward_step(data_iterator, model):
    """Forward training step."""
    timers = get_timers()

    timers('batch-generator', log_level=2).start()

    global stimer

    with stimer(bdata=True):
        vp_stage = get_attr_wrapped_model(model, "vp_stage")
        (
            tokens,
            labels,
            loss_mask,
            attention_mask,
            position_ids,
            cu_seqlens,
            max_seqlen,
        ) = get_batch(data_iterator, vp_stage)

    if cu_seqlens is None:
        packed_seq_params = None
    else:
        packed_seq_params = PackedSeqParams(
            qkv_format="thd",
            cu_seqlens_q=cu_seqlens,
            cu_seqlens_kv=cu_seqlens,
            cu_seqlens_q_padded=None,
            cu_seqlens_kv_padded=None,
            max_seqlen_q=max_seqlen,
            max_seqlen_kv=max_seqlen,
        )

    timers('batch-generator').stop()

    with stimer:
        output_tensor = model(
            tokens,
            position_ids,
            attention_mask,
            labels=labels,
            packed_seq_params=packed_seq_params,
        )

    return output_tensor, partial(loss_func, loss_mask, model=model)


# =============================================================================
# DATASETS
# =============================================================================


def is_dataset_built_on_rank(vp_stage=None, is_packed_sequence=False):
    if mpu.get_tensor_model_parallel_rank() != 0:
        return False
    elif is_packed_sequence:
        return True
    else:
        return is_first_or_last_pipeline_stage(vp_stage)


def core_gpt_dataset_config_from_args(args):
    if args.legacy_tokenizer:
        tokenizer = get_tokenizer()
    else:
        tokenizer = build_tokenizer(args)

    blend: Optional[Tuple[List[str], Optional[List[float]]]]
    blend_per_split: Optional[List[Optional[Tuple[List[str], Optional[List[float]]]]]]
    blend, blend_per_split = get_blend_and_blend_per_split(args)

    sequences_per_dataset = None
    if args.per_dataset_sequences_path is not None:
        with open(args.per_dataset_sequences_path, "r") as f:
            sequences_per_dataset = json.load(f)

    return GPTDatasetConfig(
        random_seed=args.seed,
        sequence_length=args.seq_length,
        blend=blend,
        blend_per_split=blend_per_split,
        split=args.split,
        num_dataset_builder_threads=args.num_dataset_builder_threads,
        path_to_cache=args.data_cache_path,
        mmap_bin_files=args.mmap_bin_files,
        tokenizer=tokenizer,
        reset_position_ids=args.reset_position_ids,
        reset_attention_mask=args.reset_attention_mask,
        eod_mask_loss=args.eod_mask_loss,
        create_attention_mask=args.create_attention_mask_in_dataloader,
        object_storage_cache_path=args.object_storage_cache_path,
        mid_level_dataset_surplus=args.mid_level_dataset_surplus,
        allow_ambiguous_pad_tokens=args.allow_ambiguous_pad_tokens,
        fast_cache_load=args.dataloader_fast_cache_load,
        sequences_per_dataset=sequences_per_dataset,
        defer_npy_index_mmap=args.dataloader_defer_npy_index_mmap,
        context_parallel_size=args.context_parallel_size,
    )


def train_valid_test_datasets_provider(train_val_test_num_samples, vp_stage=None):
    """Build the train test and validation datasets."""
    args = get_args()
    config = core_gpt_dataset_config_from_args(args)

    is_packed_sequence = False
    if args.sft:
        from megatron.training.datasets.sft_dataset import SFTDataset

        dataset_type = SFTDataset
        is_packed_sequence = True
    else:
        if args.mock_data:
            dataset_type = MockGPTDataset
        else:
            dataset_type = GPTDataset

    print_rank_0("> building train, validation, and test datasets ...")

    train_ds, valid_ds, test_ds = BlendedMegatronDatasetBuilder(
        dataset_type,
        train_val_test_num_samples,
        partial(is_dataset_built_on_rank, vp_stage=vp_stage, is_packed_sequence=is_packed_sequence),
        config,
    ).build()

    print_rank_0("> finished creating datasets ...")

    return train_ds, valid_ds, test_ds


# =============================================================================
# MAIN
# =============================================================================


def add_hlmodel_args(parser):
    """Add --hlmodel-path and --override to the Megatron argument parser."""
    group = parser.add_argument_group(title='hlmodel')
    group.add_argument(
        '--hlmodel-path',
        type=str,
        required=True,
        help='Path to a Python HLModel config file that defines build_model().',
    )
    group.add_argument(
        '--override',
        nargs='*',
        default=[],
        metavar='KEY=VALUE',
        help='Override HLModel config values. Example: --override hidden_size=4096',
    )
    group.add_argument(
        '--list-overrides',
        action='store_true',
        help='Print available override keys and their defaults, then exit.',
    )
    return parser


if __name__ == "__main__":
    _MAIN_ENTRY_TIME = time.time()
    set_startup_timestamps(program_start=_PROGRAM_START_TIME, main_entry=_MAIN_ENTRY_TIME)

    # Parse --hlmodel-path and --override early (before Megatron consumes args)
    import sys

    hlmodel_path = None
    overrides_raw = []
    list_overrides = False

    # Extract our args without disturbing Megatron's parser
    i = 1
    while i < len(sys.argv):
        if sys.argv[i] == '--hlmodel-path' and i + 1 < len(sys.argv):
            hlmodel_path = sys.argv[i + 1]
            i += 2
        elif sys.argv[i] == '--override':
            i += 1
            while i < len(sys.argv) and not sys.argv[i].startswith('--'):
                overrides_raw.append(sys.argv[i])
                i += 1
        elif sys.argv[i] == '--list-overrides':
            list_overrides = True
            i += 1
        else:
            i += 1

    if hlmodel_path is None:
        print("Error: --hlmodel-path is required", file=sys.stderr)
        sys.exit(1)

    # Load the HLModel config module
    hlmodel_config = load_hlmodel_config(hlmodel_path)

    if list_overrides:
        defaults = getattr(hlmodel_config, "DEFAULTS", None)
        if defaults is None:
            print("This HLModel config does not expose a DEFAULTS dict.")
        else:
            print("Available overrides (key = default):")
            for key, value in sorted(defaults.items()):
                print(f"  {key} = {value!r}")
        sys.exit(0)

    overrides = parse_overrides(overrides_raw)
    if overrides:
        print_rank_0(f"Applying HLModel overrides: {overrides}")

    # Build the HLModelConfig (not the model itself — that happens in the provider)
    hlmodel_config = hlmodel_config.build(**overrides)
    print_rank_0(f"HLModelConfig: {hlmodel_config}")

    # Temporary for transition to core datasets
    train_valid_test_datasets_provider.is_distributed = True

    # Optionally enable inprocess restart
    pretrain_fn, store = inprocess_restart.maybe_wrap_for_inprocess_restart(pretrain)

    pretrain_fn(
        train_valid_test_datasets_provider,
        partial(model_provider, partial(hlmodel_builder, hlmodel_config)),
        ModelType.encoder_or_decoder,
        forward_step,
        args_defaults={'tokenizer_type': 'GPT2BPETokenizer'},
        store=store,
        extra_args_provider=add_modelopt_args if has_nvidia_modelopt else None,
    )
