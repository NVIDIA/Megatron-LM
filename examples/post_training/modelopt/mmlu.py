# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

"""MMLU evaluation for Megatron-LM models.

The plugin runs a single prefill pass per batch and selects the answer as argmax over
the choice token logits at the last prompt position (lm-evaluation-harness style),
instead of autoregressively generating tokens.
"""

import argparse
import functools
import os
import sys
import warnings

import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))

import modelopt.torch.quantization as mtq
from modelopt.torch.utils.plugins import megatron_generate as _mg_plugin
from modelopt.torch.utils.plugins import megatron_mmlu
from utils import get_hf_tokenizer

# WAR for modelopt <= 0.44: megatron_prefill's logits slice is non-contiguous when sequence
# parallelism pads seq_length to a multiple of TP; broadcast_from_last_pipeline_stage asserts
# contiguity. Fixed upstream for 0.45.
_orig_broadcast = _mg_plugin.broadcast_from_last_pipeline_stage


def _broadcast_contiguous(size, dtype, tensor=None, pp_group=None):
    if tensor is not None:
        tensor = tensor.contiguous()
    return _orig_broadcast(size, dtype, tensor, pp_group)


_mg_plugin.broadcast_from_last_pipeline_stage = _broadcast_contiguous

from megatron.post_training.arguments import add_modelopt_args
from megatron.post_training.checkpointing import load_modelopt_checkpoint
from megatron.post_training.model_builder import modelopt_gpt_hybrid_builder
from megatron.post_training.utils import report_current_memory_info
from megatron.training import get_args, get_model, initialize_megatron
from megatron.training.arguments import parse_and_validate_args
from megatron.training.utils import print_rank_0, unwrap_model
from model_provider import model_provider

warnings.filterwarnings("ignore")


def add_mmlu_args(parser):
    """Add additional arguments for MMLU evaluation."""
    group = parser.add_argument_group(title="ModelOpt MMLU evaluation")
    group.add_argument(
        "--fraction",
        type=float,
        default=1.0,
        help="Fraction of MMLU test set (per subject) to evaluate on.",
    )
    group.add_argument(
        "--few-shots",
        type=int,
        default=0,
        help="Number of few-shot examples to prepend to each prompt.",
    )
    group.add_argument(
        "--mmlu-batch-size",
        type=int,
        default=1,
        help="Batch size for the batched prefill evaluation.",
    )
    group.add_argument(
        "--lower-bound",
        type=float,
        default=None,
        help="Optional accuracy threshold; the script asserts the average is above this value.",
    )
    # Kept for backward compatibility with prior MLM_EXTRA_ARGS callers. Has no effect:
    # `megatron_mmlu` already disables its progress bar on non-master ranks.
    group.add_argument("--disable-tqdm", action="store_true", help=argparse.SUPPRESS)
    group.add_argument(
        "--mmlu-dataset", type=str, default="cais/mmlu", help=argparse.SUPPRESS
    )
    add_modelopt_args(parser)
    return parser


if __name__ == "__main__":
    parse_and_validate_args(
        extra_args_provider=add_mmlu_args,
        args_defaults={
            "tokenizer_type": "HuggingFaceTokenizer",
            "no_load_rng": True,
            "no_load_optim": True,
        },
    )
    initialize_megatron()

    args = get_args()

    # Meta device initialization for ParallelLinear only works if using cpu initialization.
    # Meta device initialization is used such that models can be materialized in low-precision
    # directly when ModelOpt real quant is used. Otherwise, the model is first initialized
    # as BF16 in memory which may result in OOM and defeat the purpose of real quant.
    if args.init_model_with_meta_device:
        args.use_cpu_initialization = True
    else:
        warnings.warn(
            "--init-model-with-meta-device is not set. If you would like to resume the "
            "model in low-bit directly (low-memory initialization and skipping 16-bit), "
            "--init-model-with-meta-device must be set.",
            UserWarning,
        )

    model = get_model(
        functools.partial(model_provider, modelopt_gpt_hybrid_builder),
        wrap_with_ddp=False,
    )
    report_current_memory_info()

    # Materialize the model from meta device to gpu before loading the checkpoint.
    unwrapped_model = unwrap_model(model)[0]
    unwrapped_model.eval()
    unwrapped_model.to_empty(device="cuda")
    report_current_memory_info()

    tokenizer = get_hf_tokenizer()

    if args.load is not None:
        load_modelopt_checkpoint(
            model, strict=not args.untie_embeddings_and_output_weights
        )
        print_rank_0("Done loading checkpoint")

    # Fold the scalars into weight for speedup.
    # [TODO]: fold_weight current assumes all weight_quantizer has weight allocated;
    # however, this is not the case when share_embeddings_and_output_weights is False.
    # [TODO]: fold_weight does not support TEGroupedMLP (QuantTEColumnParallelGroupedLinear)
    # which stores per-expert weights as weight0, weight1, etc. instead of a single weight.
    has_grouped_mlp = any(
        "TEGroupedMLP" in type(m).__name__ for m in unwrapped_model.modules()
    )
    if (
        not getattr(unwrapped_model, "share_embeddings_and_output_weights", False)
        and not has_grouped_mlp
    ):
        mtq.fold_weight(unwrapped_model)

    with torch.no_grad():
        avg = megatron_mmlu(
            unwrapped_model,
            tokenizer,
            few_shots=args.few_shots,
            fraction=args.fraction,
            batch_size=args.mmlu_batch_size,
        )

    if torch.distributed.get_rank() == 0 and args.lower_bound is not None:
        assert avg > args.lower_bound, (
            f"MMLU accuracy {avg:.4f} below lower bound {args.lower_bound}"
        )
