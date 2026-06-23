# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import logging
import warnings
from argparse import ArgumentParser, Namespace
from typing import Literal, Optional

import torch

from megatron.core.inference.contexts import DynamicInferenceContext
from megatron.core.inference.engines import DynamicInferenceEngine
from megatron.core.inference.model_inference_wrappers.gpt.gpt_inference_wrapper import (
    GPTInferenceWrapper,
)
from megatron.core.inference.quantization.utils import quantize_model_to_mxfp8
from megatron.core.inference.text_generation_controllers.text_generation_controller import (
    TextGenerationController,
)
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.tokenizers.utils.build_tokenizer import build_tokenizer
from megatron.core.transformer.enums import InferenceCudaGraphScope
from megatron.core.transformer.module import MegatronModule
from megatron.core.utils import log_single_rank, unwrap_model
from megatron.training import get_args
from megatron.training import get_model as _get_model
from megatron.training import get_tokenizer, get_wandb_writer
from megatron.training.argument_utils import gpt_config_from_args, hybrid_config_from_args
from megatron.training.checkpointing import load_checkpoint
from megatron.training.models import GPTModelBuilder, HybridModelBuilder, ModelBuilder

try:
    from megatron.post_training.model_builder import modelopt_gpt_hybrid_builder

    HAS_NVIDIA_MODELOPT = True
except ImportError:
    HAS_NVIDIA_MODELOPT = False

logger = logging.getLogger(__name__)


def get_model_builder(
    args: Namespace, provider: Optional[Literal["gpt", "hybrid", "mamba"]] = None
) -> ModelBuilder:
    """Construct a :class:`ModelBuilder` for the requested model provider.

    Replaces the legacy ``gpt_builder`` / ``hybrid_builder`` function selector with
    a config-driven dispatch that returns a fully-configured :class:`ModelBuilder`
    instance whose ``build_model()`` and ``build_distributed_models()`` methods can
    be used to materialize the model.

    Args:
        args: The parsed argparse namespace, used to populate the model config via
            ``gpt_config_from_args`` / ``hybrid_config_from_args``.
        provider: Optional override for the model provider name. Must be one of
            ``"gpt"``, ``"hybrid"``, or the deprecated ``"mamba"``. When omitted,
            falls back to ``args.model_provider`` (set by ``add_inference_args``).

    Returns:
        A :class:`ModelBuilder` instance bound to a config derived from ``args``.
    """
    if provider is None:
        provider = args.model_provider
    if provider == "gpt":
        return GPTModelBuilder(gpt_config_from_args(args))
    if provider in ("hybrid", "mamba"):
        if provider == "mamba":
            warnings.warn(
                '"mamba" model provider is deprecated. Use "hybrid" instead.',
                DeprecationWarning,
                stacklevel=2,
            )
        return HybridModelBuilder(hybrid_config_from_args(args))
    raise ValueError(f"Invalid model provider {provider}")


def get_model_for_inference() -> MegatronModule:
    """Initialize model and load checkpoint for inference."""

    args = get_args()

    if HAS_NVIDIA_MODELOPT and getattr(args, "modelopt_enabled", False):
        # ModelOpt path keeps the legacy callable-based builder because the
        # modelopt hooks (custom layer specs, calibration, etc.) have not been
        # ported to the new ``ModelBuilder`` API yet. ``_get_model`` also takes
        # care of running the modelopt-checkpoint auto-detection side effect.
        model = _get_model(modelopt_gpt_hybrid_builder, wrap_with_ddp=False)
    else:
        builder = get_model_builder(args)
        pg_collection = ProcessGroupCollection.use_mpu_process_groups()
        model = builder.build_distributed_models(
            pg_collection=pg_collection, wrap_with_ddp=False
        )

    # Load checkpoint.
    assert args.load is not None
    args.exit_on_missing_checkpoint = True
    load_checkpoint(
        ddp_model=model,
        optimizer=None,
        opt_param_scheduler=None,
        strict=not args.inference_ckpt_non_strict,
    )

    # No virtual PP.
    assert len(model) == 1, "Above condition should have caught this"
    model = model[0]

    # Eval mode.
    model.eval()

    if args.transformer_impl == "inference_optimized" and args.fp8_recipe == "mxfp8":
        backend = args.inference_grouped_gemm_backend
        if backend == "auto" or backend == "torch":
            quant_backend = "triton"
        elif backend == "te":
            raise ValueError(
                "MXFP8 quantization is not supported with "
                "inference_grouped_gemm_backend='te'."
            )
        quantize_model_to_mxfp8(unwrap_model(model), backend=quant_backend)
    return model


def add_inference_args(parser: ArgumentParser) -> ArgumentParser:
    """Add inference command line arguments to the parser."""

    group = parser.add_argument_group(title='Inference')

    group.add_argument("--temperature", type=float, default=1.0, help='Sampling temperature.')
    group.add_argument("--top_k", type=int, default=1, help='Top k sampling.')
    group.add_argument("--top_p", type=float, default=0.0, help='Top p sampling.')
    group.add_argument(
        "--return-log-probs",
        action='store_true',
        default=False,
        help='Return the log probabilities of the final output tokens',
    )
    group.add_argument(
        "--prompts",
        metavar='N',
        type=str,
        nargs='+',
        help='Input prompts with each prompt within quotes and separated by space',
    )
    group.add_argument(
        "--num-tokens-to-prompt",
        type=int,
        nargs="+",
        default=[64, 1024],
        help='Number of tokens to use for simulated prompts. This should be a '
        'space-separated pair of integers, and the generated prompt lengths will '
        'be uniformly sampled within this range.',
    )
    group.add_argument(
        "--num-tokens-to-generate",
        type=int,
        default=30,
        help='Number of tokens to generate for each prompt',
    )
    group.add_argument(
        "--num-tokens-from-file",
        action='store_true',
        default=False,
        help='Use per-prompt num_tokens_to_generate from prompt file',
    )
    group.add_argument(
        "--top-n-logprobs",
        type=int,
        default=0,
        help=(
            "Return the top n logprobs for the generated tokens and their "
            "corresponding token as a dictionary"
        ),
    )
    group.add_argument(
        "--incoming-requests-per-step",
        type=int,
        default=None,
        help="Add a deterministic number of requests per step. This arg is "
        "prioritized over `--incoming-requests-per-sec` below (which is non-"
        "deterministic). Note that the number of requests added per step is "
        "additionally limited by the inference context's `max_requests`, "
        "`max_tokens`, and KV buffer size.",
    )
    group.add_argument(
        "--incoming-requests-per-sec",
        type=float,
        default=100.0,
        help="Simulated number of requests per second. Set to -1 to add all requests together.",
    )
    group.add_argument(
        "--incoming-requests-duration",
        type=float,
        default=10.0,
        help="Total amount of time to simulate that requests are "
        "arriving. Multiply this value with "
        "`--incoming-requests-per-sec` to get the approximate "
        "total number of requests. Set to -1 to add all requests together.",
    )
    group.add_argument(
        "--model-provider",
        choices=["hybrid", "mamba", "gpt"],
        default="gpt",
        help='Model provider. Use "hybrid" for HybridModel (formerly MambaModel). '
        '"mamba" is accepted for backward compatibility but deprecated.',
    )
    group.add_argument(
        "--skip-prompt-log-probs", action='store_true', default=False, help='Skip prompt log probs.'
    )
    group.add_argument(
        "--stop-words",
        metavar='WORD',
        type=str,
        nargs='+',
        default=None,
        help='Stop words to terminate generation. Each word should be quoted and '
        'separated by space. Example: --stop-words "\\n\\n" "END" "###"',
    )
    group.add_argument(
        "--output-path", type=str, default=None, help="Path to save generations as JSON"
    )
    group.add_argument(
        "--output-every-n-results",
        type=int,
        default=1,
        help="To minimize the output file size of larger runs, only write the "
        "results of every `n` requests.",
    )
    group.add_argument(
        "--output-request-events",
        action='store_true',
        default=False,
        help="Include request events (lifecycle + per-token block allocator metrics) "
        "in the JSON output.",
    )
    group.add_argument(
        "--prompt-file",
        help='Jsonl file containing input prompts, where each item (i.e., line) '
        'contains the field \'text\' where the value is the prompt. All other '
        'fields within each item are ignored, and may be customized for each '
        'application.',
    )
    group.add_argument(
        "--prompt-file-num-truncate",
        type=int,
        help='Number of samples to use from the loaded prompt file (see '
        '`--prompt-file` above). The first `--prompt-file-num-truncate` samples '
        'will be used, in order.',
    )
    group.add_argument(
        "--use-flashinfer-fused-rope",
        action='store_true',
        default=False,
        help='Use flashinfer fused rope implementation.',
    )
    group.add_argument(
        "--no-record-throughput",
        action='store_false',
        dest="record_throughput",
        help="Disable throughput recording in --output-file",
    )
    group.add_argument(
        "--inference-ckpt-non-strict",
        action="store_true",
        help="Load checkpoint with `strict=False`.",
    )
    group.add_argument(
        "--termination-id",
        type=int,
        default=None,
        help="Termination ID that overrides `tokenizer.eod`.",
    )
    group.add_argument(
        "--suspend-resume-interval",
        type=int,
        default=None,
        help="Suspend and resume the dynamic engine every "
        "`suspend_resume_interval` requests. This is used to test the suspend/resume "
        "system.",
    )
    group.add_argument(
        "--suspend-timeout",
        type=float,
        default=0.0,
        help="Seconds to sleep while the engine is suspended (simulates a training step).",
    )
    group.add_argument(
        "--inference-repeat-n",
        type=int,
        default=1,
        help="Repeat inference iterations N times for benchmarking.",
    )
    group.add_argument(
        "--throughput-check-only",
        action='store_true',
        default=False,
        help="If true, only run throughput check without verifying outputs.",
    )
    group.add_argument(
        "--drain-between-batches",
        action='store_true',
        default=False,
        help="Process requests in batches, draining all active requests between batches.",
    )
    group.add_argument(
        "--batch-boundaries",
        type=str,
        default=None,
        help="Comma-separated list of request indices where each batch starts. "
        "Used with --drain-between-batches.",
    )
    group.add_argument(
        "--coordinator-schedule-output-path",
        type=str,
        default=None,
        help="Path to write coordinator request scheduling decisions as JSON",
    )
    return parser


def get_inference_config_from_model_and_args(model: MegatronModule, args):
    """Returns an `InferenceConfig` constructed from the model and command line arguments.

    Delegates to ``InferenceSetupConfig.to_inference_config`` so the declarative
    ``InferenceSetupConfig`` (built from args) is the single source of truth for translating
    inference args into the runtime engine ``InferenceConfig``.
    """
    from megatron.training.argument_utils import inference_cfg_from_args

    # Get metrics writer if logging is enabled and on the logging rank.
    # Use the same rank convention as training (last rank logs).
    metrics_writer = None
    if (
        args.inference_logging_step_interval > 0
        and args.inference_wandb_logging
        and args.rank == (args.world_size - 1)
    ):
        metrics_writer = get_wandb_writer()
        if metrics_writer is None:
            log_single_rank(
                logger,
                logging.WARNING,
                "WARNING: --rl-inference-logging-step-interval is set but no metrics writer "
                "wandb module is available. Inference logging will be disabled.",
            )

    # Only kwargs that are NOT in the inference argument group are passed explicitly.
    # The rest (return_log_probs, skip_prompt_log_probs, use_flashinfer_fused_rope, and the
    # inference_dynamic_batching_* / prefix_caching_* / chunked_prefill knobs) live on
    # InferenceSetupConfig and are read from ``self`` inside ``to_inference_config``.
    setup_cfg = inference_cfg_from_args(args)
    return setup_cfg.to_inference_config(
        model,
        kv_cache_management_mode=args.rl_kv_cache_management_mode,
        static_kv_memory_pointers=args.rl_persist_cuda_graphs,
        enable_cuda_graphs=(args.inference_cuda_graph_scope != InferenceCudaGraphScope.none),
        metrics_writer=metrics_writer,
    )


def get_dynamic_inference_engine(model: Optional[MegatronModule] = None) -> DynamicInferenceEngine:
    """Builds a `DynamicInferenceEngine`."""
    args = get_args()
    if model is None:
        model = get_model_for_inference()
    tokenizer = build_tokenizer(args)

    inference_config = get_inference_config_from_model_and_args(model, args)
    context = DynamicInferenceContext(model.config, inference_config)
    inference_wrapped_model = GPTInferenceWrapper(model, context)
    controller = TextGenerationController(inference_wrapped_model, tokenizer)
    engine = DynamicInferenceEngine(controller, context)
    return engine
