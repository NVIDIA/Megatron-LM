# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import logging
from argparse import ArgumentParser
from functools import partial
from typing import Optional

from gpt_builders import gpt_builder
from mamba_builders import mamba_builder
from megatron.core.inference.config import InferenceConfig, MambaInferenceStateConfig
from megatron.core.inference.contexts import DynamicInferenceContext
from megatron.core.inference.engines import DynamicInferenceEngine
from megatron.core.inference.model_inference_wrappers.gpt.gpt_inference_wrapper import (
    GPTInferenceWrapper,
)
from megatron.core.inference.text_generation_controllers.text_generation_controller import (
    TextGenerationController,
)
from megatron.core.tokenizers.utils.build_tokenizer import build_tokenizer
from megatron.core.transformer.module import MegatronModule
from megatron.core.utils import get_attr_wrapped_model, log_single_rank
from megatron.training import get_args
from megatron.training import get_model as _get_model
from megatron.training import get_tokenizer, get_wandb_writer
from megatron.training.checkpointing import load_checkpoint
from model_provider import model_provider

logger = logging.getLogger(__name__)


def get_model_for_inference() -> MegatronModule:
    """Initialize model and load checkpoint for inference."""

    args = get_args()

    if args.model_provider == "gpt":
        model_builder = gpt_builder
    elif args.model_provider == "mamba":
        model_builder = mamba_builder
    else:
        raise ValueError(f"Invalid model provider {args.model_provider}")

    # Build model.
    model = _get_model(partial(model_provider, model_builder), wrap_with_ddp=False)

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
        "--model-provider", choices=["mamba", "gpt"], default="gpt", help="Model provider"
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
        "`suspend_resume_interval` steps. This is used to tet the suspend/resume "
        "system.",
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

    return parser


def get_inference_config_from_model_and_args(model: MegatronModule, args):
    """Returns a `InferenceConfig` constructed from the model and command line arguments."""

    # Max sequence length.
    position_embedding_type = get_attr_wrapped_model(model, "position_embedding_type")
    model_max_seq_len = get_attr_wrapped_model(model, "max_sequence_length")
    inf_max_seq_len = args.inference_max_seq_length
    max_batch_size = args.inference_dynamic_batching_max_requests

    if position_embedding_type == "learned_absolute":
        # When using absolute position embeddings, it is critical that the
        # context's `max_sequence_length` is less than or equal to the model's
        # `max_sequence_length`. Otherwise, the context's `position_ids` will
        # contain ids greater than the dimension of the position embedding
        # tensor, which will result in an index error.
        if inf_max_seq_len:
            max_sequence_length = min(model_max_seq_len, inf_max_seq_len)
        else:
            max_sequence_length = model_max_seq_len
        assert max_batch_size is None or max_batch_size <= model_max_seq_len
    else:
        max_sequence_length = inf_max_seq_len
    if args.inference_dynamic_batching_max_requests is not None:
        max_sequence_length = max(max_sequence_length, max_batch_size)

    mamba_inference_state_config = MambaInferenceStateConfig.from_model(model)
    pg_collection = get_attr_wrapped_model(model, "pg_collection")

    # Get inference logging configuration from args
    log_inference_wandb = args.inference_wandb_logging
    inference_logging_step_interval = args.inference_logging_step_interval

    # Get metrics writer if logging is enabled and on the logging rank
    # Use the same rank convention as training (last rank logs)
    metrics_writer = None
    if (
        inference_logging_step_interval > 0
        and log_inference_wandb
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

    return InferenceConfig(
        block_size_tokens=args.inference_dynamic_batching_block_size,
        buffer_size_gb=args.inference_dynamic_batching_buffer_size_gb,
        paused_buffer_size_gb=args.inference_dynamic_batching_paused_buffer_size_gb,
        mamba_memory_ratio=args.inference_dynamic_batching_mamba_memory_ratio,
        num_cuda_graphs=(
            args.inference_dynamic_batching_num_cuda_graphs
            if args.cuda_graph_impl == "local"
            else None
        ),
        max_requests=args.inference_dynamic_batching_max_requests,
        max_tokens=args.inference_dynamic_batching_max_tokens,
        unified_memory_level=args.inference_dynamic_batching_unified_memory_level,
        offload_kv_cache=args.rl_offload_kv_cache_during_training,
        cuda_graph_mixed_prefill_count=args.inference_dynamic_batching_cuda_graph_mixed_prefill_count,  # pylint: disable=line-too-long
        use_cuda_graphs_for_non_decode_steps=not args.decode_only_cuda_graphs,
        persist_cuda_graphs=args.rl_training_cuda_graphs,
        max_sequence_length=max_sequence_length,
        mamba_inference_state_config=mamba_inference_state_config,
        pg_collection=pg_collection,
        use_flashinfer_fused_rope=args.use_flashinfer_fused_rope,
        materialize_only_last_token_logits=not args.return_log_probs,
        track_paused_request_events=args.inference_dynamic_batching_track_paused_request_events,
        enable_chunked_prefill=args.enable_chunked_prefill,
        metrics_writer=metrics_writer,
        logging_step_interval=args.inference_logging_step_interval,
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
