# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

import argparse
import asyncio
import os
import sys

# tools/ lives at the repo root; put the repo root on sys.path so the
# megatron.* and examples.* packages are importable regardless of cwd.
# Also append examples/multimodal because examples/multimodal/model.py and
# its siblings use bare imports like `from config import ...`.
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
_EXAMPLES_MULTIMODAL = os.path.join(_REPO_ROOT, "examples", "multimodal")
if _EXAMPLES_MULTIMODAL not in sys.path:
    sys.path.append(_EXAMPLES_MULTIMODAL)

import torch  # noqa: E402

from examples.multimodal.multimodal_args import add_multimodal_extra_args  # noqa: E402
from megatron.core.inference.config import (  # noqa: E402
    ImageProcessingConfig,
    VideoProcessingConfig,
)
from megatron.core.inference.contexts.dynamic_context import DynamicInferenceContext  # noqa: E402
from megatron.core.inference.engines import DynamicInferenceEngine  # noqa: E402
from megatron.core.inference.model_inference_wrappers.multimodal.vlm_inference_wrapper import (  # noqa: E402,E501
    VLMInferenceWrapper,
)
from megatron.core.inference.text_generation_controllers.text_generation_controller import (  # noqa: E402,E501
    TextGenerationController,
)
from megatron.core.inference.text_generation_server.dynamic_text_gen_server import (  # noqa: E402
    start_text_gen_server,
    stop_text_gen_server,
)
from megatron.core.inference.text_generation_server.dynamic_text_gen_server.vlm_dynamic_inference import (  # noqa: E402,E501
    _detect_vlm_from_checkpoint,
    _print_resolved_args,
    add_vlm_inference_args,
    get_model as get_vlm_model,
)
from megatron.core.tokenizers.utils.build_tokenizer import build_tokenizer  # noqa: E402
from megatron.core.utils import configure_nvtx_profiling, trace_async_exceptions  # noqa: E402
from megatron.inference.utils import (  # noqa: E402
    get_dynamic_inference_engine,
    get_inference_config_from_model_and_args,
)
from megatron.post_training.arguments import add_modelopt_args  # noqa: E402
from megatron.training import get_args  # noqa: E402
from megatron.training.arguments import parse_and_validate_args  # noqa: E402
from megatron.training.initialize import initialize_megatron  # noqa: E402


def add_text_generation_server_args(parser: argparse.ArgumentParser):
    """Adds the required command line arguments for running the text generation server."""
    parser = add_modelopt_args(parser)
    # add_vlm_inference_args calls add_inference_args internally; don't double-add.
    parser = add_vlm_inference_args(parser)
    parser = add_multimodal_extra_args(parser)
    parser.add_argument("--port", type=int, default=5000, help="Port for Flask server to run on")
    parser.add_argument(
        "--host",
        type=str,
        default=None,
        help="Hostname or IP address to bind the server to. Defaults to 0.0.0.0 (all interfaces).",
    )
    parser.add_argument(
        "--parsers", type=str, nargs="+", default=[], help="Parsers to use for parsing the response"
    )
    # NOTE: --chat-template is already declared by upstream's TrainingConfig
    # (megatron/training/config/training_config.py); we don't re-register it.
    # The chat_completions endpoint reads it from args.chat_template via
    # _load_chat_template, which accepts either a file path or an inline string.
    parser.add_argument(
        "--default-temperature",
        type=float,
        default=1.0,
        help="Default temperature sampling value when a request does not specify temperature.",
    )
    parser.add_argument(
        "--default-top-p",
        type=float,
        default=1.0,
        help="Default top-p sampling value when a request does not specify top_p.",
    )
    parser.add_argument(
        "--default-top-k",
        type=int,
        default=0,
        help="Default top-k sampling value when a request does not specify top_k.",
    )
    parser.add_argument(
        "--eval-mode",
        action="store_true",
        help=(
            "Optimize defaults for pure serving. In chat requests, prevent_retokenization "
            "defaults to false so prompt token IDs are not returned."
        ),
    )
    return parser


def _build_engine_for_vlm_or_gpt(is_vlm: bool) -> DynamicInferenceEngine:
    """Build a DynamicInferenceEngine, wrapping with VLMInferenceWrapper when needed.

    The default ``get_dynamic_inference_engine`` only knows about GPT/Hybrid
    backbones; for VLM checkpoints we have to build the LLaVA-wrapped model
    ourselves and wrap it with ``VLMInferenceWrapper`` so the engine's forward
    path consumes image embeddings.
    """
    args = get_args()

    if not is_vlm:
        return get_dynamic_inference_engine()

    tokenizer = build_tokenizer(args)
    model = get_vlm_model(is_vlm=True)
    inference_config = get_inference_config_from_model_and_args(model, args)

    # Grow inference_config.max_sequence_length to accommodate the worst-case
    # image-expanded prompt, matching vlm_server.py's pre-engine bookkeeping.
    args.num_img_embeddings_per_tile = 0
    if hasattr(args, 'patch_dim'):
        dynamic_res = (
            getattr(args, 'dynamic_resolution', False)
            and not getattr(args, 'use_tiling', False)
        )
        if dynamic_res:
            max_patches = getattr(args, 'dynamic_resolution_max_patches', 128)
            max_img_embeddings = max_patches
            if getattr(args, 'pixel_shuffle', False):
                max_img_embeddings = max_img_embeddings // 4
            inference_config.max_sequence_length = max(
                inference_config.max_sequence_length,
                max_img_embeddings + args.num_tokens_to_generate + 512,
            )
        else:
            from megatron.core.models.vision.clip_vit_model import get_num_image_embeddings
            args.num_img_embeddings_per_tile = get_num_image_embeddings(
                args.img_h,
                args.img_w,
                args.patch_dim,
                args.vision_model_type,
                args.disable_vision_class_token,
                1,
                args.pixel_shuffle,
                args.use_tile_tags,
                args.max_num_tiles,
                args.tokenizer_prompt_format,
            )
            max_num_tiles = args.max_num_tiles + int(getattr(args, 'use_thumbnail', False))
            max_img_tokens = max_num_tiles * args.num_img_embeddings_per_tile
            inference_config.max_sequence_length = max(
                inference_config.max_sequence_length,
                max_img_tokens + args.num_tokens_to_generate + 512,
            )

    inference_config.image_preprocessing_config = ImageProcessingConfig(
        patch_dim=args.patch_dim,
        dynamic_resolution=getattr(args, 'dynamic_resolution', False),
        use_tiling=getattr(args, 'use_tiling', False),
        pixel_shuffle=getattr(args, 'pixel_shuffle', False),
        spatial_merge_size=getattr(args, 'spatial_merge_size', 1),
        dynamic_resolution_min_patches=getattr(
            args, 'dynamic_resolution_min_patches', 1
        ),
        dynamic_resolution_max_patches=getattr(
            args, 'dynamic_resolution_max_patches', 128
        ),
        vision_model_type=getattr(args, 'vision_model_type', 'radio'),
        pixel_mean=getattr(args, 'pixel_mean', None),
        pixel_std=getattr(args, 'pixel_std', None),
        img_h=getattr(args, 'img_h', None),
        img_w=getattr(args, 'img_w', None),
        max_num_tiles=getattr(args, 'max_num_tiles', 1),
        use_thumbnail=getattr(args, 'use_thumbnail', False),
        num_img_embeddings_per_tile=args.num_img_embeddings_per_tile,
    )
    inference_config.video_preprocessing_config = VideoProcessingConfig(
        image_config=inference_config.image_preprocessing_config,
        num_frames=int(getattr(args, "num_frames", 8)),
        temporal_patch_size=int(
            getattr(
                model,
                "temporal_patch_dim",
                getattr(getattr(model, "vision_model", None), "temporal_patch_dim", 1),
            )
        ),
    )

    context = DynamicInferenceContext(model.config, inference_config)
    wrapped_model = VLMInferenceWrapper(model, context)
    controller = TextGenerationController(wrapped_model, tokenizer)
    return DynamicInferenceEngine(controller, context)


@trace_async_exceptions
async def run_text_generation_server(
    engine: DynamicInferenceEngine,
    coordinator_port: int,
    server_port: int,
    hostname: str | None = None,
    chat_template: str | None = None,
    default_temperature: float = 1.0,
    default_top_p: float = 1.0,
    default_top_k: int = 0,
    eval_mode: bool = False,
):
    """
    Runs the text generation server from rank 0 and initializes the
    DynamicInferenceEngine on all ranks.

    Args:
        engine (DynamicInferenceEngine): The dynamic inference engine.
        coordinator_port (int): The network port for the dynamic inference DP coordinator.
        server_port (int): The network for port the frontend text generation server.
        hostname (str | None): Hostname or IP address for coordinator and HTTP traffic.
        chat_template (str | None): Inline chat template or contents loaded from a file.
        default_temperature (float): Sampling default when a request omits `temperature`.
        default_top_p (float): Sampling default when a request omits `top_p`.
        default_top_k (int): Sampling default when a request omits `top_k`.
        eval_mode (bool): Whether to use evaluation response defaults.
    """

    rank = torch.distributed.get_rank()

    coordinator_addr = await engine.start_listening_to_data_parallel_coordinator(
        inference_coordinator_port=coordinator_port,
        launch_inference_coordinator=True,
        hostname=hostname,
    )

    try:
        if rank == 0:
            start_text_gen_server(
                coordinator_addr=coordinator_addr,
                tokenizer=engine.controller.tokenizer,
                parsers=args.parsers,
                rank=rank,
                server_port=server_port,
                verbose=args.inference_text_gen_server_logging,
                hostname=hostname,
                chat_template=chat_template,
                multimodal_prompt_config=(
                    engine.controller.inference_wrapped_model.multimodal_prompt_config
                ),
                default_temperature=default_temperature,
                default_top_p=default_top_p,
                default_top_k=default_top_k,
                eval_mode=eval_mode,
            )

        # Await the engine loop directly since the server is running in a separate process
        await engine.engine_loop_task

    finally:
        # Guarantee that the separate process is terminated when the engine loop stops or is interrupted
        if rank == 0:
            stop_text_gen_server()


def _load_chat_template(value):
    """Resolve a --chat-template arg into the template string itself.

    If the value is a path to an existing file, read it. Otherwise treat the
    value as the inline template.
    """
    if value is None:
        return None
    if os.path.isfile(value):
        with open(value) as f:
            return f.read()
    return value


if __name__ == "__main__":
    with torch.inference_mode():
        os.environ.setdefault("CUDA_DEVICE_MAX_CONNECTIONS", "1")

        # Snapshot what the user actually typed BEFORE we inject defaults, so
        # _detect_vlm_from_checkpoint can tell explicit CLI args from injected
        # defaults / parser defaults.  Precedence: CLI > checkpoint > default.
        user_passed_attrs = set()
        for tok in sys.argv[1:]:
            if tok.startswith('--'):
                name = tok[2:].split('=', 1)[0]
                user_passed_attrs.add(name.replace('-', '_'))

        # Defaults that align this server with the VLM dynamic-batching path.
        # Injected into argv (not as parser defaults) so they appear *before*
        # any user-provided value. Value-taking args are overridden by later
        # explicit CLI occurrences, so the general form of "later wins" holds;
        # store_true flags have no negating counterpart, so we only inject
        # those when the user hasn't set a conflicting explicit value.
        _defaults = [
            "--micro-batch-size", "1",
            "--inference-dynamic-batching-buffer-size-gb", "2.0",
            # Placeholders for add_multimodal_extra_args' required args. These
            # are injected as defaults, so _detect_vlm_from_checkpoint will
            # replace them with the checkpoint's real values when loading a VLM.
            "--language-model-type", "placeholder",
            "--tokenizer-prompt-format", "mistral",
        ]
        # store_true flags: only inject when the user hasn't expressed a
        # conflicting choice on the CLI.
        if "fp16" not in user_passed_attrs and "bf16" not in user_passed_attrs:
            _defaults.append("--bf16")
        if "use_checkpoint_args" not in user_passed_attrs:
            _defaults.append("--use-checkpoint-args")
        if "inference_dynamic_batching" not in user_passed_attrs:
            _defaults.append("--inference-dynamic-batching")
        # Materialize logits for every prompt position (not just the last) so
        # prompt log-probs can be computed for lm-eval / MCQ likelihood scoring.
        if "return_log_probs" not in user_passed_attrs:
            _defaults.append("--return-log-probs")
        # Avoid running prefill through CUDA graphs: under a graphed prefill,
        # is_decode_only() returns True and calculate_log_probs short-circuits
        # to a single logprob per request, breaking logprob-based eval.
        if "decode_only_cuda_graphs" not in user_passed_attrs:
            _defaults.append("--decode-only-cuda-graphs")
        sys.argv[1:1] = _defaults

        parse_and_validate_args(
            extra_args_provider=add_text_generation_server_args,
            args_defaults={'no_load_rng': True, 'no_load_optim': True},
        )
        initialize_megatron()

        args = get_args()

        # Auto-detect VLM and copy VLM args from the checkpoint with precedence
        # CLI > checkpoint > parser default.  Tiling and dynamic_resolution are
        # mutually exclusive at inference, so honor --use-tiling explicitly.
        is_vlm = _detect_vlm_from_checkpoint(args, user_passed_attrs=user_passed_attrs)
        if getattr(args, 'use_tiling', False):
            args.dynamic_resolution = False
        if is_vlm:
            _print_resolved_args("resolved VLM arguments", args)

        if torch.distributed.get_rank() == 0:
            print(f"Auto-detected model type: {'VLM' if is_vlm else 'GPT'}")

        # Match training's NVTX gating (training.py only flips this when both
        # --profile and --nvtx-ranges are set). Otherwise the engine-side
        # nvtx_range_push labels (bookkeeping, Decode, _ep_establish_consensus,
        # etc.) are no-ops and the inter-step gap is unattributable in nsys.
        if args.profile and args.nvtx_ranges:
            configure_nvtx_profiling(True)

        # Already requested via --return-log-probs default above; keep this
        # explicit assignment for clarity / belt-and-braces with the engine.
        args.return_log_probs = True

        chat_template = _load_chat_template(getattr(args, 'chat_template', None))

        engine = _build_engine_for_vlm_or_gpt(is_vlm=is_vlm)

        try:
            asyncio.run(
                run_text_generation_server(
                    engine,
                    args.inference_coordinator_port,
                    args.port,
                    args.host,
                    chat_template=chat_template,
                    default_temperature=args.default_temperature,
                    default_top_p=args.default_top_p,
                    default_top_k=args.default_top_k,
                    eval_mode=args.eval_mode,
                )
            )
        except KeyboardInterrupt:
            # Catching at the top level ensures clean stdout without spamming the traceback
            print("Server process interrupted by user.")
        finally:
            # Clean up PyTorch distributed groups properly
            if torch.distributed.is_initialized():
                torch.distributed.destroy_process_group()
