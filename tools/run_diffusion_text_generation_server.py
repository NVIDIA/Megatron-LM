# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Text generation server for two-tower diffusion models.

Launch via ``torchrun``::

    CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc-per-node=2 \\
        -m tools.run_diffusion_text_generation_server \\
        --tt-diffusion-model-provider mamba_two_tower \\
        --tt-diffusion-steps-per-block 32 \\
        --use-checkpoint-args \\
        --load /path/to/checkpoint \\
        --bf16 \\
        --port 5000 \\
        --temperature 0.1

Endpoints:
    ``POST /completions``  — generate text or compute loglikelihoods.
    ``POST /api``          — legacy Megatron text generation API.
"""

import os
import sys
from contextlib import nullcontext
from functools import partial

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))

import torch

from megatron.core import mpu
from megatron.core.inference.text_generation_server import MegatronServer
from megatron.core.inference.text_generation_server.run_mcore_engine import run_mcore_engine
from megatron.diffusion.two_tower.arguments import add_two_tower_diffusion_args
from megatron.diffusion.two_tower.inference_engine import DiffusionEngine
from megatron.training import get_args, get_model, get_tokenizer, print_rank_0
from megatron.training.checkpointing import load_checkpoint
from megatron.training.initialize import initialize_megatron

try:
    from megatron.post_training.arguments import add_modelopt_args

    _HAS_MODELOPT = True
except ImportError:
    _HAS_MODELOPT = False


def _get_model_builder(args):
    """Return the appropriate model builder and provider for *args*."""
    from megatron.diffusion.two_tower.builder import two_tower_mamba_builder
    from model_provider import model_provider

    if args.tt_diffusion_model_provider == "mamba_two_tower":
        return partial(model_provider, two_tower_mamba_builder)
    elif args.tt_diffusion_model_provider == "mamba":
        from pretrain_mamba import model_provider as mamba_model_provider

        return mamba_model_provider
    else:
        raise ValueError(
            f"Invalid --tt-diffusion-model-provider: {args.tt_diffusion_model_provider}. "
            "Expected 'mamba' or 'mamba_two_tower'."
        )


def load_model(args):
    """Load model based on ``args.tt_diffusion_model_provider``.

    Supports native two-tower checkpoints and single-tower conversion
    via ``--tt-diffusion-load-single-tower``.
    """
    load_context = nullcontext()
    if getattr(args, 'fp8', False):
        from transformer_engine.pytorch.fp8 import fp8_model_init

        load_context = fp8_model_init()

    provider = _get_model_builder(args)

    if args.tt_diffusion_model_provider == "mamba_two_tower" and getattr(
        args, 'tt_diffusion_load_single_tower', False
    ):
        from megatron.diffusion.two_tower.builder import two_tower_mamba_builder
        from model_provider import model_provider

        print_rank_0("Loading two-tower from single-tower checkpoint...")

        # Build single-tower, load checkpoint, extract state dict
        from pretrain_mamba import model_provider as mamba_model_provider

        with load_context:
            single_tower = get_model(mamba_model_provider, wrap_with_ddp=False)
        load_checkpoint(
            ddp_model=single_tower,
            optimizer=None,
            opt_param_scheduler=None,
            strict=not getattr(args, 'inference_ckpt_non_strict', False),
        )
        single_state = {
            k: (v.cpu() if v is not None else v) for k, v in single_tower[0].state_dict().items()
        }
        del single_tower
        torch.cuda.empty_cache()

        args.tt_diffusion_load_single_tower = False
        with load_context:
            model = get_model(partial(model_provider, two_tower_mamba_builder), wrap_with_ddp=False)
        inner_model = model[0]
        if hasattr(inner_model, 'module'):
            inner_model = inner_model.module
        missing, unexpected = inner_model.load_from_single_tower(single_state, strict=False)
        print_rank_0(
            f"Loaded from single-tower: {len(single_state)} keys, "
            f"{len(missing)} missing, {len(unexpected)} unexpected"
        )
        del single_state
    else:
        with load_context:
            model = get_model(provider, wrap_with_ddp=False)
        if args.load is not None:
            load_checkpoint(
                ddp_model=model,
                optimizer=None,
                opt_param_scheduler=None,
                strict=not getattr(args, 'inference_ckpt_non_strict', False),
            )

    assert len(model) == 1, "Expected single model"
    model = model[0]
    model.eval()

    if getattr(args, 'tt_diffusion_context_ar_generation', False):
        inner = model.module if hasattr(model, 'module') else model
        inner._single_tower_mode = True
        del inner.denoiser_tower
        del inner.denoiser_embedding
        torch.cuda.empty_cache()
        print_rank_0("Context-AR generation: denoiser deleted, using context tower only.")

    return model


def add_base_inference_args(parser):
    """Add inference arguments to *parser*.

    Sampling args (temperature, top-k, etc.) are redefined here from
    ``megatron.inference.utils.add_inference_args`` which is not called
    by ``initialize_megatron``.
    """
    # Sampling args mirrored from megatron.inference.utils.add_inference_args
    group = parser.add_argument_group(title='base inference')
    group.add_argument("--temperature", type=float, default=1.0, help='Sampling temperature.')
    group.add_argument("--top_k", type=int, default=1, help='Top-k sampling.')
    group.add_argument("--top_p", type=float, default=0.0, help='Top-p sampling.')
    group.add_argument(
        "--return-log-probs",
        action='store_true',
        default=False,
        help='Return log probabilities of output tokens.',
    )
    group.add_argument(
        "--num-tokens-to-generate",
        type=int,
        default=30,
        help='Number of tokens to generate per prompt.',
    )
    return parser


def add_tt_diffusion_inference_args(parser):
    group = parser.add_argument_group(title='tt diffusion inference')
    group.add_argument(
        "--tt-diffusion-model-provider",
        type=str,
        default="mamba_two_tower",
        choices=["mamba", "mamba_two_tower"],
        help='Model provider: mamba (single-tower) or mamba_two_tower.',
    )
    group.add_argument(
        "--tt-diffusion-steps-per-block",
        type=int,
        default=1,
        help='Denoising steps per block for diffusion generation.',
    )
    group.add_argument(
        "--tt-diffusion-sampling-strategy",
        type=str,
        default="predict_and_noise",
        choices=["predict_and_noise", "posterior", "confidence_unmasking"],
        help='Sampling strategy for mask diffusion.',
    )
    group.add_argument(
        "--tt-diffusion-noise-schedule",
        type=str,
        default="linear",
        choices=["linear", "cosine", "exponential"],
        help='Noise schedule for the diffusion reverse process.',
    )
    group.add_argument(
        "--tt-diffusion-confidence-threshold",
        type=float,
        default=1e6,
        help='Confidence threshold for confidence_unmasking strategy.',
    )
    group.add_argument(
        "--tt-diffusion-posterior-float64",
        action="store_true",
        default=False,
        help='Use float64 precision for posterior sampling computation.',
    )
    group.add_argument(
        "--tt-diffusion-load-single-tower",
        action="store_true",
        default=False,
        help='Load a single-tower MambaModel checkpoint into the two-tower '
        'architecture for autoregressive generation. The denoiser tower is '
        'bypassed entirely; generation uses the context tower only. This is '
        'an evaluation baseline equivalent to standard AR serving.',
    )
    group.add_argument(
        "--tt-diffusion-context-ar-generation",
        action="store_true",
        default=False,
        help='Force context-tower-only autoregressive generation on a '
        'two-tower checkpoint. The denoiser tower is bypassed entirely. '
        'Useful for evaluating the AR quality of the context tower when '
        'trained with --tt-diffusion-context-ar-loss.',
    )
    return parser


def add_extra_args_provider(parser):
    parser.add_argument("--port", type=int, default=5000, help='Server port.')
    parser = add_two_tower_diffusion_args(parser)
    parser = add_base_inference_args(parser)
    parser = add_tt_diffusion_inference_args(parser)
    if _HAS_MODELOPT:
        parser = add_modelopt_args(parser)
    return parser


@torch.inference_mode()
def main():
    """Run the text generation server with a two-tower diffusion model."""
    initialize_megatron(
        extra_args_provider=add_extra_args_provider,
        args_defaults={
            'no_load_rng': True,
            'no_load_optim': True,
            'exit_on_missing_checkpoint': True,
        },
    )
    args = get_args()

    ckpt_step = getattr(args, "ckpt_step", None)
    print_rank_0(f"Checkpoint: {args.load}, " f"ckpt_step: {ckpt_step or 'latest'}")

    if args.num_layers_per_virtual_pipeline_stage is not None:
        print("Interleaved pipeline schedule is not yet supported for text generation.")
        sys.exit(1)

    args.exit_on_missing_checkpoint = True
    model = load_model(args)

    tokenizer = get_tokenizer()

    block_size = getattr(args, 'tt_diffusion_block_size', None) or getattr(args, 'block_size', 64)
    steps_per_block = args.tt_diffusion_steps_per_block
    sampling_strategy = args.tt_diffusion_sampling_strategy
    posterior_float64 = args.tt_diffusion_posterior_float64
    noise_schedule = args.tt_diffusion_noise_schedule
    confidence_threshold = args.tt_diffusion_confidence_threshold

    print_rank_0(
        f"DiffusionEngine: block_size={block_size}, steps_per_block={steps_per_block}, "
        f"strategy={sampling_strategy}, "
        f"float64={posterior_float64}, schedule={noise_schedule}, "
        f"confidence_threshold={confidence_threshold}"
    )

    inference_engine = DiffusionEngine(
        model=model,
        tokenizer=tokenizer,
        block_size=block_size,
        steps_per_block=steps_per_block,
        sampling_strategy=sampling_strategy,
        posterior_float64=posterior_float64,
        noise_schedule=noise_schedule,
        confidence_threshold=confidence_threshold,
    )

    if (
        mpu.is_pipeline_first_stage()
        and mpu.get_tensor_model_parallel_rank() == 0
        and mpu.get_expert_model_parallel_rank() == 0
    ):
        print_rank_0(f"Starting MegatronServer on port {args.port}...")
        server = MegatronServer(inference_engine, args)

        from flask import request as flask_request

        @server.app.before_request
        def _inject_sampling_defaults():
            req = flask_request.get_json(silent=True)
            if req is not None:
                req['temperature'] = args.temperature
                req['top_k'] = args.top_k
                req['top_p'] = args.top_p
                req['do_sample'] = args.temperature > 0.0

        server.run("0.0.0.0", port=args.port)

    while True:
        choice = torch.tensor([1], dtype=torch.long, device='cuda')
        torch.distributed.broadcast(choice, 0)
        if choice[0].item() == 0:
            try:
                run_mcore_engine(inference_engine)
            except ValueError:
                pass
        elif choice[0].item() == 1:
            break


if __name__ == "__main__":
    main()
