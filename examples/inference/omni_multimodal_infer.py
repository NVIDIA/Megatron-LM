# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Minimal Nemotron Omni text+image/video inference through MegatronAsyncLLM.

This is intentionally a debugging example rather than a benchmark. It loads the
Megatron-Bridge Nemotron Omni model, creates mock media in memory, and submits
one multimodal request through the high-level dynamic-inference API.

Launch with any number of GPUs. Each GPU hosts one complete data-parallel model replica:

    torchrun --standalone --nproc-per-node=<num-gpus> \
      examples/inference/omni_multimodal_infer.py \
      --hf-model nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16 \
      --graph-mode decode

Use ``--graph-mode off`` as the eager baseline and ``--graph-mode all`` to
include multimodal prefill steps in CUDA-graph selection.

Requires pip installation of Megatron-Bridge for the model provider, and
at least 2 GPUs with around 40GB of memory or higher!
"""

import argparse
import asyncio
import base64
import io
import json
import logging
import time
import urllib.error
import urllib.request
from typing import Any

import torch
import torch.distributed as dist
from PIL import Image, ImageDraw

from megatron.bridge import AutoBridge
from megatron.core.inference.apis import MegatronAsyncLLM, MegatronLLM
from megatron.core.inference.apis.serve_config import ServeConfig
from megatron.core.inference.config import (
    ImageProcessingConfig,
    InferenceConfig,
    MambaInferenceStateConfig,
    VideoProcessingConfig,
)
from megatron.core.inference.model_inference_wrappers.multimodal.nemotron_omni_inference_wrapper import (
    NemotronOmniInferenceWrapper,
)
from megatron.core.inference.sampling_params import SamplingParams
from megatron.core.tokenizers.text.text_tokenizer import MegatronTokenizerText


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--hf-model",
        default="nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16",
        help="HF model ID or local HF checkpoint directory.",
    )
    parser.add_argument(
        "--megatron-checkpoint",
        default=None,
        help="Optional converted Megatron checkpoint. Without it, Bridge converts HF weights.",
    )
    parser.add_argument(
        "--api",
        choices=("async", "sync", "completions"),
        default="async",
        help="Submit directly or through the OpenAI-compatible completions endpoint.",
    )
    parser.add_argument(
        "--graph-mode",
        choices=("off", "decode", "all"),
        default="off",
        help="CUDA graphs off, decode-only, or decode+prefill.",
    )
    parser.add_argument(
        "--cuda-graph-scope",
        choices=("layer", "block", "none"),
        default="none",
        help="Megatron local CUDA-graph capture granularity.",
    )
    parser.add_argument("--num-cuda-graphs", type=int, default=-1)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--max-sequence-length", type=int, default=2048)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--kv-cache-gb", type=int, default=4)
    parser.add_argument(
        "--media",
        choices=("image", "video"),
        default="image",
        help="Send either the generated mock PNG or mock MP4.",
    )
    parser.add_argument("--image-max-patches", type=int, default=128)
    parser.add_argument(
        "--video-num-frames",
        type=int,
        default=8,
        help="Number of frames generated and sampled for mock video input.",
    )
    parser.add_argument("--coordinator-port", type=int, default=50055)
    parser.add_argument("--http-host", default="127.0.0.1")
    parser.add_argument("--http-port", type=int, default=5000)
    parser.add_argument(
        "--prompt",
        default="Describe the visual content, including colors, shapes, text, and any motion.",
    )
    return parser.parse_args()


def make_mock_png() -> bytes:
    """Create a deterministic image without downloading a dataset."""
    image = Image.new("RGB", (384, 256), color=(235, 240, 248))
    draw = ImageDraw.Draw(image)
    draw.rectangle((30, 35, 170, 185), fill=(32, 112, 220), outline="black", width=4)
    draw.ellipse((215, 45, 350, 180), fill=(242, 92, 84), outline="black", width=4)
    draw.text((95, 215), "NEMOTRON OMNI", fill=(10, 10, 10))
    stream = io.BytesIO()
    image.save(stream, format="PNG")
    return stream.getvalue()


def make_mock_mp4(num_frames: int) -> bytes:
    """Create a deterministic moving-shape video without downloading a dataset."""
    if num_frames <= 0:
        raise ValueError("--video-num-frames must be positive.")

    try:
        import av  # type: ignore[import-not-found]
    except ImportError as error:
        raise RuntimeError("Mock video generation requires PyAV (`pip install av`).") from error

    output = io.BytesIO()
    container = av.open(output, mode="w", format="mp4")
    video_stream = container.add_stream("mpeg4", rate=4)
    video_stream.width = 384
    video_stream.height = 256
    video_stream.pix_fmt = "yuv420p"

    for frame_index in range(num_frames):
        image = Image.new("RGB", (384, 256), color=(235, 240, 248))
        draw = ImageDraw.Draw(image)
        progress = frame_index / max(num_frames - 1, 1)
        x = 25 + round(progress * 230)
        draw.rectangle((x, 55, x + 100, 155), fill=(32, 112, 220), outline="black", width=4)
        draw.ellipse((260 - x // 3, 165, 330 - x // 3, 235), fill=(242, 92, 84))
        draw.text((20, 15), f"NEMOTRON OMNI - FRAME {frame_index + 1}", fill=(10, 10, 10))
        frame = av.VideoFrame.from_image(image)
        for packet in video_stream.encode(frame):
            container.mux(packet)

    for packet in video_stream.encode():
        container.mux(packet)
    container.close()
    return output.getvalue()


def configure_provider(provider: Any, args: argparse.Namespace) -> None:
    """Use TP2/EP2 across both ranks and apply the requested graph mode."""
    provider.tensor_model_parallel_size = 2
    provider.pipeline_model_parallel_size = 1
    provider.expert_model_parallel_size = 2
    provider.expert_tensor_parallel_size = 1
    provider.sequence_parallel = True
    provider.pipeline_dtype = torch.bfloat16
    provider.dynamic_resolution = True
    provider.temporal_patch_dim = 1
    provider.separate_video_embedder = False
    provider.temporal_ckpt_compat = False
    provider.vision_class_token_len = 10
    provider.cuda_graph_impl = "none" if args.graph_mode == "off" else "local"
    provider.inference_cuda_graph_scope = (
        "none" if args.graph_mode == "off" else args.cuda_graph_scope
    )
    provider.moe_pad_experts_for_cuda_graph_inference = args.graph_mode != "off"


def model_overrides(args: argparse.Namespace) -> dict:
    """Return overrides needed when loading an already converted checkpoint."""
    return {
        "tensor_model_parallel_size": 2,
        "pipeline_model_parallel_size": 1,
        "expert_model_parallel_size": 2,
        "expert_tensor_parallel_size": 1,
        "sequence_parallel": True,
        "pipeline_dtype": torch.bfloat16,
        "dynamic_resolution": True,
        "temporal_patch_dim": 1,
        "separate_video_embedder": False,
        "temporal_ckpt_compat": False,
        "vision_class_token_len": 10,
        "cuda_graph_impl": "none" if args.graph_mode == "off" else "local",
        "inference_cuda_graph_scope": (
            "none" if args.graph_mode == "off" else args.cuda_graph_scope
        ),
    }


def load_model(args: argparse.Namespace):
    """Load/convert and return the canonical Bridge NemotronOmniModel."""
    bridge = AutoBridge.from_hf_pretrained(args.hf_model, trust_remote_code=True)
    provider = bridge.to_megatron_provider(load_weights=args.megatron_checkpoint is None)
    configure_provider(provider, args)
    provider.initialize_model_parallel(
        seed=1234,
        seed_kwargs={"inference_rng_tracker": True},
    )

    if args.megatron_checkpoint:
        distributed_models = bridge.load_megatron_model(
            args.megatron_checkpoint,
            mp_overrides=model_overrides(args),
            wrap_with_ddp=False,
        )
    else:
        provider.finalize()
        distributed_models = provider.provide_distributed_model(wrap_with_ddp=False)

    if len(distributed_models) != 1:
        raise RuntimeError(
            "This example requires pipeline_model_parallel_size=1 and one local model chunk."
        )

    model = distributed_models[0].cuda().bfloat16().eval()
    model = model.module if hasattr(model, "module") else model

    # Training checkpoints may retain a bound optimizer loss scaler.
    model.config.grad_scale_func = None
    model.language_model.config.grad_scale_func = None
    return model


def build_tokenizer(model_path: str) -> MegatronTokenizerText:
    return MegatronTokenizerText(
        model_path,
        {"library": "huggingface"},
        trust_remote_code=True,
        use_fast=True,
        include_special_tokens=True,
    )


def build_prompt_tokens(tokenizer, model, user_prompt: str) -> list[int]:
    content = f"<img><image></img>\n{user_prompt}"
    prompt = tokenizer.apply_chat_template(
        [
            {"role": "system", "content": "/no_think"},
            {"role": "user", "content": content},
        ],
        tokenize=False,
        add_generation_prompt=True,
    )
    tokens = tokenizer.tokenize(prompt)
    marker_count = tokens.count(int(model.image_token_index))
    if marker_count != 1:
        raise RuntimeError(
            f"Expected one image marker token ({model.image_token_index}), got {marker_count}. "
            "Check that the HF tokenizer matches the checkpoint."
        )
    return tokens


def build_inference_config(model, args: argparse.Namespace) -> InferenceConfig:
    image_config = ImageProcessingConfig(
        patch_dim=int(model.patch_dim),
        dynamic_resolution=True,
        use_tiling=False,
        pixel_shuffle=True,
        spatial_merge_size=1,
        dynamic_resolution_min_patches=16,
        dynamic_resolution_max_patches=args.image_max_patches,
        vision_model_type="radio",
    )
    return InferenceConfig(
        block_size_tokens=256,
        buffer_size_gb=args.kv_cache_gb,
        max_requests=2,
        max_tokens=args.max_tokens,
        max_sequence_length=args.max_sequence_length,
        mamba_inference_state_config=MambaInferenceStateConfig.from_model(
            model.language_model
        ),
        pg_collection=model.pg_collection,
        num_cuda_graphs=(
            None if args.graph_mode == "off" else args.num_cuda_graphs
        ),
        use_cuda_graphs_for_non_decode_steps=args.graph_mode == "all",
        cuda_graph_max_tokens=min(args.max_tokens, 512),
        image_preprocessing_config=image_config,
        video_preprocessing_config=VideoProcessingConfig(
            image_config=image_config,
            num_frames=args.video_num_frames,
            temporal_patch_size=int(
                getattr(model.vision_model, "temporal_patch_dim", 1)
            ),
        ),
    )


def print_result(result) -> None:
    print_generated_text(result.generated_text)


def print_generated_text(text: str) -> None:
    if dist.get_rank() != 0:
        return
    print("\n======== NEMOTRON OMNI OUTPUT ========")
    print(text)
    print("======================================")


async def run_async(args, model, tokenizer, config, prompt_tokens, media_bytes) -> None:
    async with MegatronAsyncLLM(
        model=model,
        tokenizer=tokenizer,
        inference_config=config,
        use_coordinator=True,
        coordinator_port=args.coordinator_port,
        inference_wrapper_cls=NemotronOmniInferenceWrapper,
    ) as llm:
        if llm.is_primary_rank:
            result = await llm.generate(
                prompt_tokens,
                SamplingParams(
                    temperature=1.0,
                    top_k=1,
                    num_tokens_to_generate=args.max_new_tokens,
                    termination_id=tokenizer.eod,
                    skip_prompt_log_probs=True,
                ),
                multi_modal_data={args.media: media_bytes},
            )
            torch.cuda.synchronize()
            print_result(result)


def post_completion(args, prompt_tokens, media_bytes) -> dict:
    """POST one request to the local OpenAI-compatible completions endpoint."""
    payload = json.dumps(
        {
            "prompt": prompt_tokens,
            "temperature": 1.0,
            "top_k": 1,
            "top_p": 0.0,
            "max_tokens": args.max_new_tokens,
            "multi_modal_data": {
                args.media: base64.b64encode(media_bytes).decode("ascii")
            },
        }
    ).encode("utf-8")
    request = urllib.request.Request(
        f"http://{args.http_host}:{args.http_port}/v1/completions",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    # The listening socket is bound before serve() returns, but the frontend
    # worker may still be importing dependencies when the first POST arrives.
    for attempt in range(50):
        try:
            with urllib.request.urlopen(request, timeout=300) as response:
                return json.loads(response.read())
        except urllib.error.HTTPError as error:
            details = error.read().decode("utf-8", errors="replace")
            raise RuntimeError(
                f"Completions endpoint returned HTTP {error.code}: {details}"
            ) from error
        except urllib.error.URLError:
            if attempt == 49:
                raise
            time.sleep(0.1)
    raise RuntimeError("Completions endpoint did not become ready.")


async def run_completions(
    args, model, tokenizer, config, prompt_tokens, media_bytes
) -> None:
    # Lifecycle synchronization must not use the default NCCL group while the
    # background engine is running. A worker blocked in an NCCL barrier cannot
    # participate in the TP/EP collectives required by rank 0's HTTP request.
    lifecycle_group = dist.new_group(backend="gloo")
    try:
        async with MegatronAsyncLLM(
            model=model,
            tokenizer=tokenizer,
            inference_config=config,
            use_coordinator=True,
            coordinator_port=args.coordinator_port,
            inference_wrapper_cls=NemotronOmniInferenceWrapper,
        ) as llm:
            await llm.serve(
                ServeConfig(
                    host=args.http_host,
                    port=args.http_port,
                    frontend_replicas=1,
                ),
                blocking=False,
            )
            dist.barrier(group=lifecycle_group)
            if llm.is_primary_rank:
                response = await asyncio.to_thread(
                    post_completion, args, prompt_tokens, media_bytes
                )
                print_generated_text(response["choices"][0]["text"])
            dist.barrier(group=lifecycle_group)
    finally:
        dist.destroy_process_group(lifecycle_group)


def run_sync(args, model, tokenizer, config, prompt_tokens, media_bytes) -> None:
    with MegatronLLM(
        model=model,
        tokenizer=tokenizer,
        inference_config=config,
        use_coordinator=True,
        coordinator_port=args.coordinator_port,
        inference_wrapper_cls=NemotronOmniInferenceWrapper,
    ) as llm:
        if llm.is_primary_rank:
            result = llm.generate(
                prompt_tokens,
                SamplingParams(
                    temperature=1.0,
                    top_k=1,
                    num_tokens_to_generate=args.max_new_tokens,
                    termination_id=tokenizer.eod,
                    skip_prompt_log_probs=True,
                ),
                multi_modal_data={args.media: media_bytes},
            )[0]
            torch.cuda.synchronize()
            print_result(result)


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO)
    model = load_model(args)
    tokenizer = build_tokenizer(args.hf_model)
    prompt_tokens = build_prompt_tokens(tokenizer, model, args.prompt)
    media_bytes = (
        make_mock_mp4(args.video_num_frames)
        if args.media == "video"
        else make_mock_png()
    )
    config = build_inference_config(model, args)

    if dist.get_rank() == 0:
        print(
            f"dp_replicas={dist.get_world_size()}, api={args.api}, media={args.media}, "
            f"graph_mode={args.graph_mode}, "
            f"graph_scope={args.cuda_graph_scope}, prompt_tokens={len(prompt_tokens)}"
        )

    if args.api == "async":
        asyncio.run(run_async(args, model, tokenizer, config, prompt_tokens, media_bytes))
    elif args.api == "completions":
        asyncio.run(
            run_completions(args, model, tokenizer, config, prompt_tokens, media_bytes)
        )
    else:
        run_sync(args, model, tokenizer, config, prompt_tokens, media_bytes)


if __name__ == "__main__":
    main()
