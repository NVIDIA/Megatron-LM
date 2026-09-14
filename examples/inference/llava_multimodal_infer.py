# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""End-to-end LLaVAModel image/video inference through Megatron Inference.

Unlike the toy unit tests, this script loads real Nemotron Omni weights into the
legacy MCore ``LLaVAModel`` contract, runs the real vision encoder, injects the
projected media embeddings through ``VLMInferenceWrapper``, and executes the real
language-model decoder.

Launch on an even number of GPUs (TP2/EP2, with remaining replicas used for DP):

    torchrun --standalone --nproc-per-node=4 \
      examples/inference/llava_multimodal_infer.py \
      --hf-model nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16 \
      --media image

Use ``--media video`` to exercise frame decoding and video placeholder expansion.
This is a correctness/debugging example, not a benchmark. The legacy LLaVA
checkpoint contract is intentionally selected even though the canonical Omni
model now uses an expanded-sequence implementation.
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
from functools import partial
from typing import Any

import torch
import torch.distributed as dist
from PIL import Image, ImageDraw

from megatron.bridge import AutoBridge
from megatron.bridge.models.nemotron_omni.nemotron_omni_bridge import (
    NemotronOmniLlavaBridge,
)
from megatron.core.inference.apis import MegatronAsyncLLM, MegatronLLM
from megatron.core.inference.apis.serve_config import ServeConfig
from megatron.core.inference.config import (
    ImageProcessingConfig,
    InferenceConfig,
    MambaInferenceStateConfig,
    VideoProcessingConfig,
)
from megatron.core.inference.model_inference_wrappers.multimodal.vlm_inference_wrapper import (
    VLMInferenceWrapper,
)
from megatron.core.inference.sampling_params import SamplingParams
from megatron.core.models.multimodal.llava_model import LLaVAModel
from megatron.core.tokenizers.text.text_tokenizer import MegatronTokenizerText


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--hf-model",
        default="nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16",
        help="HF model ID or local HF checkpoint directory.",
    )
    parser.add_argument("--api", choices=("async", "sync", "completions"), default="async")
    parser.add_argument("--media", choices=("image", "video"), default="image")
    parser.add_argument("--graph-mode", choices=("off", "decode", "all"), default="off")
    parser.add_argument(
        "--cuda-graph-scope", choices=("layer", "block", "none"), default="none"
    )
    parser.add_argument("--num-cuda-graphs", type=int, default=-1)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--max-sequence-length", type=int, default=2048)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--kv-cache-gb", type=int, default=4)
    parser.add_argument("--image-max-patches", type=int, default=128)
    parser.add_argument("--video-num-frames", type=int, default=8)
    parser.add_argument("--coordinator-port", type=int, default=50055)
    parser.add_argument("--http-host", default="127.0.0.1")
    parser.add_argument("--http-port", type=int, default=5000)
    parser.add_argument(
        "--prompt",
        default="Describe the visual content, including colors, shapes, text, and any motion.",
    )
    return parser.parse_args()


def make_mock_png() -> bytes:
    """Create a deterministic image without external data."""
    image = Image.new("RGB", (384, 256), color=(235, 240, 248))
    draw = ImageDraw.Draw(image)
    draw.rectangle((30, 35, 170, 185), fill=(32, 112, 220), outline="black", width=4)
    draw.ellipse((215, 45, 350, 180), fill=(242, 92, 84), outline="black", width=4)
    draw.text((115, 215), "LLAVA", fill=(10, 10, 10))
    stream = io.BytesIO()
    image.save(stream, format="PNG")
    return stream.getvalue()


def make_mock_mp4(num_frames: int) -> bytes:
    """Create a deterministic moving-shape video without external data."""
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
        draw.text((20, 15), f"LLAVA FRAME {frame_index + 1}", fill=(10, 10, 10))
        frame = av.VideoFrame.from_image(image)
        for packet in video_stream.encode(frame):
            container.mux(packet)
    for packet in video_stream.encode():
        container.mux(packet)
    container.close()
    return output.getvalue()


def configure_provider(provider: Any, args: argparse.Namespace) -> None:
    """Configure the known-good TP2/EP2 LLaVA checkpoint layout."""
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


def load_llava_model(args: argparse.Namespace) -> LLaVAModel:
    """Load HF weights explicitly through the historical LLaVA bridge."""
    auto_bridge = AutoBridge.from_hf_pretrained(args.hf_model, trust_remote_code=True)
    llava_bridge = NemotronOmniLlavaBridge()
    provider = llava_bridge.provider_bridge(auto_bridge.hf_pretrained)
    configure_provider(provider, args)
    provider.perform_initialization = False
    provider.register_pre_wrap_hook(
        partial(llava_bridge.load_weights_hf_to_megatron, auto_bridge.hf_pretrained)
    )
    provider.initialize_model_parallel(
        seed=1234,
        seed_kwargs={"inference_rng_tracker": True},
    )
    provider.finalize()
    distributed_models = provider.provide_distributed_model(wrap_with_ddp=False)
    if len(distributed_models) != 1:
        raise RuntimeError(
            "This example requires pipeline_model_parallel_size=1 and one local model chunk."
        )

    model = distributed_models[0].cuda().bfloat16().eval()
    model = model.module if hasattr(model, "module") else model
    if not hasattr(model, "llava_model"):
        raise TypeError(f"Expected a legacy LLaVA wrapper, got {type(model).__name__}.")
    llava_model = model.llava_model
    if not isinstance(llava_model, LLaVAModel):
        raise TypeError(f"Expected LLaVAModel, got {type(llava_model).__name__}.")

    llava_model.config.grad_scale_func = None
    llava_model.language_model.config.grad_scale_func = None
    return llava_model


def build_tokenizer(model_path: str) -> MegatronTokenizerText:
    return MegatronTokenizerText(
        model_path,
        {"library": "huggingface"},
        trust_remote_code=True,
        use_fast=True,
        include_special_tokens=True,
    )


def build_prompt_tokens(tokenizer, user_prompt: str) -> list[int]:
    """Build one compact media-marker prompt for the VLM wrapper."""
    marker_tokens = tokenizer.tokenize("<image>")
    if len(marker_tokens) != 1:
        raise RuntimeError(
            f"Expected <image> to tokenize to one ID, got {marker_tokens}. "
            "Check that the tokenizer matches the checkpoint."
        )
    marker_id = int(marker_tokens[0])
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
    if tokens.count(marker_id) != 1:
        raise RuntimeError(f"Expected one compact <image> marker, got {tokens.count(marker_id)}.")
    return tokens


def build_inference_config(model: LLaVAModel, args: argparse.Namespace) -> InferenceConfig:
    image_config = ImageProcessingConfig(
        patch_dim=int(model.patch_dim),
        dynamic_resolution=True,
        use_tiling=False,
        pixel_shuffle=bool(model._pixel_shuffle),
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
        mamba_inference_state_config=MambaInferenceStateConfig.from_model(model.language_model),
        pg_collection=model.pg_collection,
        num_cuda_graphs=None if args.graph_mode == "off" else args.num_cuda_graphs,
        use_cuda_graphs_for_non_decode_steps=args.graph_mode == "all",
        cuda_graph_max_tokens=min(args.max_tokens, 512),
        image_preprocessing_config=image_config,
        video_preprocessing_config=VideoProcessingConfig(
            image_config=image_config,
            num_frames=args.video_num_frames,
            temporal_patch_size=int(model.temporal_patch_dim),
        ),
    )


def sampling_params(tokenizer, args: argparse.Namespace) -> SamplingParams:
    return SamplingParams(
        temperature=1.0,
        top_k=1,
        num_tokens_to_generate=args.max_new_tokens,
        termination_id=tokenizer.eod,
        skip_prompt_log_probs=True,
    )


def print_generated_text(text: str) -> None:
    if dist.get_rank() == 0:
        print("\n======== LLAVA OUTPUT ========")
        print(text)
        print("==============================")


def validate_generation_result(result) -> None:
    """Surface engine-side admission or generation failures."""
    status = getattr(result, "status", None)
    if getattr(status, "name", status) == "COMPLETED":
        return
    errors = [
        str(event.payload)
        for event in getattr(result, "events", [])
        if getattr(getattr(event, "type", None), "name", "").startswith("ERROR_")
    ]
    details = "; ".join(errors) if errors else f"request status is {status}"
    raise RuntimeError(f"Generation failed: {details}")


async def run_async(args, model, tokenizer, config, prompt_tokens, media_bytes) -> None:
    async with MegatronAsyncLLM(
        model=model,
        tokenizer=tokenizer,
        inference_config=config,
        use_coordinator=True,
        coordinator_port=args.coordinator_port,
        inference_wrapper_cls=VLMInferenceWrapper,
    ) as llm:
        if llm.is_primary_rank:
            result = await llm.generate(
                prompt_tokens,
                sampling_params(tokenizer, args),
                multi_modal_data={args.media: media_bytes},
            )
            validate_generation_result(result)
            torch.cuda.synchronize()
            print_generated_text(result.generated_text)


def run_sync(args, model, tokenizer, config, prompt_tokens, media_bytes) -> None:
    with MegatronLLM(
        model=model,
        tokenizer=tokenizer,
        inference_config=config,
        use_coordinator=True,
        coordinator_port=args.coordinator_port,
        inference_wrapper_cls=VLMInferenceWrapper,
    ) as llm:
        if llm.is_primary_rank:
            result = llm.generate(
                prompt_tokens,
                sampling_params(tokenizer, args),
                multi_modal_data={args.media: media_bytes},
            )[0]
            validate_generation_result(result)
            torch.cuda.synchronize()
            print_generated_text(result.generated_text)


def post_completion(args, prompt_tokens, media_bytes) -> dict:
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


async def run_completions(args, model, tokenizer, config, prompt_tokens, media_bytes) -> None:
    lifecycle_group = dist.new_group(backend="gloo")
    try:
        async with MegatronAsyncLLM(
            model=model,
            tokenizer=tokenizer,
            inference_config=config,
            use_coordinator=True,
            coordinator_port=args.coordinator_port,
            inference_wrapper_cls=VLMInferenceWrapper,
        ) as llm:
            await llm.serve(
                ServeConfig(host=args.http_host, port=args.http_port, frontend_replicas=1),
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


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO)
    model = load_llava_model(args)
    tokenizer = build_tokenizer(args.hf_model)
    prompt_tokens = build_prompt_tokens(tokenizer, args.prompt)
    media_bytes = (
        make_mock_mp4(args.video_num_frames)
        if args.media == "video"
        else make_mock_png()
    )
    config = build_inference_config(model, args)

    if dist.get_rank() == 0:
        print(
            f"model={type(model).__name__}, api={args.api}, media={args.media}, "
            f"graph_mode={args.graph_mode}, prompt_tokens={len(prompt_tokens)}"
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
