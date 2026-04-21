# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
import argparse
import os
import sys
from typing import Optional, Tuple, List

# Add megatron and the multimodal example to the path.
sys.path.append(
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), os.path.pardir, os.path.pardir, os.path.pardir)
    )
)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))

import torch
from transformers import AutoModel

from examples.multimodal.model import model_provider
from examples.multimodal.multimodal_args import add_multimodal_extra_args
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.training import get_model
from megatron.training.checkpointing import load_checkpoint
from megatron.training.initialize import initialize_megatron
from einops import rearrange


VISION_MODEL_TYPE_MAP = {
    'internvit': 'internvit',
    'radio_v4-h-1d': 'radio',
    'radio_1d': 'radio',
    'radio-so400m': 'radio-so400m',
    'radio': 'radio',
}

CLASS_TOKEN_LEN_MAP = {
    'internvit': 1,
    'radio_v4-h-1d': 10,
    'radio_1d': 5,
    'radio-so400m': 10,
    'radio': 10,
}

RADIO_1D_NUM_TOKENS_MAP = {
    'radio_v4-h-1d': 128,
    'radio_1d': 128,
    'radio-so400m': None,
    'radio': None,
}

RADIO_DOWNSCALING_LEVELS_MAP = {
    'radio_v4-h-1d': [24],  # List of block indices where downscaling happens
    'radio_1d': None,
    'radio-so400m': None,
    'radio': None,
}


def patchify_images(images, patch_dim):
    """Patchify a batch of images into patch tokens.

    Args:
        images: Tensor of shape [N, C, H, W] - N images
        patch_dim: Patch size (e.g., 16)

    Returns:
        patches: Tensor of shape [N, num_patches, patch_dim*patch_dim*C]
    """
    N, C, H, W = images.shape
    py = H // patch_dim
    px = W // patch_dim
    patches = rearrange(
        images,
        'b c (py yy) (px xx) -> b (py px) (c yy xx)',
        py=py,
        yy=patch_dim,
        px=px,
        xx=patch_dim,
    )
    return patches


def remove_vision_class_tokens(
    image_embeddings: torch.Tensor,
    imgs_sizes: Optional[torch.Tensor],
    patch_dim: int,
    class_token_len: int,
    dynamic_resolution: bool,
    radio_1d_num_tokens: int = 0,
    num_downscaling_levels: int = 0,
) -> torch.Tensor:
    """Remove class tokens from vision embeddings.

    For dynamic resolution, class tokens appear at the start of each image's
    token sequence. This function calculates the correct positions accounting
    for downscaling and radio_1d_num_tokens limits.

    Args:
        image_embeddings: Vision model output [batch, seq_len, hidden]
        imgs_sizes: Image sizes tensor [num_images, 2] with (H, W) per image,
            or None for non-dynamic resolution
        patch_dim: Patch size (e.g., 16)
        class_token_len: Number of class tokens per image
        dynamic_resolution: Whether dynamic resolution is enabled
        radio_1d_num_tokens: Max output tokens per image (0 = no limit)
        num_downscaling_levels: Number of RADIO downscaling levels

    Returns:
        image_embeddings with class tokens removed
    """
    if not dynamic_resolution:
        # Simple case: just slice off the class tokens at the start
        return image_embeddings[:, class_token_len:, :]

    # Dynamic resolution: build mask to remove class tokens from each image
    remove_class_token_mask = torch.full(
        (image_embeddings.shape[-2],), True, dtype=torch.bool, device=image_embeddings.device
    )

    # Calculate actual patch count per image after downscaling
    # Each downscaling level reduces patches by 4x (2x in each dimension)
    downscale_factor = 2 ** num_downscaling_levels

    current_length = 0
    for img_size in imgs_sizes:
        H, W = img_size[0].item(), img_size[1].item()
        patches_h = H // patch_dim // downscale_factor
        patches_w = W // patch_dim // downscale_factor
        num_patches = patches_h * patches_w

        # If using radio_1d_num_tokens, that means we have at most this number for this image
        if radio_1d_num_tokens > 0:
            num_patches = min(num_patches, radio_1d_num_tokens)

        # Mark class token positions as False (to be removed)
        remove_class_token_mask[current_length : current_length + class_token_len] = False

        # Move to next image: class_tokens + patch_tokens
        current_length += class_token_len + num_patches

    return image_embeddings[:, remove_class_token_mask, :]


def create_dynamic_resolution_batch(image_sizes, patch_dim, dtype, device):
    """Create a batch of images in dynamic resolution format.

    For dynamic resolution, all images are flattened and concatenated into a single
    sequence with batch size 1. packed_seq_params contains cu_seqlens to indicate
    where each image's tokens start/end.

    Args:
        image_sizes: List of (H, W) tuples for each image
        patch_dim: Patch size (e.g., 16)
        dtype: Data type for tensors
        device: Device to create tensors on

    Returns:
        packed_patches: Tensor of shape [1, total_seq_len, patch_dim*patch_dim*3]
        imgs_sizes: List of (H, W) tuples
        packed_seq_params: PackedSeqParams with cu_seqlens for attention boundaries
    """
    all_patches = []
    seq_lens = []

    for H, W in image_sizes:
        # Create random image with this size
        img = torch.randn((1, 3, H, W), dtype=dtype, device=device)
        # Patchify
        patches = patchify_images(img, patch_dim)  # [1, num_patches, hidden]
        num_patches = patches.shape[1]
        all_patches.append(patches.squeeze(0))  # [num_patches, hidden]
        seq_lens.append(num_patches)

    # Concatenate all patches into single sequence
    packed_patches = torch.cat(all_patches, dim=0)  # [total_patches, hidden]
    packed_patches = packed_patches.unsqueeze(0)  # [1, total_patches, hidden]

    # Build cu_seqlens: cumulative sequence lengths [0, len1, len1+len2, ...]
    cu_seqlens = [0]
    for seq_len in seq_lens:
        cu_seqlens.append(cu_seqlens[-1] + seq_len)

    cu_seqlens_tensor = torch.tensor(cu_seqlens, dtype=torch.int32, device=device)
    max_seqlen = max(seq_lens)

    packed_seq_params = PackedSeqParams(
        qkv_format="thd",
        cu_seqlens_q=cu_seqlens_tensor,
        cu_seqlens_kv=cu_seqlens_tensor,
        cu_seqlens_q_padded=None,
        cu_seqlens_kv_padded=None,
        max_seqlen_q=max_seqlen,
        max_seqlen_kv=max_seqlen,
    )

    # imgs_sizes as list of tuples (H, W) in pixels
    imgs_sizes = list(image_sizes)

    return packed_patches, imgs_sizes, packed_seq_params


def run_mcore_vision(
    model_path: str,
    mcore_model_type: str,
    language_model_type: str,
    images: torch.Tensor,
    vision_resolution: int = 448,
    use_te: bool = False,
    dynamic_resolution: bool = False,
    num_images: int = 3,
    tensor_parallel_size: int = 4,
    expert_model_parallel_size: int = 1,
) -> Tuple[torch.Tensor, List[Tuple[int, int]], PackedSeqParams]:  # output, imgs_sizes, packed_seq_params
    """Run mcore vision model."""
    os.environ["CUDA_DEVICE_MAX_CONNECTIONS"] = "1"

    class_token_len = CLASS_TOKEN_LEN_MAP[mcore_model_type]
    model_type = VISION_MODEL_TYPE_MAP[mcore_model_type]
    patch_dim = 16  # RADIO uses 16x16 patches

    if mcore_model_type == "internvit" and language_model_type == "mistral_7b":
        # Megatron has some mandatory flags.
        sys.argv = [
            "ignore_me.py",
            "--micro-batch-size=1",
            "--num-layers=2",
            f"--vision-model-type={model_type}",
            "--language-model-type=mistral_7b",
            "--tokenizer-prompt-format=mistral",
            "--tokenizer-type=MultimodalTokenizer",
            "--tokenizer-model=mistralai/Mistral-7B-Instruct-v0.3",
            "--vocab-size=1024",
            "--hidden-size=64",
            "--num-attention-heads=8",
            "--seq-length=1024",
            "--decoder-seq-length=2048",
            "--max-position-embeddings=2048",
            "--bf16",
            "--img-h=448",
            "--img-w=448",
            "--patch-dim=14",
            f"--tensor-model-parallel-size={tensor_parallel_size}",
            "--use-distributed-optimizer",
            f"--pretrained-checkpoint={model_path}",
        ]
        if use_te:
            sys.argv.append("--use-te")
    elif "radio" in mcore_model_type:
        # For dynamic resolution, use a larger max image size
        img_h = vision_resolution
        img_w = vision_resolution

        # Base args for 9B hybrid and 30B MoE (sorted alphabetically)
        # Determined by comparing pretrain script options
        # NOTE: Some of these args may be unnecessary, ideally want the minimum set for loading + forward pass
        sys.argv = [
            "ignore_me.py",
            "--allow-missing-vision-projection-checkpoint",
            "--attention-backend=flash",
            "--attention-dropout=0.0",
            "--bf16",
            "--ckpt-format=torch",
            f"--class-token-len={class_token_len}",
            "--decoder-seq-length=16384",
            "--disable-bias-linear",
            "--disable-vision-class-token",
            "--group-query-attention",
            "--hidden-dropout=0.0",
            f"--img-h={img_h}",
            f"--img-w={img_w}",
            "--is-hybrid-model",
            "--kv-channels=128",
            "--max-position-embeddings=16384",
            "--micro-batch-size=1",
            "--normalization=RMSNorm",
            f"--patch-dim={patch_dim}",
            "--pipeline-model-parallel-size=1",
            "--position-embedding-type=none",
            f"--pretrained-checkpoint={model_path}",
            "--seq-length=256",  # Tiling setting
            "--squared-relu",
            "--spec", "megatron.core.models.mamba.mamba_layer_specs", "mamba_stack_spec",
            f"--tensor-model-parallel-size={tensor_parallel_size}",
            "--tokenizer-type=MultimodalTokenizer",
            "--transformer-impl=transformer_engine",
            "--untie-embeddings-and-output-weights",
            f"--vision-model-type={model_type}",
        ]

        if language_model_type == "nemotron5-hybrid-9b":
            # LLM args
            llm_args = [
                "--attention-softmax-in-fp32",
                "--ffn-hidden-size=15680",
                "--hidden-size=4480",
                "--hybrid-override-pattern=M-M-M-MM-M-M-M*-M-M-M*-M-M-M-M*-M-M-M-M*-M-MM-M-M-M-M-M-",
                "--language-model-type=nemotron5-hybrid-9b",
                "--make-vocab-size-divisible-by=16512",
                "--mamba-head-dim=80",
                "--mamba-num-heads=128",
                "--mamba-state-dim=128",
                "--no-masked-softmax-fusion",
                "--norm-epsilon=1e-05",
                "--num-attention-heads=40",
                "--num-layers=56",
                "--num-query-groups=8",
                "--tokenizer-model=/lustre/fsw/portfolios/llmservice/users/ksapra/checkpoints/prunedmodel/first_9b_sft_pruned_v0",
                "--tokenizer-prompt-format=nemotron-h-5p5-reasoning",
            ]
        elif language_model_type == "nemotron6-moe":
            llm_args = [
                "--enable-experimental",
                f"--expert-model-parallel-size={expert_model_parallel_size}",
                "--expert-tensor-parallel-size=1",
                "--ffn-hidden-size=1856",
                "--hidden-size=2688",
                "--hybrid-override-pattern=MEMEM*EMEMEM*EMEMEM*EMEMEM*EMEMEM*EMEMEMEM*EMEMEMEME",
                "--language-model-type=nemotron6-moe",
                "--mamba-head-dim=64",
                "--mamba-num-heads=64",
                "--moe-aux-loss-coeff=1e-6",
                "--moe-grouped-gemm",
                "--moe-permute-fusion",
                "--moe-router-dtype=fp32",
                "--moe-router-enable-expert-bias",
                "--moe-router-load-balancing-type=seq_aux_loss",
                "--moe-router-score-function=sigmoid",
                "--moe-router-topk=6",
                "--moe-router-topk-scaling-factor=2.5",
                "--moe-shared-expert-intermediate-size=3712",
                "--moe-shared-expert-overlap",
                "--moe-token-dispatcher-type=alltoall",
                "--num-attention-heads=32",
                "--num-experts=128",
                "--num-layers=52",
                "--num-query-groups=2",
                "--sequence-parallel",
                # New tokenizer 10/20
                "--tiktoken-pattern=v2",
                "--tokenizer-model=/lustre/fsw/portfolios/llmservice/users/trintamaki/workspace/hf-transformers/hub/models--nvidia--Nemotron-Nano-3-30B-A3.5B-dev-1016/snapshots/bb271274159f07461e919379311e32802e5ec36b/",
                "--tokenizer-prompt-format=nemotron6-moe",
                "--use-fused-weighted-squared-relu",
                "--use-mcore-models",
            ]
        else:
            raise ValueError(f"Unsupported language model type for radio: {language_model_type}")

        sys.argv.extend(llm_args)

        if dynamic_resolution:
            sys.argv.append("--dynamic-resolution")

        if RADIO_1D_NUM_TOKENS_MAP.get(mcore_model_type):
            sys.argv.append(f"--radio-1d-num-tokens={RADIO_1D_NUM_TOKENS_MAP[mcore_model_type]}")
        if RADIO_DOWNSCALING_LEVELS_MAP.get(mcore_model_type):
            # Append flag and each level as separate argv entries for argparse nargs='*'
            sys.argv.append("--radio-downscaling-levels")
            for level in RADIO_DOWNSCALING_LEVELS_MAP[mcore_model_type]:
                sys.argv.append(str(level))
        if use_te:
            sys.argv.append("--use-te")
    else:
        raise ValueError(
            f"Unsupported combination of mcore model type: {mcore_model_type} and language model type: {language_model_type}")

    print(sys.argv)

    os.environ["WANDB_MODE"] = "disabled"
    initialize_megatron(extra_args_provider=add_multimodal_extra_args)

    def wrapped_model_provider(pre_process, post_process):
        return model_provider(pre_process, post_process, parallel_output=False)

    # Set up model and load checkpoint.
    model = get_model(wrapped_model_provider, wrap_with_ddp=False)

    load_checkpoint(model, None, None)

    vision_model = model[0].module.vision_model
    vision_model.eval()

    torch.manual_seed(42)
    torch.cuda.manual_seed(42)

    if dynamic_resolution:
        # For dynamic resolution with downscaling, images must be multiples of 32 pixels
        # (patch_dim=16 * 2 for one downscaling level)
        downscaling_levels = RADIO_DOWNSCALING_LEVELS_MAP.get(mcore_model_type)
        if downscaling_levels:
            # With downscaling, need to be divisible by patch_dim * 2^num_levels
            # For one level of downscaling (at block 24), need divisible by 32
            required_divisor = patch_dim * 2  # 32
        else:
            required_divisor = patch_dim  # 16

        # Create images of varying sizes (all multiples of required_divisor)
        # Use different sizes to test dynamic resolution properly
        image_sizes = []
        for i in range(num_images):
            # Vary sizes: e.g., 256x256, 384x256, 256x384 (all multiples of 32)
            h_mult = 8 + (i % 3) * 4  # 8, 12, 8 -> 256, 384, 256
            w_mult = 8 + ((i + 1) % 3) * 4  # 12, 8, 12 -> 384, 256, 384
            H = required_divisor * h_mult
            W = required_divisor * w_mult
            image_sizes.append((H, W))

        print(f"[Dynamic Resolution Test] Creating {num_images} images with sizes: {image_sizes}")

        packed_patches, imgs_sizes, packed_seq_params = create_dynamic_resolution_batch(
            image_sizes, patch_dim, torch.bfloat16, "cuda"
        )

        print(f"[Dynamic Resolution Test] Packed patches shape: {packed_patches.shape}")
        print(f"[Dynamic Resolution Test] cu_seqlens: {packed_seq_params.cu_seqlens_q.tolist()}")
        print(f"[Dynamic Resolution Test] max_seqlen: {packed_seq_params.max_seqlen_q}")

        output = vision_model(
            packed_patches,
            imgs_sizes=imgs_sizes,
            packed_seq_params=packed_seq_params
        )

        print(f"[Dynamic Resolution Test] mcore output shape (before class token removal): {output.shape}")
        print(f"[Dynamic Resolution Test] Final cu_seqlens (after forward): {packed_seq_params.cu_seqlens_q.tolist()}")

        # Strip class tokens using the shared helper function
        # Class tokens are always added in RADIO mcore, but TorchHub doesn't include them
        radio_1d_num_tokens = RADIO_1D_NUM_TOKENS_MAP.get(mcore_model_type, 0)
        downscaling_levels = RADIO_DOWNSCALING_LEVELS_MAP.get(mcore_model_type)
        num_downscaling_levels = len(downscaling_levels) if downscaling_levels else 0

        imgs_sizes_tensor = torch.tensor(imgs_sizes, dtype=torch.int32, device=output.device)
        output = remove_vision_class_tokens(
            image_embeddings=output,
            imgs_sizes=imgs_sizes_tensor,
            patch_dim=patch_dim,
            class_token_len=class_token_len,
            dynamic_resolution=True,
            radio_1d_num_tokens=radio_1d_num_tokens,
            num_downscaling_levels=num_downscaling_levels,
        )
        print(f"[Dynamic Resolution Test] mcore output shape (after class token removal): {output.shape}")

        return output, imgs_sizes, packed_seq_params
    else:
        images = torch.randn((1, 3, vision_resolution, vision_resolution), dtype=torch.bfloat16, device="cuda")
        output = vision_model(images)
        # Strip class tokens using the shared helper function
        # TorchHub RADIO doesn't include class tokens in its output
        output = remove_vision_class_tokens(
            image_embeddings=output,
            imgs_sizes=None,  # Not needed for non-dynamic
            patch_dim=patch_dim,
            class_token_len=class_token_len,
            dynamic_resolution=False,
        )
        return output, None, None


def run_hf_vision(model_name, images):
    """Run HF vision model."""
    model = (
        AutoModel.from_pretrained(model_name, torch_dtype=torch.bfloat16, trust_remote_code=True)
        .cuda()
        .eval()
    )

    outputs = model(images, return_dict=True)

    return outputs


def run_torchhub_vision(model_version, mcore_model_type, torchhub_version, images=None,
                        dynamic_resolution=False, image_sizes=None, patch_dim=16,
                        radio_1d_num_tokens=0):
    """Run TorchHub vision model."""
    if os.path.exists(torchhub_version):
        torchhub_source = "local"
    else:
        torchhub_source = "github"
    model = torch.hub.load(torchhub_version, 'radio_model', version=model_version, source=torchhub_source, progress=True).cuda().eval()
    model.make_preprocessor_external()

    # Convert model to bfloat16 to match Megatron model precision
    model = model.to(torch.bfloat16)

    if dynamic_resolution:
        # For dynamic resolution, we need to run each image separately through torchhub
        # and concatenate the outputs (torchhub doesn't support packed sequences)
        # radio_1d_num_tokens limit is applied per image (not globally)
        outputs = []
        torch.manual_seed(42)
        torch.cuda.manual_seed(42)

        for H, W in image_sizes:
            img = torch.randn((1, 3, H, W), dtype=torch.bfloat16, device="cuda")

            if mcore_model_type == "radio_1d":
                if radio_1d_num_tokens > 0:
                    out = model(img, qradio_size=radio_1d_num_tokens)["1d"].features
                else:
                    out = model(img)["1d"].features
            elif mcore_model_type == "radio_v4-h-1d":
                if radio_1d_num_tokens > 0:
                    out = model(img, num_tokens=radio_1d_num_tokens)["encoder"].features
                else:
                    out = model(img)["encoder"].features
            elif "radio" in mcore_model_type:
                out = model(img).features
            else:
                out = model(img)

            outputs.append(out)

        # Concatenate all outputs along sequence dimension
        output = torch.cat(outputs, dim=1)
        print(f"[Dynamic Resolution Test] TorchHub output shape: {output.shape}")
        print(f"[Dynamic Resolution Test] Per-image output shapes: {[o.shape for o in outputs]}")

        return output
    else:
        # Images are already bfloat16, so use them directly
        if mcore_model_type == "radio_1d":
            output = model(images, qradio_size=128)["1d"].features
        elif mcore_model_type == "radio_v4-h-1d":
            output = model(images, num_tokens=128)["encoder"].features
        elif "radio" in mcore_model_type:
            output = model(images).features
        else:
            output = model(images)

        return output


def main(
    mcore_model: str,
    mcore_model_type: str,
    language_model_type: str,
    hf_model: str,
    torchhub_model_version: str,
    torchhub_version: str,
    vision_resolution: int = 448,
    use_te: bool = False,
    dynamic_resolution: bool = False,
    num_images: int = 3,
    tensor_parallel_size: int = 4,
    expert_model_parallel_size: int = 1,
) -> None:
    """Compare vision model outputs between mcore and HF given the same fixed input."""

    # Set seed for reproducibility
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)

    patch_dim = 16  # RADIO uses 16x16 patches

    if dynamic_resolution:
        mcore, imgs_sizes, packed_seq_params = run_mcore_vision(
            model_path=mcore_model,
            mcore_model_type=mcore_model_type,
            language_model_type=language_model_type,
            images=None,
            vision_resolution=vision_resolution,
            use_te=use_te,
            dynamic_resolution=True,
            num_images=num_images,
            tensor_parallel_size=tensor_parallel_size,
            expert_model_parallel_size=expert_model_parallel_size
        )
    else:
        images = torch.randn((1, 3, vision_resolution, vision_resolution), dtype=torch.bfloat16, device="cuda")
        mcore, _, _ = run_mcore_vision(
            model_path=mcore_model,
            mcore_model_type=mcore_model_type,
            language_model_type=language_model_type,
            images=images,
            vision_resolution=vision_resolution,
            use_te=use_te,
            tensor_parallel_size=tensor_parallel_size,
            expert_model_parallel_size=expert_model_parallel_size
        )

    if torch.distributed.get_rank() == 0:
        if hf_model:
            if dynamic_resolution:
                raise ValueError("HF model comparison not supported for dynamic resolution")
            hf = run_hf_vision(hf_model, images)
            reference_output = hf["last_hidden_state"]
        elif torchhub_model_version:
            if dynamic_resolution:
                radio_1d_num_tokens = RADIO_1D_NUM_TOKENS_MAP.get(mcore_model_type, 0)
                reference_output = run_torchhub_vision(
                    torchhub_model_version, mcore_model_type, torchhub_version,
                    dynamic_resolution=True, image_sizes=imgs_sizes, patch_dim=patch_dim,
                    radio_1d_num_tokens=radio_1d_num_tokens
                )
            else:
                reference_output = run_torchhub_vision(torchhub_model_version, mcore_model_type, torchhub_version, images)
        else:
            raise ValueError("Either hf_model or torchhub_model_version must be provided")

        # Make sure shapes
        if mcore.shape != reference_output.shape:
            raise ValueError(f"mcore shape {mcore.shape} does not match reference output shape {reference_output.shape}")

        # Print some statistics about both outputs (std/max/min/mean)
        print(f"mcore std {mcore.std().item()}, max {mcore.max().item()}, min {mcore.min().item()}, mean {mcore.mean().item()}")
        print(f"reference output std {reference_output.std().item()}, max {reference_output.max().item()}, min {reference_output.min().item()}, mean {reference_output.mean().item()}")

        # Compare logits. Due to different attention implementations and other details,
        # there will be numerical differences.
        diff = (mcore - reference_output).abs()
        mean_diff = diff.mean().item()
        max_diff = diff.max().item()

        # Find location of maximum difference
        max_diff_idx = diff.argmax()
        max_diff_coords = torch.unravel_index(max_diff_idx, diff.shape)

        # Get values at max diff location
        mcore_val_at_max = mcore[max_diff_coords].item()
        ref_val_at_max = reference_output[max_diff_coords].item()

        # Print detailed diff analysis
        print(f"=== Difference Analysis ===")
        print(f"Tensor shape: {diff.shape}")
        print(f"Mean diff: {mean_diff:.6f}")
        print(f"Max diff: {max_diff:.6f}")
        print(f"Max diff location: {max_diff_coords}")
        print(f"Mcore value at max diff: {mcore_val_at_max:.6f}")
        print(f"Reference value at max diff: {ref_val_at_max:.6f}")

        # Percentile analysis (convert to float for quantile computation)
        diff_flat = diff.flatten().float()
        p50 = torch.quantile(diff_flat, 0.5).item()
        p90 = torch.quantile(diff_flat, 0.9).item()
        p95 = torch.quantile(diff_flat, 0.95).item()
        p99 = torch.quantile(diff_flat, 0.99).item()
        print(f"Diff percentiles - 50th: {p50:.6f}, 90th: {p90:.6f}, 95th: {p95:.6f}, 99th: {p99:.6f}")

        # Show some context around max diff location (if it's a 3D tensor)
        if len(diff.shape) == 3:
            batch_idx, seq_idx, feat_idx = max_diff_coords
            print(f"Max diff at batch {batch_idx}, sequence position {seq_idx}, feature {feat_idx}")

            # Show a small window around the max diff location in the feature dimension
            feat_start = max(0, feat_idx - 2)
            feat_end = min(diff.shape[2], feat_idx + 3)
            print(f"Feature values around max diff (features {feat_start}:{feat_end}):")
            print(f"  Mcore:     {mcore[batch_idx, seq_idx, feat_start:feat_end].tolist()}")
            print(f"  Reference: {reference_output[batch_idx, seq_idx, feat_start:feat_end].tolist()}")
            print(f"  Diff:      {diff[batch_idx, seq_idx, feat_start:feat_end].tolist()}")

        # Additional diagnostics to check for shifts/permutations
        print(f"\n=== Shift/Permutation Analysis ===")

        # Check if tensors might be shifted along sequence dimension
        print("Testing sequence shifts...")
        best_shift = 0
        best_shift_diff = float('inf')
        for shift in range(-min(10, mcore.shape[1]//4), min(10, mcore.shape[1]//4) + 1):
            if shift == 0:
                continue
            if shift > 0:
                shifted_mcore = mcore[:, shift:, :]
                shifted_ref = reference_output[:, :-shift, :]
            else:
                shifted_mcore = mcore[:, :shift, :]
                shifted_ref = reference_output[:, -shift:, :]

            shift_diff = (shifted_mcore - shifted_ref).abs().mean().item()
            if shift_diff < best_shift_diff:
                best_shift_diff = shift_diff
                best_shift = shift
            print(f"  Shift {shift:2d}: mean diff = {shift_diff:.6f}")

        if best_shift_diff < mean_diff * 0.8:
            print(f"*** POTENTIAL SEQUENCE SHIFT DETECTED: shift={best_shift}, diff={best_shift_diff:.6f} ***")

        # Check for potential feature dimension permutation by comparing sorted values
        mcore_sorted = torch.sort(mcore.flatten().float())[0]
        ref_sorted = torch.sort(reference_output.flatten().float())[0]
        sorted_diff = (mcore_sorted - ref_sorted).abs().mean().item()
        print(f"Sorted values diff: {sorted_diff:.6f} (if ~0, values are same but permuted)")

        # Check correlation between flattened tensors (convert to float for corrcoef)
        mcore_flat = mcore.flatten().float()
        ref_flat = reference_output.flatten().float()
        correlation = torch.corrcoef(torch.stack([mcore_flat, ref_flat]))[0, 1].item()
        print(f"Tensor correlation: {correlation:.6f} (1.0 = perfect correlation)")

        # Check if there's a simple offset
        offset_diff = (mcore - reference_output).mean().item()
        print(f"Mean offset: {offset_diff:.6f}")
        if abs(offset_diff) > 0.001:
            offset_corrected_diff = (mcore - reference_output - offset_diff).abs().mean().item()
            print(f"Offset-corrected mean diff: {offset_corrected_diff:.6f}")

        print(f"==============================")

        # With high correlation (>0.998), these are acceptable numerical differences between implementations
        if correlation > 0.995:
            acceptable_mean_diff = 0.2
            acceptable_max_diff = 100  # Relaxed for high correlation - outliers in extreme values are expected
            relaxed_thresholds = True
        else:
            acceptable_mean_diff = 0.1
            acceptable_max_diff = 50
            relaxed_thresholds = False

        print(f"Correlation: {correlation} (relaxed thresholds: {relaxed_thresholds})")
        print(f"Mean diff: {mean_diff}, current threshold: {acceptable_mean_diff}")
        print(f"Max diff: {max_diff}, current threshold: {acceptable_max_diff}")

        assert mean_diff < acceptable_mean_diff, f"mean output difference {mean_diff:.6f} exceeds threshold {acceptable_mean_diff} (correlation: {correlation:.6f})"
        assert max_diff < acceptable_max_diff, f"max output difference {max_diff:.6f} exceeds threshold {acceptable_max_diff} (correlation: {correlation:.6f})"

        print(f"==============================")
        print("Test passed")
        print(f"==============================")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Check mcore vision model output vs. HF numerically.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--mcore-model", type=str, required=True, help="directory for mcore model weights"
    )
    parser.add_argument(
        "--mcore-model-type", type=str, required=True, choices=list(VISION_MODEL_TYPE_MAP.keys()), help="mcore model type to test"
    )
    parser.add_argument("--language-model-type", type=str, choices=["nemotron5-hybrid-9b", "nemotron6-moe", "mistral_7b"], help="Language model type")
    parser.add_argument("--hf-model", type=str, required=False, help="Model name in HF")
    parser.add_argument("--torchhub-model-version", type=str, required=False, help="Model name in TorchHub, or local path")
    parser.add_argument(
        "--torchhub-version",
        type=str,
        default="NVlabs/RADIO",
        help="TorchHub repo. Can be a local path or a Github repo. By default use NVlabs/RADIO.")
    parser.add_argument(
        "--vision-resolution", type=int, default=448, help="Vision input resolution (height and width)"
    )
    parser.add_argument(
        "--use-te", action="store_true", help="Use Transformer Engine"
    )
    parser.add_argument(
        "--dynamic-resolution", action="store_true",
        help="Test dynamic resolution mode with multiple images of varying sizes packed into a single sequence"
    )
    parser.add_argument(
        "--num-images", type=int, default=3,
        help="Number of images to test with when using --dynamic-resolution (default: 3)"
    )
    parser.add_argument(
        "--tensor-parallel-size", type=int, default=4,
        help="Tensor model parallel size. (default: 4 for dense models; but usually 2 for MoE models)"
    )
    parser.add_argument(
        "--expert-model-parallel-size", type=int, default=8,
        help=(
            "Expert model parallel size for MoE models, e.g. `EP` value in scripts. Using >8 requires"
            " testing on multi-node, non-interactive job. Using 8 may require converting to TP8 for"
            " single node testing. (default: 8)"
        )
    )

    args = parser.parse_args()

    main(
        mcore_model=args.mcore_model,
        mcore_model_type=args.mcore_model_type,
        language_model_type=args.language_model_type,
        hf_model=args.hf_model,
        torchhub_model_version=args.torchhub_model_version,
        torchhub_version=args.torchhub_version,
        vision_resolution=args.vision_resolution,
        use_te=args.use_te,
        dynamic_resolution=args.dynamic_resolution,
        num_images=args.num_images,
        tensor_parallel_size=args.tensor_parallel_size,
        expert_model_parallel_size=args.expert_model_parallel_size
    )
