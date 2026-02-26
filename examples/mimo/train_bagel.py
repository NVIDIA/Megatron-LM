# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

"""
This script provides a basic training loop for MIMO models.
"""

import os
import sys
from functools import partial
from typing import Any, Dict, Iterator

import torch

from megatron.core.parallel_state import (
    get_tensor_model_parallel_group,
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_src_rank,
)

# Add the parent directory to the path to import from megatron
# sys.path.append(
#     os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir, os.path.pardir))
# )
# Ensure bagel package is importable when train.py is run without example_bagel_gen_training.sh
_repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
_bagel_package = os.path.join(_repo_root, "bagel-package")
for _p in (_repo_root, _bagel_package):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from data.energon_avlm_task_encoder import llava_avlm_dataloader_provider
from data.energon_vlm_task_encoder import llava_vlm_dataloader_provider
from data.mock import (
    train_valid_test_datasets_provider as mock_train_valid_test_datasets_provider,
)
from data.bagel import bagel_dataloader_provider, bagel_packed_batch_to_mimo_batch
from model_providers.llava_avlm import model_provider_llava_avlm
from model_providers.llava_vlm import model_provider_llava_vlm
from model_providers.mock import model_provider_mock_vlm_single_encoder
from model_providers.bagel import model_provider_bagel
from utils.data_helpers import broadcast_nested_data_batch

from megatron.core.enums import ModelType
from megatron.training import get_args, pretrain

_MODEL_PROVIDERS = {
    "mock": model_provider_mock_vlm_single_encoder,
    "llava_vlm": model_provider_llava_vlm,
    "video_llava_vlm": partial(model_provider_llava_vlm, is_video_input=True),
    "llava_avlm": model_provider_llava_avlm,
    "bagel": model_provider_bagel,
    "bagel_mot": partial(model_provider_bagel, decoder_layer_module="Qwen2MoTDecoderLayer"),
}

_DATASET_PROVIDERS = {
    "mock": mock_train_valid_test_datasets_provider,
    "llava_vlm": llava_vlm_dataloader_provider,
    "video_llava_vlm": partial(llava_vlm_dataloader_provider, is_video_input=True),
    "llava_avlm": llava_avlm_dataloader_provider,
    "bagel": bagel_dataloader_provider,
}

def add_mimo_args(parser):
    """Add MIMO-specific arguments to the parser."""
    group = parser.add_argument_group('MIMO', 'MIMO specific arguments')

    # MIMO-specific parameters
    group.add_argument('--dataset-provider', type=str, default='mock', help='Dataset provider to choose from [mock, llava_vlm, video_llava_vlm, llava_avlm, bagel]')
    group.add_argument('--model-provider', type=str, default='mock', help='Model provider to choose from [mock, llava_vlm, video_llava_vlm, llava_avlm, bagel]')
    group.add_argument('--model-path', type=str, default=None, help='Path to model checkpoint to load')

    # mock dataloader related args
    # can control mock samples with total seq length and image seq length
    group.add_argument('--image-size', type=int, default=224, help='Image size for vision encoder')
    group.add_argument('--total-seq-length', type=int, default=512, help='Total sequence length')
    group.add_argument('--pad-token-id', type=int, default=0, help='Padding token ID')
    group.add_argument('--image-token-id', type=int, default=32000, help='Image token ID')
    group.add_argument(
        '--image-seq-length', type=int, default=197, help='Number of image tokens to pad'
    )
    group.add_argument(
        '--audio-encoder-model', type=str, default=None, help='Audio encoder model name'
    )
    group.add_argument(
        '--hf-assign-unused-tokens', type=str, nargs='+', default=None,
                       help='Assigning unused tokens to special tokens. Example: '
                       '--hf-assign-unused-tokens "<audio>,32002" "<video>,32003"'
    )
    # checkpoint related args
    group.add_argument('--language-model-checkpoint', type=str, default=None, help='Path to language model checkpoint to load')
    # energon dataloader related args
    group.add_argument('--packing-buffer-size', type=int, default=None, help='Packing buffer size when using sequence packing')

    # Bagel-specific args
    group.add_argument('--text-cond-dropout-prob', type=float, default=0.1, help='Text conditional dropout probability')
    group.add_argument('--vit-cond-dropout-prob', type=float, default=0.4, help='VIT conditional dropout probability')
    group.add_argument('--vae-cond-dropout-prob', type=float, default=0.3, help='VAE conditional dropout probability')
    # group.add_argument('--vae-image-downsample', type=int, default=16, help='VAE image downsample factor')
    group.add_argument('--max-latent-size', type=int, default=32, help='Maximum latent grid size (patches per side) for the VAE latent tensor.')
    group.add_argument('--vit-patch-size', type=int, default=14, help='VIT patch size')
    group.add_argument('--max-num-patch-per-side', type=int, default=70, help='Max number of patches per side')
    group.add_argument('--max-num-tokens-per-sample', type=int, default=16384, help='Max number of tokens per sample')
    group.add_argument('--max-num-tokens', type=int, default=36864, help='Max number of tokens')
    group.add_argument('--prefer-buffer-before', type=int, default=16384, help='Prefer buffer before this number of tokens')
    group.add_argument('--interpolate-pos', action='store_true', help='Whether to interpolate position embeddings')
    group.add_argument('--use-flex-attention', action='store_true', help='Whether to use flex attention')

    # Bagel-specific args
    group.add_argument('--llm-path', type=str, default=None, help='Path to LLM checkpoint to load')
    group.add_argument('--vit-path', type=str, default=None, help='Path to VIT checkpoint to load')

    # Language model backend selection
    group.add_argument('--language-use-mcore', action='store_true',
                       help='Use Megatron Core GPTModel-based language model instead of HuggingFace. '
                            'Default is False (use HuggingFace BagelLLMHuggingFaceModel)')


    #diffusion related args
    group.add_argument('--vae-path', type=str, default=None, help='Path to vae checkpoint')
    group.add_argument('--latent-patch-size', type=int, default=2, help='Spatial size (in VAE pixels) covered by each latent patch.')
    group.add_argument('--timestep-shift', type=float, default=1.0, help='Timestep shift for the diffusion model')
    return parser


def get_batch(data_iterator: Iterator[Dict[str, Any]]):
    """Generate a batch for MIMO model training.

    Args:
        data_iterator: Iterator over the dataset

    Returns:
        tuple: Batch data for model training
    """
    args = get_args()
    # cur_rank = torch.distributed.get_rank()
    # print(f"Run get batch on rank {cur_rank}")

    # Assert that context parallelism and pipeline parallelism are not supported yet
    assert (
        getattr(args, 'context_parallel_size', 1) == 1
    ), "Context parallelism is not supported yet in MIMO implementation"

    assert (getattr(args, 'pipeline_model_parallel_size', 1) == 1), \
        "Pipeline parallelism is not supported yet in MIMO implementation"

    # Broadcast data - only get data on tensor parallel rank 0
    # data iterator is None on other tp ranks
    # TP Rank-0 reads next batch.
    if get_tensor_model_parallel_rank() == 0:
        try:
            data = next(data_iterator)
            has_data = torch.tensor([1], dtype=torch.uint8, device='cuda')
        except StopIteration:
            has_data = torch.tensor([0], dtype=torch.uint8, device='cuda')
            data = None
    else:
        has_data = torch.empty(1, dtype=torch.uint8, device='cuda')
        data = None
        diffusion_wrapper = getattr(args, 'diffusion_wrapper', None)
        if diffusion_wrapper is not None:
            diffusion_wrapper.remove_vae()

    src = get_tensor_model_parallel_src_rank()
    group = get_tensor_model_parallel_group()
    torch.distributed.broadcast(has_data, src, group=group)

    if has_data.item() == 0:
        # iterator exhausted on all ranks
        # we need this to avoid race condition when first tp rank hits StopIteration
        return None

    # MiMo forward pass expects
    # input_ids: torch.Tensor,
    # position_ids: Optional[torch.Tensor] = None,
    # attention_mask: Optional[torch.Tensor] = None,
    # loss_mask: Optional[torch.Tensor] = None,
    # labels: Optional[torch.Tensor] = None,
    # modality_inputs: Optional[Dict[str, Dict[str, Any]]] = None,
    # modality_seq_lengths: Optional[Dict[str, torch.Tensor]] = None,

    # For the modality inputs, the keys can be arbitrary
    # so we do a broadcast of the schema followed by a broadcast of the actual data
    # check broadcast_nested_data_batch for more details

    # Check if this is bagel dataset (PackedDataset format)
    if args.dataset_provider == 'bagel' and data is not None:
        # Convert bagel packed batch to MIMO format
        diffusion_wrapper = getattr(args, 'diffusion_wrapper', None)
        if diffusion_wrapper is not None:
            diffusion_wrapper.cuda()
        # print(f"Run bagel_packed_batch_to_mimo_batch on rank {cur_rank}")
        data = bagel_packed_batch_to_mimo_batch(data, diffusion_wrapper=diffusion_wrapper)

    # print(f"Run broadcast_nested_data_batch on rank {cur_rank}")
    batch = broadcast_nested_data_batch(data)
    return batch


def loss_func(loss_mask, output_tensor):
    """Simple loss function for MIMO model training.

    Args:
        loss_mask: mask indicating which tokens contribute to the loss
        output_tensor: model output tensor or dict (for Bagel)

    Returns:
        tuple: (loss, num_tokens, metrics_dict)
    """
    # Handle Bagel output format (dict with 'ce' key)
    if isinstance(output_tensor, dict):
        ce_loss = output_tensor.get('ce')
        mse_loss = output_tensor.get('mse')
        # print(f"ce_loss: {ce_loss}, mse_loss: {mse_loss}")

        total_loss = torch.tensor(0.0, device='cuda')
        total_tokens = torch.tensor(0, dtype=torch.int, device='cuda')

        # Cross-entropy loss for understanding tasks
        if ce_loss is not None:
            # ce_loss is already per-token CE loss at ce_loss_indexes
            # loss_mask contains weights at those positions
            # Extract non-zero weights from loss_mask
            weights = loss_mask.view(-1)
            non_zero_mask = weights > 0
            ce_weights = weights[non_zero_mask]

            # Make sure ce_loss and ce_weights have same length
            if ce_loss.numel() == ce_weights.numel():
                weighted_ce_loss = (ce_loss.float() * ce_weights).sum()
                total_loss = total_loss + weighted_ce_loss
                total_tokens = total_tokens + ce_weights.sum().to(torch.int)
            else:
                # Fallback: just sum the ce_loss
                total_loss = total_loss + ce_loss.float().sum()
                total_tokens = total_tokens + ce_loss.numel()

        # MSE loss for generation tasks (if present)
        if mse_loss is not None:
            mse_tokens = mse_loss.shape[0]
            if mse_tokens == 0:
                mse_tokens = 1
            mse_tokens = torch.tensor(mse_tokens, dtype=torch.int, device='cuda')
            total_tokens = total_tokens + mse_tokens
            mse_loss = mse_loss.float().mean(dim=-1).sum()/mse_tokens.float()
            total_loss = total_loss + mse_loss

        # Ensure total_tokens is at least 1 to avoid division by zero
        if total_tokens == 0:
            total_tokens = torch.tensor(1, dtype=torch.int, device='cuda')

        reporting_loss = torch.cat([total_loss.clone().detach().view(1), total_tokens.float().view(1)])
        reporting_mse_loss = torch.cat([mse_loss.clone().detach().view(1), mse_tokens.float().view(1)])
        return (total_loss.bfloat16(), total_tokens, {'lm loss': reporting_loss, 'mse loss': reporting_mse_loss})

    # Standard MIMO output format (tensor)
    losses = output_tensor.float()

    loss_mask = loss_mask.contiguous().view(-1).float()

    total_tokens = loss_mask.sum().clone().detach().to(torch.int)
    total_loss = torch.sum(losses.view(-1) * loss_mask)
    reporting_loss = torch.cat([total_loss.clone().detach().view(1), total_tokens.view(1)])

    return (total_loss, total_tokens, {'lm loss': reporting_loss})


def forward_step(data_iterator, model):
    """Forward step for MIMO model training.

    Args:
        data_iterator: iterator over the dataset
        model: MIMO model instance

    Returns:
        tuple: (output_tensor, loss_function)
    """
    data_batch = get_batch(data_iterator)
    output_tensor, loss_mask = model(**data_batch)
    # Return output and loss function
    return output_tensor, partial(loss_func, loss_mask)


def train_valid_test_datasets_provider(*provider_args, **provider_kwargs):
    """Dataset provider for MIMO model training.

    Args:
        *provider_args: Additional arguments for the dataset provider
        **provider_kwargs: Additional keyword arguments for the dataset provider
    """
    runtime_args = get_args()
    try:
        dataset_provider = _DATASET_PROVIDERS[runtime_args.dataset_provider]
    except KeyError as e:
        raise ValueError(
            f"Unsupported dataset provider '{runtime_args.dataset_provider}'. "
            f"Available providers: {list(_DATASET_PROVIDERS.keys())}"
        ) from e

    return dataset_provider(*provider_args, **provider_kwargs)

def model_provider(
    pre_process: bool = True,
    post_process: bool = True,
    add_encoder: bool = True,
    add_decoder: bool = True,
    image_special_token_id: int = 32000,
    audio_special_token_id: int = 32002,
):
    """Model provider for MIMO model training.

    Args:
        pre_process: Whether to pre-process the model
        post_process: Whether to post-process the model
        add_encoder: Whether to add an encoder to the model (not supported yet)(default: True)
        add_decoder: Whether to add a decoder to the model (not supported yet)(default: True)
        image_special_token_id: Special token ID for the image modality (default: 32000)
        audio_special_token_id: Special token ID for the audio modality (default: 32002)
    """
    runtime_args = get_args()

    try:
        builder_fn = _MODEL_PROVIDERS[runtime_args.model_provider]
    except KeyError as e:
        raise ValueError(
            f"Unsupported model provider '{runtime_args.model_provider}'. "
            f"Available providers: {list(_MODEL_PROVIDERS.keys())}"
        ) from e

    if runtime_args.model_provider == "llava_vlm":
        kwargs = {
            "image_special_token_id": image_special_token_id,
        }
    elif runtime_args.model_provider == "llava_avlm":
        kwargs = {
            "image_special_token_id": image_special_token_id,
            "audio_special_token_id": audio_special_token_id,
        }
    elif runtime_args.model_provider == "bagel":
        language_use_mcore = getattr(runtime_args, 'language_use_mcore', False)
        print(f"Using {'Megatron Core GPTModel-based language model' if language_use_mcore else 'HuggingFace BagelLLMHuggingFaceModel'}")
        kwargs = {
            "image_special_token_id": image_special_token_id,
            "model_path": runtime_args.model_path,
            "language_use_mcore": language_use_mcore,
            "llm_path": runtime_args.llm_path,
            "vit_path": runtime_args.vit_path,
        }
    elif runtime_args.model_provider == "bagel_mot":
        language_use_mcore = getattr(runtime_args, 'language_use_mcore', False)
        print(f"Using {'Megatron Core GPTModel-based language model' if language_use_mcore else 'HuggingFace BagelLLMHuggingFaceModel'}")
        kwargs = {
            "image_special_token_id": image_special_token_id,
            "model_path": runtime_args.model_path,
            "language_use_mcore": language_use_mcore,
            "decoder_layer_module": "Qwen2MoTDecoderLayer",
            "llm_path": runtime_args.llm_path,
            "vit_path": runtime_args.vit_path,
        }
    elif runtime_args.model_provider == "mock":
        kwargs = {
            "special_token_id": image_special_token_id,
        }
    else:
        raise ValueError(f"Unknown model provider: {runtime_args.model_provider}. Must be one of ['llava_vlm', 'llava_avlm', 'bagel', 'bagel_mot', 'mock']")

    return builder_fn(
        pre_process,
        post_process,
        add_encoder,
        add_decoder,
        **kwargs,
    )


if __name__ == "__main__":

    train_valid_test_datasets_provider.is_distributed = True
    pretrain(
        train_valid_test_datasets_provider,
        model_provider,
        ModelType.encoder_or_decoder,
        forward_step,
        args_defaults={},
        extra_args_provider=add_mimo_args,
    )
