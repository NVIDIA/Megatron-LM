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
    get_data_parallel_group,
    get_tensor_model_parallel_group,
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_src_rank,
)

# Add the parent directory to the path to import from megatron
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir, os.path.pardir))
)
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
    group.add_argument('--vit-cond-dropout-prob', type=float, default=0.3, help='VIT conditional dropout probability')
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

    # Multimodal FLOPS estimation (for accurate throughput reporting)
    group.add_argument('--vit-hidden-size', type=int, default=1152,
                       help='ViT hidden size for FLOPS calculation')
    group.add_argument('--vit-num-layers', type=int, default=26,
                       help='ViT number of layers for FLOPS calculation')
    group.add_argument('--vit-intermediate-size', type=int, default=4304,
                       help='ViT MLP intermediate size for FLOPS calculation')
    group.add_argument('--avg-vit-tokens-per-batch', type=int, default=4900,
                       help='Average ViT tokens per micro-batch for FLOPS estimation')
    group.add_argument('--avg-latent-tokens-per-batch', type=int, default=1024,
                       help='Average VAE latent tokens per micro-batch for FLOPS estimation')
    group.add_argument('--avg-ce-tokens-per-batch', type=int, default=0,
                       help='Average CE-supervised tokens per micro-batch for FLOPS estimation (0 disables CE logits correction)')
    group.add_argument('--vae-latent-channels', type=int, default=16,
                       help='VAE latent channel count (z_channels) used by FLOPS estimation')
    group.add_argument('--freeze-vit', action='store_true', default=False,
                       help='Whether ViT is frozen (affects FLOPS: forward-only vs fwd+bwd)')
    return parser


def get_batch(data_iterator: Iterator[Dict[str, Any]]):
    """Generate a batch for MIMO model training.

    Args:
        data_iterator: Iterator over the dataset

    Returns:
        tuple: Batch data for model training
    """
    args = get_args()

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

        # Match HF bagel: CE and MSE are normalized SEPARATELY by their own
        # global token counts, then summed. Megatron schedule will later do
        #   loss /= num_tokens /= num_microbatches
        # and grad all-reduce averages across DP ranks (/dp_size).
        # So we pre-compute the averaged loss here and return num_tokens=1
        # to make the schedule's /num_tokens a no-op.

        dp_group = get_data_parallel_group()
        dp_size = torch.distributed.get_world_size(dp_group)

        total_loss = torch.tensor(0.0, device='cuda')

        # --- CE loss: separately normalized by global CE tokens ---
        ce_loss_sum = torch.tensor(0.0, device='cuda')
        ce_tokens = torch.tensor(0, dtype=torch.int, device='cuda')
        if ce_loss is not None:
            num_ce = ce_loss.numel()
            ce_loss_sum = ce_loss.float().sum()
            ce_tokens = torch.tensor(num_ce, dtype=torch.int, device='cuda')
            # All-reduce to get global CE token count
            global_ce_tokens = ce_tokens.clone().float()
            torch.distributed.all_reduce(global_ce_tokens, op=torch.distributed.ReduceOp.SUM, group=dp_group)
            global_ce_tokens = torch.clamp(global_ce_tokens, min=1.0)
            # HF: ce = ce.sum() * world_size / total_ce_tokens
            # After DDP grad avg (/dp_size): effective = ce.sum() / total_ce_tokens (global avg)
            # We need schedule to produce the same. Schedule does: loss / num_tokens / num_microbatches,
            # then grad avg /dp_size. With num_tokens=1, num_microbatches=1:
            #   effective = loss / dp_size → so loss = ce_avg * dp_size
            ce_avg = ce_loss_sum * dp_size / global_ce_tokens
            total_loss = total_loss + ce_avg

        # --- MSE loss: separately normalized by global MSE tokens ---
        mse_loss_sum = torch.tensor(0.0, device='cuda')
        mse_tokens = torch.tensor(0, dtype=torch.int, device='cuda')
        if mse_loss is not None:
            num_mse = mse_loss.shape[0]
            if num_mse == 0:
                num_mse = 1
            mse_tokens = torch.tensor(num_mse, dtype=torch.int, device='cuda')
            mse_loss_sum = mse_loss.float().mean(dim=-1).sum()
            # All-reduce to get global MSE token count
            global_mse_tokens = mse_tokens.clone().float()
            torch.distributed.all_reduce(global_mse_tokens, op=torch.distributed.ReduceOp.SUM, group=dp_group)
            global_mse_tokens = torch.clamp(global_mse_tokens, min=1.0)
            mse_avg = mse_loss_sum * dp_size / global_mse_tokens
            total_loss = total_loss + mse_avg

        # num_tokens=1 so schedule's /num_tokens is a no-op (we already averaged above)
        num_tokens = torch.tensor(1, dtype=torch.int, device='cuda')

        # Reporting: CE and MSE are reported individually with [raw_sum, count] —
        # training.py all-reduces these correctly to get per-token averages.
        reporting_mse = torch.cat([mse_loss_sum.clone().detach().view(1), mse_tokens.float().view(1)])
        reporting_ce = torch.cat([ce_loss_sum.clone().detach().view(1), ce_tokens.float().view(1)])
        # Total loss reporting: must match HF's "ce_avg + mse_avg" (separately normalized).
        # training.py does: all_reduce([sum, count]) then sum/count.
        # Store [local_avg, 1/dp_size]. After all_reduce(SUM) across dp_size ranks:
        #   sum = Σ local_avg_i = global_ce_sum/global_ce + global_mse_sum/global_mse
        #   count = dp_size * (1/dp_size) = 1
        #   result = (ce_avg + mse_avg) / 1 = ce_avg + mse_avg ✓
        local_loss_avg = torch.tensor(0.0, device='cuda')
        if ce_loss is not None:
            local_loss_avg = local_loss_avg + ce_loss_sum.clone().detach() / global_ce_tokens
        if mse_loss is not None:
            local_loss_avg = local_loss_avg + mse_loss_sum.clone().detach() / global_mse_tokens
        inv_dp = torch.tensor(1.0 / dp_size, device='cuda')
        reporting_loss = torch.cat([local_loss_avg.view(1), inv_dp.view(1)])

        return (total_loss.bfloat16(), num_tokens, {'loss': reporting_loss, 'mse': reporting_mse, 'ce': reporting_ce})

    # Standard MIMO output format (tensor)
    losses = output_tensor.float()

    loss_mask = loss_mask.contiguous().view(-1).float()

    total_tokens = loss_mask.sum().clone().detach().to(torch.int)
    total_loss = torch.sum(losses.view(-1) * loss_mask)
    reporting_loss = torch.cat([total_loss.clone().detach().view(1), total_tokens.view(1)])

    return (total_loss, total_tokens, {'lm loss': (reporting_loss)})


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
    config=None,
    pg_collection=None,
    vp_stage=None,
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
