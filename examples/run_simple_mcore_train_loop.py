# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.

import os
from functools import partial
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, Tuple

import torch
from torch.optim import Adam
from torch.utils.data import DataLoader

from megatron.core import dist_checkpointing, parallel_state
from megatron.core.datasets.blended_megatron_dataset_builder import BlendedMegatronDatasetBuilder
from megatron.core.datasets.gpt_dataset import GPTDatasetConfig, MockGPTDataset
from megatron.core.datasets.utils import compile_helpers
from megatron.core.distributed import DistributedDataParallel, DistributedDataParallelConfig
from megatron.core.distributed.finalize_model_grads import finalize_model_grads
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_local_spec
from megatron.core.models.gpt.gpt_model import GPTModel
from megatron.core.pipeline_parallel.schedules import get_forward_backward_func
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.tokenizers import MegatronTokenizer
from megatron.core.transformer.transformer_config import TransformerConfig

_SEQUENCE_LENGTH: int = 64


def initialize_distributed(
    tensor_model_parallel_size: int = 1, pipeline_model_parallel_size: int = 1
) -> None:
    """
    Set up torch.distributed and Megatron-Core model parallel groups.

    Args:
        tensor_model_parallel_size (int): Number of GPUs to use for tensor model parallelism.
        pipeline_model_parallel_size (int): Number of GPUs to use for pipeline model parallelism.
    """
    parallel_state.destroy_model_parallel()

    # Torch setup for distributed training
    rank: int = int(os.environ["RANK"])
    world_size: int = int(os.environ["WORLD_SIZE"])
    local_rank: int = int(os.environ["LOCAL_RANK"])

    torch.cuda.set_device(local_rank)
    torch.distributed.init_process_group(backend="nccl", rank=rank, world_size=world_size)

    # Megatron core distributed training initialization
    parallel_state.initialize_model_parallel(
        tensor_model_parallel_size, pipeline_model_parallel_size
    )


def model_provider() -> GPTModel:
    """
    Construct a minimal GPT model for demonstration and testing purposes.

    Returns:
        GPTModel: A small GPT model instance with 2 layers.
    """
    transformer_config: TransformerConfig = TransformerConfig(
        num_layers=2,
        hidden_size=12,
        num_attention_heads=4,
        use_cpu_initialization=True,
        pipeline_dtype=torch.float32,
    )

    gpt_model: GPTModel = GPTModel(
        config=transformer_config,
        transformer_layer_spec=get_gpt_layer_local_spec(),
        vocab_size=100,
        max_sequence_length=_SEQUENCE_LENGTH,
    )

    return gpt_model


def get_train_data_iterator() -> Iterator:
    """
    Initialize and return an iterator over the training dataset for the GPT model.

    This function sets up a mock dataset using the provided configuration and tokenizer, builds the dataset,
    and returns an iterator for use in the training loop. It ensures that helper functions are compiled
    across distributed processes if running in a distributed environment.

    Returns:
        Iterator: An iterator that yields training batches for the GPT model.
    """
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        if torch.distributed.get_rank() == 0:
            compile_helpers()
        torch.distributed.barrier()
    else:
        compile_helpers()

    config: GPTDatasetConfig = GPTDatasetConfig(
        random_seed=0,
        sequence_length=_SEQUENCE_LENGTH,
        reset_position_ids=False,
        reset_attention_mask=False,
        eod_mask_loss=False,
        tokenizer=MegatronTokenizer.from_pretrained(
            metadata_path={"library": "null-text"}, vocab_size=_SEQUENCE_LENGTH
        ),
        mid_level_dataset_surplus=0.005,
    )

    datasets = BlendedMegatronDatasetBuilder(
        MockGPTDataset, [1000, None, None], lambda: True, config
    ).build()

    train_dataloader: DataLoader = DataLoader(datasets[0], batch_size=8, shuffle=True)

    train_iterator: Iterator = iter(train_dataloader)

    return train_iterator


def forward_step_func(
    data_iterator: Iterator, model: torch.nn.Module
) -> Tuple[torch.Tensor, Callable]:
    """
    Perform a forward pass on a batch of training data and return the model output and loss function.

    This function retrieves the next batch from the data iterator, moves all tensors to the appropriate device,
    and computes the model's output tensor. It also defines and returns a loss function, partially applied with the
    current loss mask, for use in the training loop.

    Args:
        data_iterator (Iterator): Iterator yielding training batches as dictionaries of tensors.
        model (torch.nn.Module): The GPT model to be trained.

    Returns:
        Tuple[torch.Tensor, Callable]:
            - output_tensor: The output tensor from the model's forward pass.
            - loss_function: A callable that computes the loss when invoked with the model output.
    """

    def loss_func(
        loss_mask: torch.Tensor, output_tensor: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        losses: torch.Tensor = output_tensor.float()
        loss_mask = loss_mask.view(-1).float()
        loss: torch.Tensor = torch.sum(losses.view(-1) * loss_mask) / loss_mask.sum()
        # If you have data parallel reduce loss across data parallel groups.
        # If pipeline parallel, loss computation is done only in last stage.

        return loss, {"lm loss": loss}

    data: Dict[str, torch.Tensor] = next(data_iterator)
    tokens: torch.Tensor = data["tokens"].to(device)
    attention_mask: torch.Tensor = data["attention_mask"].to(device)
    position_ids: torch.Tensor = data["position_ids"].to(device)
    labels: torch.Tensor = data["labels"].to(device)
    loss_mask: torch.Tensor = data["loss_mask"].to(device)

    output_tensor: torch.Tensor = model(tokens, position_ids, attention_mask, labels=labels)

    return output_tensor, partial(loss_func, loss_mask)


def save_distributed_checkpoint(checkpoint_path: str, gpt_model: torch.nn.Module) -> None:
    """
    Save a distributed checkpoint of the GPT model using Megatron-Core utilities.

    This function extracts the underlying model if wrapped with DistributedDataParallel (DDP),
    obtains its sharded state dictionary, and saves it to the specified directory using
    Megatron-Core's distributed checkpointing mechanism.

    Args:
        checkpoint_path (str): Directory path where the checkpoint will be saved.
        gpt_model (torch.nn.Module): The GPT model to checkpoint (may be wrapped with DDP).
    """
    # Access underlying model if wrapped with DDP
    model: torch.nn.Module = gpt_model.module if hasattr(gpt_model, "module") else gpt_model
    sharded_state_dict: Dict = model.sharded_state_dict(prefix="")
    dist_checkpointing.save(sharded_state_dict=sharded_state_dict, checkpoint_dir=checkpoint_path)


def load_distributed_checkpoint(
    checkpoint_path: str, gpt_model: torch.nn.Module
) -> torch.nn.Module:
    """
    Load a distributed checkpoint into the GPT model using Megatron-Core utilities.

    This function extracts the underlying model if wrapped with DistributedDataParallel (DDP),
    loads the checkpoint from the specified directory, and updates the model's state dictionary.

    Args:
        checkpoint_path (str): Directory path from which to load the checkpoint.
        gpt_model (torch.nn.Module): The GPT model to load the checkpoint into (may be wrapped with DDP).

    Returns:
        torch.nn.Module: The model with loaded checkpoint weights.
    """
    # Access underlying model if wrapped with DDP
    model: torch.nn.Module = gpt_model.module if hasattr(gpt_model, "module") else gpt_model
    sharded_state_dict: Dict = model.sharded_state_dict(prefix="")
    checkpoint: Dict = dist_checkpointing.load(
        sharded_state_dict=sharded_state_dict, checkpoint_dir=checkpoint_path
    )
    model.load_state_dict(checkpoint)
    return gpt_model


if __name__ == "__main__":
    initialize_distributed(tensor_model_parallel_size=2, pipeline_model_parallel_size=1)
    model_parallel_cuda_manual_seed(123)

    gpt_model: GPTModel = model_provider()
    device: torch.device = torch.device("cuda")
    gpt_model.to(device)

    # Wrap model with DistributedDataParallel for proper gradient synchronization.
    # This provides the finish_grad_sync() method required by finalize_model_grads().
    config: TransformerConfig = gpt_model.config
    ddp_config: DistributedDataParallelConfig = DistributedDataParallelConfig(
        grad_reduce_in_fp32=False, overlap_grad_reduce=False, use_distributed_optimizer=False
    )
    gpt_model = DistributedDataParallel(config=config, ddp_config=ddp_config, module=gpt_model)

    optim: Adam = Adam(gpt_model.parameters())

    train_iterator: Iterator = get_train_data_iterator()

    forward_backward_func: Callable[..., Dict[str, Any]] = get_forward_backward_func()

    # Running the model for 5 iterations
    for iteration in range(5):
        optim.zero_grad()

        losses_reduced: Dict[str, Any] = forward_backward_func(
            forward_step_func=forward_step_func,
            data_iterator=train_iterator,
            model=gpt_model,
            num_microbatches=1,
            seq_length=_SEQUENCE_LENGTH,
            micro_batch_size=8,
            decoder_seq_length=_SEQUENCE_LENGTH,
            forward_only=False,
        )

        # Finalize model gradients: all-reduce across DP and TP groups.
        # This synchronizes gradients for non-tensor-parallel parameters (e.g., LayerNorm)
        # across tensor parallel ranks and all gradients across data parallel ranks.
        finalize_model_grads([gpt_model])

        optim.step()

        print(f"Iteration {iteration}: Losses reduced: {losses_reduced}")

    # Saving the model
    ckpt_path: str = os.getcwd() + "/ckpt"
    Path(ckpt_path).mkdir(exist_ok=True)
    save_distributed_checkpoint(gpt_model=gpt_model, checkpoint_path=ckpt_path)

    # Loading the model
    gpt_model = load_distributed_checkpoint(gpt_model=gpt_model, checkpoint_path=ckpt_path)
    gpt_model.to(device)
    print("Successfully loaded the model")
