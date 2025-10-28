from typing import Any, Dict, Iterator, List
import torch
from examples.mimo.data.mock import MockVLMDataset
from megatron.core.pipeline_parallel.utils import is_pp_first_stage, is_pp_last_stage
from tests.unit_tests.models.heterogenous_parallel.parallel_utils import is_current_rank_in_grid
from torch.utils.data import DataLoader

def _collate_fn(batch: List[Dict], image_seq_length: int = 1024, hidden_size: int = 1024) -> Dict[str, torch.Tensor]:
    """
    Collate function for the DataLoader.

    Args:
        batch: List of dictionaries from the dataset
        image_seq_length: Sequence length for image tokens
        hidden_size: Hidden size for the vision encoder output

    Returns:
        Dictionary of batched tensors
    """
    input_ids = torch.stack([item["input_ids"] for item in batch])
    labels = torch.stack([item["labels"] for item in batch])
    loss_mask = torch.stack([item["loss_mask"] for item in batch])
    position_ids = torch.stack([item["position_ids"] for item in batch])

    bsz = input_ids.shape[0]

    return {
        "input_ids": input_ids,
        "labels": labels,
        "loss_mask": loss_mask,
        "position_ids": position_ids,
        "modality_inputs": {
            "images": {
                "clip_encoder": {'hidden_states': torch.randn(image_seq_length, bsz, hidden_size, dtype=torch.bfloat16), 'attention_mask': None},
            }
        },
    }

def move_to_device(data, device):
    """Recursively move tensors in nested dicts to device."""
    if isinstance(data, torch.Tensor):
        return data.to(device)
    elif isinstance(data, dict):
        return {k: move_to_device(v, device) for k, v in data.items()}
    return data

def get_data_iterator(encoder_grid, llm_grid, image_seq_length, seq_length, image_special_token_id, batch_size, vocab_size, vision_hidden_size):
    data_iterator = None

    # we initialize iterator on first pp stage of encoders and LLM

    encoder_1_condition =   is_current_rank_in_grid(encoder_grid) and is_pp_first_stage(
        encoder_grid.get_pg("pp")
    )
    

    llm_condition = is_current_rank_in_grid(llm_grid) and (is_pp_first_stage(
        llm_grid.get_pg("pp")
    ) or is_pp_last_stage(llm_grid.get_pg("pp")))

    if encoder_1_condition or llm_condition:
        dataset = MockVLMDataset(
            size=256,
            image_size=224,
            seq_len=seq_length,
            image_seq_length=image_seq_length,
            pad_token_id=0,
            image_token_id=image_special_token_id
        )
        dataloader =  DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            collate_fn=lambda batch: _collate_fn(batch, image_seq_length=image_seq_length, hidden_size=vision_hidden_size),
        )
        data_iterator = iter(dataloader)
    return data_iterator

def get_batch(data_iterator: Iterator[Dict[str, Any]]):
    if data_iterator is not None:
        input_tensor = next(data_iterator)
        if input_tensor is not None:
            input_tensor = move_to_device(input_tensor, torch.device("cuda"))
    else:
        input_tensor = None

    return input_tensor