
import torch
from megatron import print_rank_0

def get_ltor_attention_masks_and_position_ids(data, eod_token_id):
    """Build attention masks and position id for left to right model."""

    micro_batch_size, seq_length = data.size()

    # Attention mask
    attention_mask = torch.tril(torch.ones((micro_batch_size, seq_length, seq_length), device=data.device)).view(micro_batch_size, 1, seq_length, seq_length)

    # mask padded tokens
    for b in range(micro_batch_size):
        for idx in range(seq_length-1):
            if data[b, idx] == eod_token_id:
                # pad tokens that come after the eod token
                attention_mask[b, 0, idx+1:, :] = 0.0

    # Position ids.
    position_ids = torch.arange(seq_length, dtype=torch.long, device=data.device)
    position_ids = position_ids.unsqueeze(0).expand_as(data)

    # # reset attentino mask and position ids
    # # Loop through the batches:
    # for b in range(micro_batch_size):
    #     # Find indecies where EOD token is.
    #     eod_index = position_ids[b, data[b] == eod_token_id]
    #     eod_index = eod_index.clone()

    #     # Loop through EOD indecies:
    #     prev_index = 0
    #     for j in range(eod_index.size()[0]):
    #         i = eod_index[j]
    #         # Mask attention loss.
    #         attention_mask[b, 0, (i + 1):, :(i + 1)] = 0
    #         # Reset positions.
    #         position_ids[b, (i + 1):] -= (i + 1 - prev_index)
    #         prev_index = i + 1
    
    # Convert attention mask to binary:
    attention_mask = (attention_mask < 0.5)

    return attention_mask, position_ids
    