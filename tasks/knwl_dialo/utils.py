
"""Utils (functions) for both prompting and finetuning"""

import torch
from megatron import mpu
from megatron import get_args
from megatron import get_tokenizer
from megatron.utils import get_ltor_masks_and_position_ids, unwrap_model
from megatron.p2p_communication import recv_forward, send_forward
from torch.nn.parallel.distributed import DistributedDataParallel as torchDDP
from megatron.model import DistributedDataParallel as LocalDDP
from megatron.model import Float16Module


def get_ltor_attention_masks_and_position_ids(data, eod_token_id):
    """
    Build attention masks and position id for left to right model.
    Different from the existing get_ltor_masks_and_position_ids function,
    we add padding to the input sequences to make sure their lengths are the same.
    """

    micro_batch_size, seq_length = data.size()

    # Attention mask
    attention_mask = torch.tril(torch.ones(
        (micro_batch_size, seq_length, seq_length), device=data.device)).view(
            micro_batch_size, 1, seq_length, seq_length)

    # mask padded tokens
    for b in range(micro_batch_size):
        for idx in range(seq_length-1):
            if data[b, idx] == eod_token_id:
                # pad tokens that come after the eod token
                attention_mask[b, 0, idx+1:, :] = 0.0

    # Position ids.
    position_ids = torch.arange(seq_length, dtype=torch.long, device=data.device)
    position_ids = position_ids.unsqueeze(0).expand_as(data)
    
    # Convert attention mask to binary:
    attention_mask = (attention_mask < 0.5)

    return attention_mask, position_ids


def switch(val1, val2, boolean):
    """Return either val1 or val2 depending on boolean"""

    boolean = boolean.type_as(val1)
    return (1 - boolean) * val1 + boolean * val2


def forward_step(model, tokens, position_ids, attention_mask, tokentype_ids,
                 layer_past=None, get_key_value=None,
                 forward_method_parallel_output=None):
    """Forward step to get the outputs"""
    
    # functions the correct size
    args = get_args()
    orig_seq_length = args.seq_length
    args.seq_length = tokens.shape[1]

    input_tensor = recv_forward()

    # Forward pass through the model.
    unwrapped_model = unwrap_model(
        model, (torchDDP, LocalDDP, Float16Module))
    unwrapped_model.set_input_tensor(input_tensor)
    output_tensor = model(tokens, position_ids, attention_mask,
                          tokentype_ids=tokentype_ids)

    if get_key_value:
        output_tensor, layer_past = output_tensor

    send_forward(output_tensor)

    args.seq_length = orig_seq_length
    if get_key_value:
        return output_tensor, layer_past
    return output_tensor
    

def pad_batch(batch, pad_id, args):
    """Pad the context tokens using pad_id"""

    context_lengths = []
    for tokens in batch:
        context_length = len(tokens)
        # padding
        if context_length < args.seq_length:
            tokens.extend([pad_id] * (args.seq_length - context_length))
        # record the original context length
        context_lengths.append(context_length)
    return batch, context_lengths


def get_batch(context_tokens):
    """Generate batch from context tokens."""

    args = get_args()
    tokenizer = get_tokenizer()

    # Move to GPU.
    tokens = context_tokens.view(args.micro_batch_size, -1).contiguous().cuda()
    # Get the attention mask and postition ids for the context tokens.
    attention_mask, _, position_ids = get_ltor_masks_and_position_ids(
        tokens,
        tokenizer.eod,
        args.reset_position_ids,
        args.reset_attention_mask,
        args.eod_mask_loss)

    return tokens, attention_mask, position_ids


def sample_sequence_batch(model, context_tokens, context_lengths,
                          attention_mask, position_ids,
                          maxlen=None, type_ids=None):
    """Obtain batch-level generation outputs"""

    args = get_args()
    tokenizer = get_tokenizer()

    model.eval()
    with torch.no_grad():
        context_length = context_lengths.min().item()

        # added eos_id to support the function generate_samples_eval that passes
        # eos_id as an argument and needs termination when that id id found.
        if hasattr(args, 'eos_id'):
            eos_id = args.eos_id
        else:
            eos_id = tokenizer.eod

        counter = 0
        org_context_length = context_length

        # prepare batch size, context tokens, maximum length
        layer_past = None
        batch_size = context_tokens.size(0)
        is_done = torch.zeros([batch_size]).byte().cuda()
        tokens = context_tokens
        if maxlen is None:
            maxlen = args.seq_length - 1
        lengths = torch.ones([batch_size]).long().cuda() * maxlen

        # start the generation process
        while context_length <= (maxlen):
            # forward and obtain the logits
            output = forward_step(model, tokens,
                                    position_ids,
                                    attention_mask,
                                    tokentype_ids=type_ids,
                                    forward_method_parallel_output=False)
            if mpu.is_pipeline_last_stage():
                assert output is not None
                logits = output[:, context_length - 1, :]
            
            # generate tokens iteratively
            if mpu.is_pipeline_last_stage():
                prev = torch.argmax(logits, dim=-1).view(-1)
                
                # start to add new tokens when the generated length
                # exceeds the context length
                started = context_lengths <= context_length
                new_tokens = switch(
                    tokens[:, context_length].view(-1), prev, started)
                tokens[:, context_length] = new_tokens
                src = mpu.get_pipeline_model_parallel_last_rank()
                group = mpu.get_embedding_group()
                torch.distributed.broadcast(new_tokens, src, group)

                # check whether the generation is finished
                done_token = (prev == eos_id).byte() & started.byte()
                just_finished = (done_token & ~is_done).bool()
                lengths[just_finished.view(-1)] = context_length
                is_done = is_done | done_token

                done = torch.all(is_done)
                src = mpu.get_pipeline_model_parallel_last_rank()
                group = mpu.get_pipeline_model_parallel_group()
                torch.distributed.broadcast(done, src, group)
                yield tokens, lengths

            else:
                if mpu.is_pipeline_first_stage():
                    src = mpu.get_pipeline_model_parallel_last_rank()
                    group = mpu.get_embedding_group()
                    new_tokens = torch.empty_like(tokens[:, context_length])
                    torch.distributed.broadcast(new_tokens, src, group)
                    tokens[:, context_length] = new_tokens
                    yield tokens, None
                else:
                    yield None, None

                done = torch.cuda.ByteTensor([0])
                src = mpu.get_pipeline_model_parallel_last_rank()
                group = mpu.get_pipeline_model_parallel_group()
                torch.distributed.broadcast(done, src, group)

            context_length += 1
            counter += 1
            if done:
                break


def get_token_stream(model, context_tokens):
    """Get output tokens iteratively"""

    # get tokenizer
    args = get_args()
    tokenizer = get_tokenizer()

    # padding for context tokens
    context_tokens, context_lengths = pad_batch(context_tokens,
                                                tokenizer.eod, args)

    # move tokens to CUDA
    context_tokens_tensor = torch.cuda.LongTensor(context_tokens)
    context_length_tensor = torch.cuda.LongTensor(context_lengths)

    torch.distributed.broadcast(context_length_tensor,
                                mpu.get_tensor_model_parallel_src_rank(),
                                group=mpu.get_tensor_model_parallel_group())
    torch.distributed.broadcast(context_tokens_tensor,
                                mpu.get_tensor_model_parallel_src_rank(),
                                group=mpu.get_tensor_model_parallel_group())

    # prepare batch
    context_length = context_length_tensor.min().item()
    tokens, attention_mask, position_ids = get_batch(context_tokens_tensor)

    # get generation outputs
    batch_token_iterator = sample_sequence_batch(model, context_tokens_tensor,
                                                 context_length_tensor,
                                                 attention_mask, position_ids)
    for tokens, lengths in batch_token_iterator:
        context_length += 1
        if tokens is not None:
            yield tokens[:, :context_length], lengths
        else:
            yield None, None


