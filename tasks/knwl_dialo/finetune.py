
"""Dialogue Finetuning"""

import torch
from functools import partial
from megatron import mpu
from megatron import get_args
from megatron import get_timers
from megatron import print_rank_0
from megatron import get_tokenizer
from megatron.model import GPTModel
from megatron.training import evaluate_and_print_results
from megatron.training import get_model
from megatron.utils import average_losses_across_data_parallel_group
from megatron.initialize import initialize_megatron
from tasks.finetune_utils import finetune
from tasks.knwl_dialo.data import build_train_valid_datasets
from tasks.knwl_dialo.utils import get_ltor_attention_masks_and_position_ids
from tasks.knwl_dialo.utils import get_token_stream


def model_provider(pre_process=True, post_process=True):
    """Build the model."""

    print_rank_0('building GPT model ...')
    model = GPTModel(
        num_tokentypes=0,
        parallel_output=True,
        pre_process=pre_process,
        post_process=post_process
    )
    return model


def train_valid_datasets_provider():
    """Build train, valid, and test datasets for dialog/control module"""
    args = get_args()

    print_rank_0('> building train, validation, and test datasets for %s module ...' % args.train_module)
    
    train_ds, valid_ds = build_train_valid_datasets(
        train_data_path=args.train_data_path,
        valid_data_path=args.test_data_path,
        train_module=args.train_module,
        max_seq_len=args.max_seq_len,
        seed=args.seed,
        last_turn=args.last_turn,
        no_control_code=args.no_control_code,
        add_separator=args.add_separator,
        add_ctrl_code_to_dialog=args.add_ctrl_code_to_dialog,
        remove_ctrl_sent=args.remove_ctrl_sent)
        
    print_rank_0("> finished creating datasets for %s module ..." % args.train_module)
    print_rank_0('> Train size: %d' % len(train_ds))
    print_rank_0('> Validation size: %d' % len(valid_ds))

    args.eval_interval = len(train_ds) // args.global_batch_size
    print_rank_0('> evaluation interval: %d' % args.eval_interval)

    args.eval_iters = len(valid_ds) // args.global_batch_size
    print_rank_0('> evaluation iteration: %d' % args.eval_iters)

    return train_ds, valid_ds


def process_batch(batch):
    """Generate a batch"""
    args = get_args()
    tokenizer = get_tokenizer()

    # Items and their type.
    keys = ['text', 'loss_mask']
    datatype = torch.int64

    data_b = mpu.broadcast_data(keys, batch, datatype)

    tokens_ = data_b['text'].long()
    labels = tokens_[:, 1:].contiguous()
    tokens = tokens_[:, :-1].contiguous()

    loss_mask = data_b['loss_mask'].float()

    # Get the attention_mask and postition ids.
    attention_mask, position_ids = \
        get_ltor_attention_masks_and_position_ids(tokens, tokenizer.eod_id)

    return tokens, labels, loss_mask, attention_mask, position_ids


def loss_func(loss_mask, output_tensor):
    losses = output_tensor.float()
    loss_mask = loss_mask.view(-1).float()
    loss = torch.sum(losses.view(-1) * loss_mask) / loss_mask.sum()

    # Reduce loss for logging.
    averaged_loss = average_losses_across_data_parallel_group([loss])

    return loss, {'lm loss': averaged_loss[0]}


def forward_step(batch, model):
    """Forward step."""
    args = get_args()
    timers = get_timers()
    
    try:
        batch_ = next(batch)
    except BaseException:
        batch_ = batch

    tokens, labels, loss_mask, attention_mask, position_ids = process_batch(batch_)

    output_tensor = model(tokens, position_ids, attention_mask,
                          labels=labels)

    return output_tensor, partial(loss_func, loss_mask)


def generate_samples_input_from_file(model):

    args = get_args()
    tokenizer = get_tokenizer()

    # Read the sample file and open the output file.
    assert args.sample_input_file is not None, \
        'sample input file is not provided.'
    if mpu.is_pipeline_first_stage() and mpu.get_tensor_model_parallel_rank() == 0:
        fname = open(args.sample_input_file, "r")
        all_raw_text = fname.readlines()
        input_count = len(all_raw_text)
        input_pos = 0
        if args.sample_output_file is None:
            sample_output_file = args.sample_input_file + ".out"
            print('`sample-output-file` not specified, setting '
                    'it to {}'.format(sample_output_file))
        else:
            sample_output_file = args.sample_output_file

        fname_out = open(sample_output_file, "w")

    context_count = 0
    model.eval()
    with torch.no_grad():
        while True:
            raw_text_len = 0

            if mpu.is_pipeline_first_stage() \
               and mpu.get_tensor_model_parallel_rank() == 0:
                raw_text = all_raw_text[input_pos]
                input_pos += 1
                raw_text_len = len(raw_text)
                context_tokens = tokenizer.tokenize(raw_text)
            
            else:
                context_tokens = tokenizer.tokenize("EMPTY TEXT")

            if input_pos % 100 == 0:
                print_rank_0("input_pos: %d" % input_pos)

            token_stream = get_token_stream(model, [context_tokens])
            for _, decode_tokens in enumerate(token_stream):
                pass

            if mpu.get_tensor_model_parallel_rank() == 0:
                if mpu.is_pipeline_first_stage():

                    decode_tokens, _ = decode_tokens
                    decode_tokens = decode_tokens[0].cpu().numpy().tolist()
                    trim_decode_tokens = tokenizer.detokenize(
                        decode_tokens)[raw_text_len:]

                    if "\r" in trim_decode_tokens:
                        trim_decode_tokens = trim_decode_tokens.replace("\r", "")
                    if "\n" in trim_decode_tokens:
                        trim_decode_tokens = trim_decode_tokens.replace("\n", "")
                    fname_out.write(trim_decode_tokens)
                    fname_out.write("\n")

            raw_text = None
            context_count += 1

            if input_pos == input_count:
                return


def run_generation(model_provider):

    args = get_args()
    if args.num_layers_per_virtual_pipeline_stage is not None:
        print("Interleaved pipeline schedule is not yet supported for text generation.")
        exit()

    # Set up model and load checkpoint.
    model = get_model(model_provider)

    if args.load is not None:
        _ = load_checkpoint(model, None, None)

    assert len(model) == 1, "Above condition should have caught this"
    model = model[0]

    generate_samples_input_from_file(model)


def main():
    args = get_args()

    if "finetune" in args.task:
        finetune(train_valid_datasets_provider, model_provider, \
                 forward_step=forward_step)
    else:
        # generate
        run_generation(model_provider)
