# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

"""Calibrate a GPT model for FP8 scaling factors."""
import os
import sys

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir, os.path.pardir))
)
import math

import torch
import transformer_engine.pytorch as te

from megatron.core import parallel_state, tensor_parallel
from megatron.core.models.gpt import GPTModel
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_with_transformer_engine_spec
from megatron.core.pipeline_parallel.p2p_communication import recv_forward, send_forward
from megatron.core.transformer.spec_utils import import_module
from megatron.training import get_args, get_model, is_last_rank, print_rank_0
from megatron.training.arguments import core_transformer_config_from_args
from megatron.training.checkpointing import load_checkpoint
from megatron.training.initialize import initialize_megatron
from megatron.training.training import save_checkpoint_and_time
from megatron.training.utils import unwrap_model
from megatron.training.yaml_arguments import core_transformer_config_from_yaml
from tasks.finetune_utils import build_data_loader
from tasks.zeroshot_gpt.datasets import build_dataset
from tasks.zeroshot_gpt.evaluate import process_batch


def model_provider(pre_process=True, post_process=True) -> GPTModel:
    """Builds the model.

        Args:
            pre_process (bool, optional): Set to true if you need to compute embeddings. Defaults to True.
            post_process (bool, optional): Set to true if you need to want to compute output logits/loss. Defaults to True.

        Returns:
            GPTModel: The returned model. Only works for Transformer Engine implementations.
        """

    args = get_args()

    print_rank_0('building GPT model ...')

    # Experimental loading arguments from yaml
    if args.yaml_cfg is not None:
        config = core_transformer_config_from_yaml(args, "language_model")
    else:
        config = core_transformer_config_from_args(args)

    if args.use_legacy_models or args.transformer_impl != "transformer_engine":
        raise NotImplementedError(
            'Calibration is only supported for models using TransformerEngine.'
        )
    else:
        if args.spec is not None:
            transformer_layer_spec = import_module(args.spec)
        else:
            transformer_layer_spec = get_gpt_layer_with_transformer_engine_spec(
                args.num_experts, args.moe_grouped_gemm
            )
        model = GPTModel(
            config=config,
            transformer_layer_spec=transformer_layer_spec,
            vocab_size=args.padded_vocab_size,
            max_sequence_length=args.max_position_embeddings,
            pre_process=pre_process,
            post_process=post_process,
            fp16_lm_cross_entropy=args.fp16_lm_cross_entropy,
            parallel_output=True,
            share_embeddings_and_output_weights=not args.untie_embeddings_and_output_weights,
            position_embedding_type=args.position_embedding_type,
            rotary_percent=args.rotary_percent
        )

    return model


def forward_step(batch, model, config):
    """Forward step."""

    # Get the batch.
    tokens, labels, attention_mask, position_ids, loss_mask = process_batch(batch)

    args = get_args()
    args.micro_batch_size = len(labels)

    tensor_shape = (args.seq_length, args.micro_batch_size, args.hidden_size)
    input_tensor = recv_forward(tensor_shape, config)

    # Forward pass through the model.
    unwrapped_model = unwrap_model(model)
    unwrapped_model.set_input_tensor(input_tensor)
    output = model(tokens, position_ids, attention_mask)

    send_forward(output, config)

    if parallel_state.is_pipeline_last_stage():
        losses = tensor_parallel.vocab_parallel_cross_entropy(
            output.contiguous().float(), labels.contiguous()
        )
        loss = torch.sum(losses.view(-1) * loss_mask.contiguous().view(-1).float())
        return loss

    return None


def calibrate(data_loader, model):
    args = get_args()
    config = core_transformer_config_from_args(args)

    # Turn on evaluation mode which disables dropout.
    model.eval()

    total_output = 0.0
    num_examples = min(len(data_loader), args.calib_size)
    data_loader = iter(data_loader)

    with torch.no_grad():
        iteration = 0
        while iteration < num_examples - 1:
            batch = next(data_loader)
            if iteration % args.log_interval == 0:
                print_rank_0('> working on iteration: {}'.format(iteration))
            with te.fp8_autocast(enabled=False, calibrating=True), torch.autocast(
                device_type='cuda', dtype=torch.bfloat16
            ):
                output = forward_step(batch, model, config)

                # Reduce across processes.
                if parallel_state.is_pipeline_last_stage():
                    torch.distributed.all_reduce(
                        output, group=parallel_state.get_data_parallel_group()
                    )

                    total_output += output
            iteration += 1

        print_rank_0(f"Compute scaling factors with FP8 autocast ...")
        with te.fp8_autocast(enabled=True), torch.autocast(
            device_type='cuda', dtype=torch.bfloat16
        ):
            forward_step(batch, model, config)

            if parallel_state.is_pipeline_last_stage():
                torch.distributed.all_reduce(output, group=parallel_state.get_data_parallel_group())

                total_output += output

    print_rank_0(f"Saving calibrated checkpoint ...")
    save_checkpoint_and_time(
        iteration,
        [model],
        optimizer=None,
        opt_param_scheduler=None,
        num_floating_point_operations_so_far=0,
        checkpointing_context=None,
    )

    return total_output


def calibrate_and_print_results(task, data_loader, model):
    """Calibrate and print results on screen."""

    # Calibrate and save scaling factors
    output = calibrate(data_loader, model)

    string = ' validation results on {} | '.format(task)
    if is_last_rank():
        num_tokenized_tokens = data_loader.dataset.num_tokenized_tokens
        num_original_tokens = data_loader.dataset.num_original_tokens
        val_loss = output / (num_tokenized_tokens - 1)
        ppl = math.exp(min(20, val_loss))
        token_ratio = (num_tokenized_tokens - 1) / (num_original_tokens - 1)
        adjusted_ppl = math.exp(min(20, val_loss * token_ratio))
        string += 'avg loss: {:.4E} | '.format(val_loss)
        string += 'ppl: {:.4E} | '.format(ppl)
        string += 'adjusted ppl: {:.4E} | '.format(adjusted_ppl)
        string += 'token ratio: {} |'.format(token_ratio)

        length = len(string) + 1
        print('-' * length)
        print(string)
        print('-' * length)


def add_calib_args(parser):
    group = parser.add_argument_group(title='calibration')
    group.add_argument("--task", type=str, help="Calibration task to run. Defaults to WIKITEXT103.")
    group.add_argument('--valid-data', nargs='*', default=None, help='Calibration dataset')
    group.add_argument(
        '--overlapping-eval',
        type=int,
        default=32,  # Required for reusing _build_wikitext103_dataset()
        help='Sliding window for overlapping evaluation.',
    )
    group.add_argument(
        "--calib-size", type=int, default=512, help="Number of samples to use for calibration."
    )
    return parser


if __name__ == "__main__":
    initialize_megatron(
        extra_args_provider=add_calib_args,
        args_defaults={
            'tokenizer_type': 'GPT2BPETokenizer',
            'no_load_rng': True,
            'no_load_optim': True,
        },
    )

    args = get_args()

    if args.num_layers_per_virtual_pipeline_stage is not None:
        print("Interleaved pipeline schedule is not yet supported for calibration.")
        exit()

    # Set up model and load checkpoint.
    model = get_model(model_provider, wrap_with_ddp=False)
    if args.load is not None:
        _ = load_checkpoint(model, None, None)

    assert len(model) == 1, "Above condition should have caught this"
    model = model[0]

    # Setup data loader.
    dataset = build_dataset(args.task)
    dataloader = build_data_loader(
        dataset, args.micro_batch_size, args.num_workers, drop_last=False
    )

    # Run calibration.
    calibrate_and_print_results(args.task, dataloader, model)

    print_rank_0('Calibration successfully completed.')
