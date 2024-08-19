# Copyright (C) 2024 Habana Labs, Ltd. an Intel Company.
# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.

"""General utilities."""

import sys
import os

import torch
from torch.nn.parallel import DistributedDataParallel as torchDDP

from deepspeed.accelerator import get_accelerator
if get_accelerator().device_name() == 'cuda':
    from apex.multi_tensor_apply import multi_tensor_applier
    import amp_C

from megatron import (
    get_args,
    get_adlr_autoresume,
    get_num_microbatches
)
from megatron.core import mpu
from megatron.core.tensor_parallel import param_is_not_tensor_parallel_duplicate
from megatron.model.module import param_is_not_shared
from megatron.model.rotary_pos_embedding import RotaryEmbedding


def update_rotary_pos_emb(seq_length):
    args = get_args()
    rotary_dim = args.hidden_size // args.num_attention_heads \
        if args.kv_channels is None else args.kv_channels

    if args.rotary_percent < 1.0:
        rotary_dim = int(rotary_dim * args.rotary_percent)

    # partial rotary embeddings, which is better than full rotary
    # Wang and Komatsuzaki et al
    # https://github.com/kingoflolz/mesh-transformer-jax/
    rotary_pos_emb = RotaryEmbedding(rotary_dim, theta=args.rope_theta)(seq_length).to(
        get_accelerator().current_device_name())
    args.rotary_pos_emb = rotary_pos_emb


def unwrap_model(model, module_instances=(torchDDP)):
    return_list = True
    if not isinstance(model, list):
        model = [model]
        return_list = False
    unwrapped_model = []
    for model_module in model:
        while isinstance(model_module, module_instances):
            model_module = model_module.module
        unwrapped_model.append(model_module)
    if not return_list:
        return unwrapped_model[0]
    return unwrapped_model


def calc_params_l2_norm(model):
    """Calculate l2 norm of parameters """
    args = get_args()
    if not isinstance(model, list):
        model = [model]
    # Remove duplicate params.
    params_data = []
    for model_ in model:
        for param in model_.parameters():
            is_not_shared = param_is_not_shared(param)
            is_not_tp_duplicate = param_is_not_tensor_parallel_duplicate(param)
            if is_not_shared and is_not_tp_duplicate:
                if args.bf16:
                    params_data.append(param.data.float())
                else:
                    params_data.append(param.data)
    # Calculate norm
    dummy_overflow_buf = get_accelerator().IntTensor([0])
    
    if get_accelerator().device_name() == 'cuda':

        norm, _ = multi_tensor_applier(
            amp_C.multi_tensor_l2norm,
            dummy_overflow_buf,
            [params_data],
            False # no per-parameter norm
        )
    else :
        norm = torch.norm(params_data,p=2.0)
    norm_2 = norm * norm
    # Sum across all model-parallel GPUs.
    torch.distributed.all_reduce(norm_2,
                                 op=torch.distributed.ReduceOp.SUM,
                                 group=mpu.get_model_parallel_group())
    return norm_2.item() ** 0.5


def average_losses_across_data_parallel_group(losses):
    """Reduce a tensor of losses across all GPUs."""
    averaged_losses = torch.cat(
        [loss.clone().detach().view(1) for loss in losses])
    torch.distributed.all_reduce(averaged_losses,
                                 group=mpu.get_data_parallel_group())
    averaged_losses = averaged_losses / \
        torch.distributed.get_world_size(group=mpu.get_data_parallel_group())

    return averaged_losses


def report_memory(name):
    """Simple GPU memory report."""
    mega_bytes = 1024.0 * 1024.0
    string = name + ' memory (MB)'
    string += ' | allocated: {}'.format(
        get_accelerator().memory_allocated() / mega_bytes)
    string += ' | max allocated: {}'.format(
        get_accelerator().max_memory_allocated() / mega_bytes)
    string += ' | reserved: {}'.format(
        get_accelerator().memory_reserved() / mega_bytes)
    string += ' | max reserved: {}'.format(
        get_accelerator().max_memory_reserved() / mega_bytes)
    if mpu.get_data_parallel_rank() == 0:
        print("[Rank {}] {}".format(torch.distributed.get_rank(), string),
              flush=True)


def print_params_min_max_norm(optimizer, iteration):
    """Print min, max, and norm of all parameters."""
    index = 0
    rank = torch.distributed.get_rank()
    string = 'iteration, rank, index, tensor-model-parallel, min, max, norm\n'
    optimizer_ = optimizer.optimizer
    for param_group in optimizer_.param_groups:
        for param in param_group['params']:
            index += 1
            min_ = param.data.min()
            max_ = param.data.max()
            norm = torch.linalg.norm(param.data)
            string += '{:7d}, {:4d}, {:4d}, {:2d}, '.format(
                iteration, rank, index, int(param.tensor_model_parallel))
            string += '{:.6E}, {:.6E}, {:.6E}\n'.format(min_, max_, norm)
    print(string, flush=True)


def check_adlr_autoresume_termination(iteration, model,
                                      optimizer, opt_param_scheduler):
    """Check for autoresume signal and exit if it is received."""
    from megatron.checkpointing import save_checkpoint

    args = get_args()
    autoresume = get_adlr_autoresume()
    # Add barrier to ensure consistnecy.
    torch.distributed.barrier()
    if autoresume.termination_requested():
        if args.save:
            save_checkpoint(iteration, model, optimizer, opt_param_scheduler)
        print_rank_0(">>> autoresume termination request found!")
        if torch.distributed.get_rank() == 0:
            autoresume.request_resume()
        print_rank_0(">>> training terminated. Returning")
        sys.exit(0)


def get_ltor_masks_and_position_ids(data,
                                    eod_token,
                                    reset_position_ids,
                                    reset_attention_mask,
                                    eod_mask_loss,
                                    skip_mask=False):
    """Build masks and position id for left to right model."""

    # Extract batch size and sequence length.
    micro_batch_size, seq_length = data.size()

    # Attention mask (lower triangular).
    if reset_attention_mask:
        att_mask_batch = micro_batch_size
    else:
        att_mask_batch = 1

    attention_mask = None
    if not skip_mask:
        attention_mask = torch.tril(torch.ones(
            (att_mask_batch, seq_length, seq_length), device=data.device)).view(att_mask_batch, 1, seq_length, seq_length)

    # Loss mask.
    loss_mask = torch.ones(data.size(), dtype=torch.float, device=data.device)
    if eod_mask_loss:
        loss_mask[data == eod_token] = 0.0

    # Position ids.
    position_ids = torch.arange(seq_length, dtype=torch.long,
                                device=data.device)
    position_ids = position_ids.unsqueeze(0).expand_as(data)
    # We need to clone as the ids will be modifed based on batch index.
    if reset_position_ids:
        position_ids = position_ids.clone()

    if reset_position_ids or reset_attention_mask:
        # Loop through the batches:
        for b in range(micro_batch_size):

            # Find indecies where EOD token is.
            eod_index = position_ids[b, data[b] == eod_token]
            # Detach indecies from positions if going to modify positions.
            if reset_position_ids:
                eod_index = eod_index.clone()

            # Loop through EOD indecies:
            prev_index = 0
            for j in range(eod_index.size()[0]):
                i = eod_index[j]
                # Mask attention loss.
                if reset_attention_mask and not skip_mask:
                    attention_mask[b, 0, (i + 1):, :(i + 1)] = 0
                # Reset positions.
                if reset_position_ids:
                    position_ids[b, (i + 1):] -= (i + 1 - prev_index)
                    prev_index = i + 1

    # Convert attention mask to binary:
    if not skip_mask:
        attention_mask = (attention_mask < 0.5)

    return attention_mask, loss_mask, position_ids


def print_rank_0(message):
    """If distributed is initialized, print only on rank 0."""
    if torch.distributed.is_initialized():
        if torch.distributed.get_rank() == 0:
            print(message, flush=True)
    else:
        print(message, flush=True)

def is_last_rank():
    return torch.distributed.get_rank() == (
        torch.distributed.get_world_size() - 1)

def print_rank_last(message):
    """If distributed is initialized, print only on last rank."""
    if torch.distributed.is_initialized():
        if is_last_rank():
            print(message, flush=True)
    else:
        print(message, flush=True)

def is_aml():
    # Are we running inside an Azure Machine Learning (AML) environment?
    return 'AZUREML_EXPERIMENT_ID' in os.environ

def is_rank_0():
    """Check whether it is rank 0. For AML, check if it is rank 0 of a node"""
    if torch.distributed.is_initialized():
        if torch.distributed.get_rank() == 0 or (
            is_aml() and torch.distributed.get_rank() % get_accelerator().device_count() == 0
            ):
            return True
        else:
            return False
    else:
        return True

def get_parameters_in_billions(model):
    gpus_per_model = torch.distributed.get_world_size(group=mpu.get_model_parallel_group())

    approx_parameters_in_billions = sum([sum([p.ds_numel if hasattr(p,'ds_id') else  p.nelement() for p in model_module.parameters()])
                                        for model_module in model])

    return approx_parameters_in_billions*gpus_per_model/(1e9)

def throughput_calculator(model, args, iteration_time, total_iterations):
    batch_size = args.micro_batch_size * get_num_microbatches() * args.data_parallel_size
    approx_parameters_in_billions = None if (model is None) else get_parameters_in_billions(model)
    elapsed_time_per_iter = iteration_time/total_iterations
    samples_per_second = batch_size / elapsed_time_per_iter

    #flops calculator
    hidden_size = args.hidden_size
    num_attention_heads = args.num_attention_heads
    head_dim = hidden_size // num_attention_heads
    ffn_hidden_size = args.ffn_hidden_size
    num_layers = args.num_layers
    vocab_size = args.padded_vocab_size
    gqa = args.num_attention_heads // args.num_key_value_heads
    ffn_multiplier = 3 if args.swiglu else 2
    macs_per_flops = 2

    # General TFLOPs formula (borrowed from Equation 3 in Section 5.1 of
    # https://arxiv.org/pdf/2104.04473.pdf).
    # correction has been made to TFLOPs formula due to incorrect behavior
    # observed with selective recompute when GQA not used and for all with GQA
    seq_len = args.seq_length
    if hasattr(args, 'actual_seq_length'):
        seq_len = args.actual_seq_length

    pre_and_post_mha_gemm_macs = batch_size * num_layers * (1 + (2 // gqa) + 1) * (hidden_size**2) * seq_len
    mha_bgemm_macs = batch_size * num_layers * 2 * head_dim * num_attention_heads * (seq_len**2)
    ffn_gemm_macs = batch_size * num_layers * ffn_multiplier * ffn_hidden_size * hidden_size * seq_len
    logit_lmhead_gemm_macs = batch_size * vocab_size * hidden_size * seq_len

    fwd_macs = pre_and_post_mha_gemm_macs + mha_bgemm_macs + ffn_gemm_macs + logit_lmhead_gemm_macs
    bwd_macs = 2 * fwd_macs
    fwd_bwd_macs = fwd_macs + bwd_macs

    if (hasattr(args, 'checkpoint_activations') and args.checkpoint_activations) or (hasattr(args, 'recompute_granularity') and args.recompute_granularity == 'full'):
        fwd_bwd_macs += fwd_macs
    if hasattr(args, 'recompute_granularity') and args.recompute_granularity == 'selective':
        fwd_bwd_macs += mha_bgemm_macs

    flops_per_iteration = fwd_bwd_macs * macs_per_flops
    tflops = flops_per_iteration / (elapsed_time_per_iter * args.world_size * (10**12))
    return samples_per_second, tflops, approx_parameters_in_billions

def checkpoint_throughput_calculator(model, latency_second):
    approx_parameters_in_billions = get_parameters_in_billions(model)
    checkpoint_multiplier = 14  # fp16 weights (2), fp32 weights (4), fp32 momentum (4), fp32 variance (4)
    checkpoint_GB = approx_parameters_in_billions * checkpoint_multiplier
    GB_per_second = checkpoint_GB / latency_second
    print_rank_0(f"Checkpoint Save GB: {round(checkpoint_GB, 3)}, GB/Sec: {round(GB_per_second,2)}, Latency(second): {round(latency_second, 3)}")


def get_fingerprint_header():
    return f"{'min':^13} {'max':^13} {'mean':^13} {'l2 norm':^12} metadata"

def get_fingerprint(p):
    return f"{p.min():13.6e} {p.max():13.6e} {p.mean():13.6e} {p.norm():12.6e}"


def dump_position_embed_weights(preamble, iteration, model):
    # return 
    from deepspeed.utils import safe_get_full_fp32_param
    tp_rank = mpu.get_tensor_model_parallel_rank()
    pp_rank = mpu.get_pipeline_model_parallel_rank()
    dp_rank = mpu.get_data_parallel_rank()
    get_fingerprint_header()
    for n, p in model[0].named_parameters():
        if 'position_embeddings' in n:
            tag = "pos_embed"
        elif "word_embeddings" in n:
            tag = "word_embed"
        else:
            continue 
        print(f"iter {iteration} {preamble} {tag} lp {tp_rank}/{pp_rank}/{dp_rank}: {get_fingerprint(p)} {p.shape}\n")
        fp32_value = safe_get_full_fp32_param(p)
        if fp32_value is not None: 
            print(f"iter {iteration} {preamble} {tag} hp {tp_rank}/{pp_rank}/{dp_rank}: {get_fingerprint(fp32_value)} {p.shape}\n")

def dump_weights(preamble, iteration, model, optimizer, tensor=None):
    # return
    tp_rank = mpu.get_tensor_model_parallel_rank()
    pp_rank = mpu.get_pipeline_model_parallel_rank()
    dp_rank = mpu.get_data_parallel_rank()
    dp_size = mpu.get_data_parallel_world_size()
    fn = f"debug-bf16-{iteration}-pp{pp_rank}-tp{tp_rank}-dp{dp_rank}-{preamble}.txt"

    # only care for first and last pp stages and dp0 tp0
    #if not (mpu.is_pipeline_first_stage() or mpu.is_pipeline_last_stage()):
    #    return

    #if not (tp_rank == 0 and dp_rank == 0):
    #    return

    if tensor is not None:
        orig_tensor = tensor
        if hasattr(tensor, "_hp_param"):
            numel = tensor._hp_param.numel() # // dp_size
            tensor = tensor.flatten().narrow(0, 0, numel)

    #print(fn)
    with open(fn, "w") as fh:
        fh.write(f"{get_fingerprint_header()}\n")

        if tensor is not None:
            fh.write(f"{get_fingerprint(tensor)} tensor {tensor.shape}\n")
        else:
            for n, p in model[0].named_parameters():
                fh.write(f"{get_fingerprint(p)} {n} {p.shape}\n")


    return


    # until we figure out how to dump the actual fp32 values don't do this
    fn = f"debug-fp32-{iteration}-pp{pp_rank}-tp{tp_rank}-dp{dp_rank}-{preamble}.txt"
    with open(fn, "w") as fh:
        fh.write(f"{get_fingerprint_header()}\n")
        if tensor is not None:
            tensor = orig_tensor
            if hasattr(tensor, "_hp_param"):
                fh.write(f"{get_fingerprint(tensor._hp_param)} tensor {tensor._hp_param.shape}\n")
                #fh.write(f"{get_fingerprint(tensor._hp_grad)} tensor grad\n")
            else:
                fh.write(f"{get_fingerprint(tensor)} tensor {tensor.shape}\n")
                #fh.write(f"{get_fingerprint(tensor.grad)} tensor grad\n")

        else:
            if hasattr(model[0].module.tied_modules, "embed"):
                p = model[0].module.tied_modules.embed.word_embeddings.weight._hp_param
                fh.write(f"{get_fingerprint(p)} module.tied_modules.embed.word_embeddings.weight._hp_param {p.shape}\n")


def found_kill_switch():
    args = get_args()
    if args.kill_switch_file is not None and os.path.exists(args.kill_switch_file):
        return True
    else:
        return False

