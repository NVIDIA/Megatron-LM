# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

# Note (rwaleffe): This is a temporary file for hybrid mamba-transformer model checkpoint conversion.
# This functionality should be integrated with the megatron core checkpoint loader/saver.


import copy
import os
import re
import shutil
from collections import OrderedDict

import torch
import argparse


tp_split_dim = {
    'word_embeddings.weight': 0,
    'norm.weight': -1,
    'final_norm.weight': -1,
    'output_layer.weight': 0,
    # mamba1/2
    'A_log': 0,
    'D': 0,
    'dt_bias': 0,
    'in_proj.weight': 0,
    'conv1d.weight': 0,
    'conv1d.bias': 0,
    'x_proj.weight': 1,
    'dt_proj.weight': 0,
    'dt_proj.bias': 0,
    'out_proj.weight': 1,
    'mixer.norm.weight': 0,
    # mlp
    'linear_fc1.layer_norm_weight': -1,
    'linear_fc1.weight': 0,
    'linear_fc2.weight': 1,
    # attention
    'self_attention.linear_proj.weight': 1,
    'self_attention.linear_qkv.layer_norm_weight': -1,
    'self_attention.linear_qkv.weight': 0,
}


def get_split_dim(tensor_name):
    # norm.weight will match tensor_name of mixer.norm.weight and norm.weight, need to distinguish
    if 'norm.weight' in tensor_name:
        if 'mixer.norm.weight' in tensor_name:
            return tp_split_dim['mixer.norm.weight']
        else:
            return tp_split_dim['norm.weight']

    for key in tp_split_dim.keys():
        if key in tensor_name:
            return tp_split_dim[key]
    raise Exception("Unknown tensor name {}".format(tensor_name))


def combine_tp_tensors(params, key, dim, tensors):
    tp_size = len(tensors)

    if 'mixer.in_proj.weight' in key and params.mamba_version == 1:
        xs = []; zs = []
        for tensor in tensors:
            x, z = torch.split(tensor, [params.mamba_d_inner//tp_size,
                                        params.mamba_d_inner//tp_size], dim=dim)
            xs.append(x); zs.append(z)
        return torch.cat([torch.cat(xs, dim=dim), torch.cat(zs, dim=dim)], dim=dim)

    elif 'mixer.in_proj.weight' in key and params.mamba_version == 2:
        xs = []; zs = []; Bs = []; Cs = []; dts = []
        for tensor in tensors:
            x, z, B, C, dt = torch.split(tensor, [params.mamba_d_inner // tp_size,
                                                  params.mamba_d_inner // tp_size,
                                                  (params.mamba2_n_groups // tp_size) * args.mamba_d_state,
                                                  (params.mamba2_n_groups // tp_size) * args.mamba_d_state,
                                                  params.mamba2_n_heads // tp_size], dim=dim)
            xs.append(x); zs.append(z); Bs.append(B); Cs.append(C); dts.append(dt)

        for ii in range(len(Bs)):
            Bs[ii] = torch.reshape(Bs[ii], (-1, params.mamba_d_state, Bs[ii].shape[-1]))
            Cs[ii] = torch.reshape(Cs[ii], (-1, params.mamba_d_state, Cs[ii].shape[-1]))
        B = torch.cat(Bs, dim=dim); C = torch.cat(Cs, dim=dim)
        x = torch.cat(xs, dim=dim); z = torch.cat(zs, dim=dim); dt = torch.cat(dts, dim=dim)

        return torch.cat([x, z, B.flatten(0, 1), C.flatten(0, 1), dt], dim=dim)

    elif 'mixer.conv1d' in key and params.mamba_version == 2:
        xs = []; Bs = []; Cs = []
        for tensor in tensors:
            x, B, C = torch.split(tensor, [params.mamba_d_inner//tp_size,
                                           (params.mamba2_n_groups // tp_size) * params.mamba_d_state,
                                           (params.mamba2_n_groups // tp_size) * params.mamba_d_state], dim=dim)
            xs.append(x); Bs.append(B); Cs.append(C)

        for ii in range(len(Bs)):
            if 'weight' in key:
                Bs[ii] = torch.reshape(Bs[ii], (-1, params.mamba_d_state, Bs[ii].shape[-2], Bs[ii].shape[-1]))
                Cs[ii] = torch.reshape(Cs[ii], (-1, params.mamba_d_state, Cs[ii].shape[-2], Cs[ii].shape[-1]))
            elif 'bias' in key:
                Bs[ii] = torch.reshape(Bs[ii], (-1, params.mamba_d_state))
                Cs[ii] = torch.reshape(Cs[ii], (-1, params.mamba_d_state))
            else:
                raise Exception("Unknown key")
        B = torch.cat(Bs, dim=dim); C = torch.cat(Cs, dim=dim)
        x = torch.cat(xs, dim=dim)

        return torch.cat([x, B.flatten(0, 1), C.flatten(0, 1)], dim=dim)

    else:
        return torch.cat(tensors, dim=dim)


def split_tensor_for_tp(params, key, dim, tensor):
    tp_size = params.target_tp_size
    tensor_sliced = []

    if 'mixer.in_proj.weight' in key and params.mamba_version == 1:
        x, z = torch.split(tensor, [params.mamba_d_inner, params.mamba_d_inner], dim=dim)
        x_sliced = torch.chunk(x, tp_size, dim=dim)
        z_sliced = torch.chunk(z, tp_size, dim=dim)
        for (x, z) in zip(x_sliced, z_sliced):
            tensor_sliced.append(torch.cat((x, z), dim=dim))

    elif 'mixer.in_proj.weight' in key and params.mamba_version == 2:
        x, z, B, C, dt = torch.split(tensor, [params.mamba_d_inner, params.mamba_d_inner,
                                                      params.mamba2_n_groups * params.mamba_d_state,
                                                      params.mamba2_n_groups * params.mamba_d_state,
                                                      params.mamba2_n_heads], dim=dim)
        B = torch.reshape(B, (-1, params.mamba_d_state, B.shape[-1]))
        C = torch.reshape(C, (-1, params.mamba_d_state, C.shape[-1]))

        B_sliced = torch.chunk(B, tp_size, dim=dim)
        C_sliced = torch.chunk(C, tp_size, dim=dim)
        x_sliced = torch.chunk(x, tp_size, dim=dim)
        z_sliced = torch.chunk(z, tp_size, dim=dim)
        dt_sliced = torch.chunk(dt, tp_size, dim=dim)

        tensor_sliced = []
        for (x, z, B, C, dt) in zip(x_sliced, z_sliced, B_sliced, C_sliced, dt_sliced):
            tensor_sliced.append(torch.cat((x, z, B.flatten(0, 1), C.flatten(0, 1), dt), dim=dim))

    elif 'mixer.conv1d' in key and params.mamba_version == 2:
        x, B, C = torch.split(tensor, [params.mamba_d_inner,
                                               params.mamba2_n_groups * params.mamba_d_state,
                                               params.mamba2_n_groups * params.mamba_d_state], dim=dim)
        if 'weight' in key:
            B = torch.reshape(B, (-1, params.mamba_d_state, B.shape[-2], B.shape[-1]))
            C = torch.reshape(C, (-1, params.mamba_d_state, C.shape[-2], C.shape[-1]))
        elif 'bias' in key:
            B = torch.reshape(B, (-1, params.mamba_d_state))
            C = torch.reshape(C, (-1, params.mamba_d_state))
        else:
            raise Exception("Unknown key")

        B_sliced = torch.chunk(B, tp_size, dim=dim)
        C_sliced = torch.chunk(C, tp_size, dim=dim)
        x_sliced = torch.chunk(x, tp_size, dim=dim)

        tensor_sliced = []
        for (x, B, C) in zip(x_sliced, B_sliced, C_sliced):
            tensor_sliced.append(torch.cat((x, B.flatten(0, 1), C.flatten(0, 1)), dim=dim))

    else:
        tensor_sliced = torch.chunk(tensor, tp_size, dim=dim)

    return tensor_sliced


def finalize_checkpoint(sample_model, model, params, verbose=False):
    # make sure the rest of the checkpoint is how we want it from the original (i.e., other than the 'model')
    reset_iterations = params.reset_iterations

    # checkpoint 'args'
    model['args'] = copy.deepcopy(sample_model['args'])
    model['args'].tensor_model_parallel_size = params.target_tp_size
    model['args'].pipeline_model_parallel_size = params.target_pp_size
    if reset_iterations:
        model['args'].iteration = 0
        model['args'].consumed_valid_samples = 0
        model['args'].consumed_train_samples = 0
        model['args'].train_iters = 0
        model['args'].train_samples = 0

    # checkpoint 'checkpoint_version'
    model['checkpoint_version'] = copy.deepcopy(sample_model['checkpoint_version'])

    # checkpoint 'iteration'
    model['iteration'] = copy.deepcopy(sample_model['iteration'])
    if reset_iterations:
        model['iteration'] = 0

    # checkpoint 'optimizer'
    # ignore

    # checkpoint 'opt_param_scheduler'
    if 'opt_param_scheduler' in sample_model.keys():
        model['opt_param_scheduler'] = copy.deepcopy(sample_model['opt_param_scheduler'])

    # checkpoint 'rng_state'
    model['rng_state'] = copy.deepcopy(sample_model['rng_state'])

    # report on argument difference
    if verbose:
        original_args = sample_model['args'].__dict__
        final_args = model['args'].__dict__
        for key in original_args:
            if key in final_args:
                if final_args[key] != original_args[key]:
                    print("KEY MISMATCH: {}".format(key))
                    print("\toriginal: {}\n\tfinal: {}".format(original_args[key], final_args[key]))
            else:
                print("KEY MISSING from final: {}, value {}".format(key, original_args[key]))
        print("")
        for key in final_args:
            if key not in original_args:
                print("KEY ADDED to final: {}, value {}".format(key, final_args[key]))

    return model


def main(args):
    print("\n====RUNNING CHECKPOINT CONVERSION====\n")

    args.mamba_d_inner = args.d_model * 2
    args.mamba2_n_heads = args.mamba_d_inner // args.mamba2_head_dim

    # get the latest iteration
    tracker_filename = os.path.join(args.load_dir, 'latest_checkpointed_iteration.txt')
    with open(tracker_filename, 'r') as f:
        metastring = f.read().strip()
        try:
            iteration = int(metastring)
        except ValueError:
            raise Exception("")
    out_iteration = iteration if not args.reset_iterations else 0

    # get model directory and model parallel ranks
    input_model_dir = os.path.join(args.load_dir, 'iter_{:07d}'.format(iteration))
    input_sub_models = os.listdir(input_model_dir)
    # input_sub_models = sorted(input_sub_models, key=lambda x: int(re.search(r'\d+', x).group()))

    # load one of the model parallel ranks to get arguments
    sample_model_file = os.path.join(input_model_dir, input_sub_models[0], "model_optim_rng.pt")
    sample_model = torch.load(sample_model_file)
    print(f"Sample model {sample_model_file} is loaded.\n")

    # input tensor and pipeline parallel size
    input_tp_rank = sample_model['args'].tensor_model_parallel_size
    input_pp_rank = sample_model['args'].pipeline_model_parallel_size
    num_layers_per_pipeline_rank = sample_model['args'].num_layers // input_pp_rank

    # construct full model
    full_model = OrderedDict()
    for pp in range(input_pp_rank):
        print("[INFO] Processing input pipeline rank {}".format(pp))
        tp_models = []
        for tp in range(input_tp_rank):
            dir_name = "mp_rank_{:02d}".format(tp)
            if input_pp_rank > 1:
                dir_name += "_{:03d}".format(pp)
            model_file = os.path.join(input_model_dir, dir_name, "model_optim_rng.pt")

            tp_models.append(torch.load(model_file))
            print(f"Model {model_file} is loaded.")

        if input_tp_rank > 1:
            combined_tp_model = OrderedDict()
            for ii, (key, original_tensor) in enumerate(tp_models[0]['model'].items()):
                if "_extra_state" in key:
                    combined_tp_model[key] = original_tensor
                    continue

                split_dim = get_split_dim(key)
                original_shape = list(original_tensor.shape)
                combined_shape = copy.deepcopy(original_shape)
                combined_shape[split_dim] *= input_tp_rank
                # print("{}, {}, {}".format(ii, key, split_dim))

                if split_dim != -1:
                    # slice together model
                    # print("\tshape mismatch: original {}, combined {}".format(original_shape, combined_shape))
                    combined_tensor = combine_tp_tensors(args, key, split_dim,
                                                    [tp_models[jj]['model'][key].cpu() for jj in range(input_tp_rank)])
                    combined_tp_model[key] = combined_tensor
                else:
                    # copy model
                    combined_tp_model[key] = original_tensor
        else:
            combined_tp_model = tp_models[0]['model']
        # print("Combined tp model: {}".format(combined_tp_model.keys()))

        for ii, (key, original_tensor) in enumerate(combined_tp_model.items()):
            try:
                layer_num = int(re.findall(r'\d+', key)[0])
                new_key = key.replace(str(layer_num), str(layer_num + pp*num_layers_per_pipeline_rank), 1)
            except Exception:
                new_key = key
            full_model[new_key] = original_tensor
    # print("Combined model: {}".format(full_model.keys()))
    print("\n[INFO] Loaded combined model\n")

    # sort by layer
    # full_model_sorted = dict(sorted(people.items(), key=lambda item: item[1]))

    # create new split model
    pp_offset = 0
    num_layers_per_pipeline_rank = sample_model['args'].num_layers // args.target_pp_size

    for pp in range(args.target_pp_size):
        print("[INFO] Processing output pipeline rank {}".format(pp))
        tp_models = []
        for ii in range(args.target_tp_size):
            tp_models.append({'model': OrderedDict()})

        for ii, (key, original_tensor) in enumerate(full_model.items()):
            try:
                layer_num = int(re.findall(r'\d+', key)[0])
                if layer_num >= num_layers_per_pipeline_rank * (pp+1):
                    break
                new_key = key.replace(str(layer_num), str(layer_num - (pp * num_layers_per_pipeline_rank)), 1)
            except Exception:
                new_key = key

            if ii < pp_offset:
                continue
            else:
                pp_offset += 1

            if "_extra_state" in new_key:
                # copy
                for jj in range(args.target_tp_size):
                    tp_models[jj]['model'][new_key] = original_tensor
                continue

            split_dim = get_split_dim(new_key)
            original_shape = list(original_tensor.shape)
            v0 = original_shape[split_dim]
            split_size = v0 // args.target_tp_size
            split_shape = copy.deepcopy(original_shape)
            split_shape[split_dim] = split_size
            # print("{}, {}, {}".format(ii, new_key, split_dim))

            if split_dim != -1:
                # split model
                # print("\tshape mismatch: original {}, combined {}".format(original_shape, split_shape))
                tensor_sliced = split_tensor_for_tp(args, new_key, split_dim, original_tensor)
                for jj in range(args.target_tp_size):
                    tp_models[jj]['model'][new_key] = tensor_sliced[jj]
            else:
                # copy model
                for jj in range(args.target_tp_size):
                    tp_models[jj]['model'][new_key] = original_tensor
        # print(tp_models[0]['model'].keys())

        for tp in range(args.target_tp_size):
            dir_name = "mp_rank_{:02d}".format(tp)
            if args.target_pp_size > 1:
                dir_name += "_{:03d}".format(pp)

            model = finalize_checkpoint(sample_model, tp_models[tp], args, verbose=False)

            save_dir = os.path.join(args.save_dir, 'iter_{:07d}'.format(out_iteration), dir_name)
            os.makedirs(save_dir, exist_ok=True)
            model_file = os.path.join(save_dir, "model_optim_rng.pt")
            torch.save(model, model_file)
            print(f"Model {model_file} is saved.")

    # shutil.copyfile(tracker_filename, os.path.join(args.save_dir, 'latest_checkpointed_iteration.txt'))
    tracker_filename = os.path.join(args.save_dir, 'latest_checkpointed_iteration.txt')
    with open(tracker_filename, 'w') as f:
        f.write(str(out_iteration))


if __name__ == "__main__":
    # example run command:
    # python hybrid_conversion.py
    # --load-dir mamba2-840m-test/checkpoints/
    # --save-dir mamba2-840m-test-conversion/checkpoints/
    # --target-pp-size 1
    # --target-tp-size 1

    parser = argparse.ArgumentParser()
    parser.add_argument('--load-dir', type=str)
    parser.add_argument('--save-dir', type=str)
    parser.add_argument('--target-tp-size', type=int, default=1)
    parser.add_argument('--target-pp-size', type=int, default=1)
    parser.add_argument('--reset-iterations', action='store_true')

    parser.add_argument('--d-model', type=int, default=4096)
    parser.add_argument('--mamba-version', type=int, default=2)
    parser.add_argument('--mamba-d-state', type=int, default=128)
    parser.add_argument('--mamba2-n-groups', type=int, default=8)
    parser.add_argument('--mamba2-head-dim', type=int, default=64)

    args = parser.parse_args()

    main(args)
