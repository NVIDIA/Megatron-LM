#!/usr/bin/env python

import argparse
import os
import torch
from collections import OrderedDict
from deepspeed_checkpoint import ARGS_KEY, DeepSpeedCheckpoint

MODEL_KEY = 'model'
ARGS_KEY = 'args'
LANGUGAGE_MODEL_KEY = 'language_model'
EMBEDDING_KEY = 'embedding'
ENCODER_KEY = 'encoder'
WORD_EMBEDDINGS_FOR_HEAD_KEY = 'word_embeddings_for_head'
WORD_EMBEDDINGS_KEY = 'word_embeddings'
FINAL_LAYER_NORM_KEY ='final_layernorm'
CHECKPOINT_VERSION_KEY = 'checkpoint_version'
CHECKPOINT_VERSION_VALUE = 3.0
ITERATION_KEY = 'iteration'

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_folder', default=None, type=str, help='Input DeepSpeed Checkpoint folder')
    parser.add_argument('--output_folder', default=None, type=str, help='Output Megatron checkpoint folder')
    parser.add_argument('--target_tp', default=1, type=int, help='Target TP degree')
    parser.add_argument('--target_pp', default=1, type=int, help='Target PP degree')
    parser.add_argument('--for_release', action='store_true', help='Convert for release purpose, reset some (progress) counters.')
    args = parser.parse_args()
    print(f'args = {args}')
    return args


def _convert_ds_transformer_state(sd_list):
    new_sd = OrderedDict()
    for i, sd in enumerate(sd_list):
        for key, value in sd.items():
            new_key = f'layers.{i}.{key}'
            new_sd[new_key] = value

    return new_sd

def _create_checkpoint_paths(base_folder, iteration, tp_degree, pp_degree):
    path_list = []
    iter_folder = f'iter_{iteration:07d}'
    for i in range(0, tp_degree):
        path_list.append([])
        for j in range(0, pp_degree):
            rank_folder = f'mp_rank_{i:02d}' if pp_degree == 1 else f'mp_rank_{i:02d}_{j:03d}'
            ckpt_path = os.path.join(rank_folder, 'model_optim_rng.pt')
            path_list[i].append(os.path.join(base_folder, iter_folder, ckpt_path))

    return path_list


def _create_megatron_dict():
    language_model_dict = {
        EMBEDDING_KEY: {},
        ENCODER_KEY: {}
    }
    megatron_dict = {
        MODEL_KEY: {LANGUGAGE_MODEL_KEY: language_model_dict},
        CHECKPOINT_VERSION_KEY: CHECKPOINT_VERSION_VALUE
    }
    return megatron_dict


def _save_checkpoint(file_path, chkpt_sd):
    dir, _ = os.path.split(file_path)
    os.makedirs(dir, exist_ok=True)
    torch.save(chkpt_sd, file_path)


def _renest_sd(sd):
    new_sd = OrderedDict()
    for key, value in sd.items():
        a, b = key.split('.')
        new_sd[a] = {b: value}
    return new_sd


def _create_rank_checkpoint(ds_checkpoint, tp_index, pp_index, for_release=False):
    meg_encoder_sd = OrderedDict()
    meg_embedding_sd = OrderedDict()
    meg_embedding_for_head_sd = OrderedDict()

    transformer_sd = ds_checkpoint.get_transformer_state(tp_index, pp_index)
    meg_encoder_sd.update(_convert_ds_transformer_state(transformer_sd))

    if pp_index in [0, ds_checkpoint.pp_degree - 1]:
        embedding_sd = ds_checkpoint.get_embedding_state(tp_index)
        nested_embedding_sd = _renest_sd(embedding_sd)
        if pp_index == 0:
            meg_embedding_sd.update(nested_embedding_sd)

        if pp_index == ds_checkpoint.pp_degree - 1:
            for key, value in embedding_sd.items():
                if key.startswith(WORD_EMBEDDINGS_KEY):
                    fields = key.split('.')
                    new_fields = fields[1:]
                    new_key = '.'.join(new_fields)
                    meg_embedding_for_head_sd[new_key] = value

            final_norm_sd = ds_checkpoint.get_final_norm_state(tp_index)
            new_final_norm_sd = {f'{FINAL_LAYER_NORM_KEY}.{key}': value for key, value in final_norm_sd.items()}
            meg_encoder_sd.update(new_final_norm_sd)

    checkpoint_sd = _create_megatron_dict()

    iteration = ds_checkpoint.get_iteration()
    checkpoint_sd[ITERATION_KEY] = iteration
    if pp_index == 0:
        checkpoint_sd[MODEL_KEY][LANGUGAGE_MODEL_KEY][EMBEDDING_KEY] = meg_embedding_sd
    checkpoint_sd[MODEL_KEY][LANGUGAGE_MODEL_KEY][ENCODER_KEY] = meg_encoder_sd
    if pp_index == ds_checkpoint.pp_degree - 1:
        checkpoint_sd[MODEL_KEY][WORD_EMBEDDINGS_FOR_HEAD_KEY] = meg_embedding_for_head_sd

    checkpoint_sd[ARGS_KEY] = ds_checkpoint.get_args()
    # Adjust specific fields
    checkpoint_sd[ARGS_KEY].tensor_model_parallel_size = ds_checkpoint.tp_degree
    checkpoint_sd[ARGS_KEY].pipeline_model_parallel_size = ds_checkpoint.pp_degree
    if for_release:
        checkpoint_sd[ARGS_KEY].consumed_train_samples = 0
        checkpoint_sd[ARGS_KEY].consumed_valid_samples = 0

    return checkpoint_sd


def _create_latest_file(base_folder, iteration):
    file_path = os.path.join(base_folder, 'latest_checkpointed_iteration.txt')
    os.makedirs(base_folder, exist_ok=True)
    with open(file_path, 'w') as f:
        f.write(str(iteration))

def main():
    print(f'Convert DeepSpeed Checkpoint to Megatron Checkpoint')

    args = parse_arguments()
    print(f'Converting DeepSpeed checkpoint in {args.input_folder} to Megatron checkpoint in {args.output_folder}')

    ds_checkpoint = DeepSpeedCheckpoint(args.input_folder, args.target_tp, args.target_pp)
    iteration = ds_checkpoint.get_iteration()
    _create_latest_file(args.output_folder, iteration)
    checkpoint_paths = _create_checkpoint_paths(args.output_folder, iteration, ds_checkpoint.tp_degree, ds_checkpoint.pp_degree)
    for i in range(0, ds_checkpoint.tp_degree):
        for j in range(0, ds_checkpoint.pp_degree):
            sd = _create_rank_checkpoint(ds_checkpoint, i, j, args.for_release)
            _save_checkpoint(checkpoint_paths[i][j], sd)

if __name__ == "__main__":
    main()
