# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.

"""Input/output checkpointing."""

import os
import random
import sys
import numpy as np

import torch

from megatron import update_num_microbatches
from megatron.core import mpu, tensor_parallel
from .global_vars import get_args
from .utils import (unwrap_model,
                    print_rank_0)


_CHECKPOINT_VERSION = None


def set_checkpoint_version(value):
    global _CHECKPOINT_VERSION
    if _CHECKPOINT_VERSION is not None:
        assert _CHECKPOINT_VERSION == value, \
            "checkpoint versions do not match"
    _CHECKPOINT_VERSION = value


def get_checkpoint_version():
    global _CHECKPOINT_VERSION
    return _CHECKPOINT_VERSION


def check_checkpoint_args(checkpoint_args):
    """Ensure fixed arguments for a model are the same for the input
    arguments and the one retrieved from checkpoint."""
    args = get_args()

    def _compare(arg_name, old_arg_name=None, default=None):
        if old_arg_name is not None:
            ckpt_arg_name = old_arg_name
        else:
            ckpt_arg_name = arg_name
        if default is not None:
            checkpoint_value = getattr(checkpoint_args, ckpt_arg_name, default)
        else:
            checkpoint_value = getattr(checkpoint_args, ckpt_arg_name)
        args_value = getattr(args, arg_name)
        error_message = '{} value from checkpoint ({}) is not equal to the ' \
                        'input argument value ({}).'.format(
                            arg_name, checkpoint_value, args_value)
        assert checkpoint_value == args_value, error_message

    _compare('num_layers')
    _compare('hidden_size')
    _compare('num_attention_heads')
    _compare('add_position_embedding', default=True)
    if args.vocab_file:
        _compare('max_position_embeddings')
        _compare('make_vocab_size_divisible_by')
        _compare('padded_vocab_size')
        _compare('tokenizer_type')
    if args.data_parallel_random_init:
        _compare('data_parallel_random_init')
    if get_checkpoint_version() < 3.0:
        _compare('tensor_model_parallel_size',
                 old_arg_name='model_parallel_size')
    if get_checkpoint_version() >= 3.0:
        _compare('tensor_model_parallel_size')
        _compare('pipeline_model_parallel_size')


def ensure_directory_exists(filename):
    """Build filename's path if it does not already exists."""
    dirname = os.path.dirname(filename)
    os.makedirs(dirname, exist_ok = True)


def get_checkpoint_name(checkpoints_path, iteration, release=False,
                        pipeline_parallel=None,
                        tensor_rank=None, pipeline_rank=None,
                        expert_parallel=None, expert_rank=None):
    """Determine the directory name for this rank's checkpoint."""
    if release:
        directory = 'release'
    else:
        directory = 'iter_{:07d}'.format(iteration)

    # Use both the tensor and pipeline MP rank.
    if pipeline_parallel is None:
        pipeline_parallel = (mpu.get_pipeline_model_parallel_world_size() > 1)
    if tensor_rank is None:
        tensor_rank = mpu.get_tensor_model_parallel_rank()
    if pipeline_rank is None:
        pipeline_rank = mpu.get_pipeline_model_parallel_rank()
    if expert_parallel is None:
        expert_parallel = (mpu.get_expert_model_parallel_world_size() > 1)
    if expert_rank is None:
        expert_rank = mpu.get_expert_model_parallel_rank()

    # Use both the tensor and pipeline MP rank. If using the distributed
    # optimizer, then the optimizer's path must additionally include the
    # data parallel rank.
    if not pipeline_parallel:
        common_path = os.path.join(checkpoints_path, directory,
                            f'mp_rank_{tensor_rank:02d}')
    else:
        common_path = os.path.join(checkpoints_path, directory,
                f'mp_rank_{tensor_rank:02d}_{pipeline_rank:03d}')

    if expert_parallel:
        common_path = common_path + f'_{expert_rank:03d}'

    return os.path.join(common_path, "model_optim_rng.pt")


def get_distributed_optimizer_checkpoint_name(model_checkpoint_name):
    return os.path.join(os.path.dirname(model_checkpoint_name),
                        "distrib_optim.pt")


def find_checkpoint_rank_0(checkpoints_path, iteration, release=False):
    """Finds the checkpoint for rank 0 without knowing if we are using
    pipeline parallelism/expert parallelism or not.

    Since the checkpoint naming scheme changes if pipeline or expert
    parallelism is present, we need to look for both naming schemes if
    we don't know if the checkpoint has pipeline or expert parallelism.
    """

    # Look for checkpoint with no pipelining and no expert parallelism
    filename = get_checkpoint_name(checkpoints_path, iteration, release,
                                   pipeline_parallel=False,
                                   tensor_rank=0, pipeline_rank=0,
                                   expert_parallel=False, expert_rank=0)
    if os.path.isfile(filename):
        return filename

    # Look for checkpoint with no pipelining and expert parallelism
    filename = get_checkpoint_name(checkpoints_path, iteration, release,
                                   pipeline_parallel=False,
                                   tensor_rank=0, pipeline_rank=0,
                                   expert_parallel=True, expert_rank=0)
    if os.path.isfile(filename):
        return filename

    # Look for checkpoint with pipelining and no expert parallelism
    filename = get_checkpoint_name(checkpoints_path, iteration, release,
                                   pipeline_parallel=True,
                                   tensor_rank=0, pipeline_rank=0,
                                   expert_parallel=False, expert_rank=0)
    if os.path.isfile(filename):
        return filename

    # Look for checkpoint with pipelining and expert parallelism
    filename = get_checkpoint_name(checkpoints_path, iteration, release,
                                   pipeline_parallel=True,
                                   tensor_rank=0, pipeline_rank=0,
                                   expert_parallel=True, expert_rank=0)
    if os.path.isfile(filename):
        return filename

    return None, None


def get_checkpoint_tracker_filename(checkpoints_path):

    """Tracker file rescords the latest chckpoint during
    training to restart from."""
    return os.path.join(checkpoints_path, 'latest_checkpointed_iteration.txt')


def read_metadata(tracker_filename):
    # Read the tracker file and either set the iteration or
    # mark it as a release checkpoint.
    iteration = 0
    release = False
    with open(tracker_filename, 'r') as f:
        metastring = f.read().strip()
        try:
            iteration = int(metastring)
        except ValueError:
            release = metastring == 'release'
            if not release:
                print_rank_0('ERROR: Invalid metadata file {}. Exiting'.format(
                    tracker_filename))
                sys.exit()
    assert iteration > 0 or release, 'error parsing metadata file {}'.format(
        tracker_filename)

    # Get the max iteration retrieved across the ranks.
    if torch.distributed.is_initialized():
        iters_cuda = torch.tensor([iteration], dtype=torch.long, device='cuda')
        torch.distributed.all_reduce(iters_cuda, op=torch.distributed.ReduceOp.MAX)
        max_iter = iters_cuda[0].item()

        # We should now have all the same iteration.
        # If not, print a warning and chose the maximum
        # iteration across all ranks.
        if iteration != max_iter:
            rank = torch.distributed.get_rank()
            print('WARNING: on rank {} found iteration {} in the '
                  'metadata while max iteration across the ranks '
                  'is {}, replacing it with max iteration.'.format(
                      rank, iteration, max_iter), flush=True)
    else:
        # When loading a checkpoint outside of training (for example,
        # when editing it), we might not have torch distributed
        # initialized, in this case, just assume we have the latest
        max_iter = iteration
    return max_iter, release


def get_rng_state():
    """ collect rng state across data parallel ranks """
    args = get_args()
    rng_state = {
        'random_rng_state': random.getstate(),
        'np_rng_state': np.random.get_state(),
        'torch_rng_state': torch.get_rng_state(),
        'cuda_rng_state': torch.cuda.get_rng_state(),
        'rng_tracker_states': tensor_parallel.get_cuda_rng_tracker().get_states()}

    rng_state_list = None
    if torch.distributed.is_initialized() and \
            mpu.get_data_parallel_world_size() > 1 and \
            args.data_parallel_random_init:
        rng_state_list = \
            [None for i in range(mpu.get_data_parallel_world_size())]
        torch.distributed.all_gather_object(
            rng_state_list,
            rng_state,
            group=mpu.get_data_parallel_group())
    else:
        rng_state_list = [rng_state]

    return rng_state_list


def save_checkpoint(iteration, model, optimizer, opt_param_scheduler):
    """Save a model checkpoint."""
    args = get_args()

    # Only rank zero of the data parallel writes to the disk.
    model = unwrap_model(model)

    print_rank_0('saving checkpoint at iteration {:7d} to {}'.format(
        iteration, args.save))

    # Collect rng state across data parallel ranks.
    rng_state = get_rng_state()

    # Checkpoint name.
    checkpoint_name = get_checkpoint_name(args.save, iteration)

    # Save distributed optimizer's custom parameter state.
    if args.use_distributed_optimizer and not args.no_save_optim and optimizer is not None:
        optim_checkpoint_name = \
            get_distributed_optimizer_checkpoint_name(checkpoint_name)
        ensure_directory_exists(optim_checkpoint_name)
        optimizer.save_parameter_state(optim_checkpoint_name)

    # Collect args, model, RNG.
    if not torch.distributed.is_initialized() \
            or mpu.get_data_modulo_expert_parallel_rank() == 0:

        # Arguments, iteration, and model.
        state_dict = {}
        state_dict['args'] = args
        state_dict['checkpoint_version'] = 3.0
        state_dict['iteration'] = iteration
        if len(model) == 1:
            state_dict['model'] = model[0].state_dict_for_save_checkpoint()
        else:
            for i in range(len(model)):
                mpu.set_virtual_pipeline_model_parallel_rank(i)
                state_dict['model%d' % i] = \
                    model[i].state_dict_for_save_checkpoint()

        # Optimizer stuff.
        if not args.no_save_optim:
            if optimizer is not None:
                state_dict['optimizer'] = optimizer.state_dict()
            if opt_param_scheduler is not None:
                state_dict['opt_param_scheduler'] = \
                    opt_param_scheduler.state_dict()

        # RNG states.
        if not args.no_save_rng:
            state_dict["rng_state"] = rng_state

        # Save.
        ensure_directory_exists(checkpoint_name)
        torch.save(state_dict, checkpoint_name)

    # Wait so everyone is done (necessary)
    if torch.distributed.is_initialized():
        torch.distributed.barrier()

    print_rank_0('  successfully saved checkpoint at iteration {:7d} to {}' \
                 .format(iteration, args.save))

    # And update the latest iteration
    if not torch.distributed.is_initialized() \
       or torch.distributed.get_rank() == 0:
        tracker_filename = get_checkpoint_tracker_filename(args.save)
        with open(tracker_filename, 'w') as f:
            f.write(str(iteration))

    # Wait so everyone is done (not necessary)
    if torch.distributed.is_initialized():
        torch.distributed.barrier()


def _transpose_first_dim(t, num_splits, num_splits_first, model):
    input_shape = t.size()
    # We use a self_attention module but the values extracted aren't
    # specific to self attention so should work for cross attention as well
    while hasattr(model, 'module'):
        model = model.module
    attention_module = model.language_model.encoder.layers[0].self_attention
    hidden_size_per_attention_head = attention_module.hidden_size_per_attention_head
    num_attention_heads_per_partition = attention_module.num_attention_heads_per_partition
    if num_splits_first:
        """[num_splits * np * hn, h]
        -->(view) [num_splits, np, hn, h]
        -->(tranpose) [np, num_splits, hn, h]
        -->(view) [np * num_splits * hn, h] """

        intermediate_shape = \
            (num_splits, num_attention_heads_per_partition,
             hidden_size_per_attention_head) + input_shape[1:]

        t = t.view(*intermediate_shape)
        t = t.transpose(0, 1).contiguous()
    else:
        """[np * hn * num_splits, h]
        -->(view) [np, hn, num_splits, h]
        -->(tranpose) [np, num_splits, hn, h]
        -->(view) [np * num_splits * hn, h] """

        intermediate_shape = \
            (num_attention_heads_per_partition,
             hidden_size_per_attention_head, num_splits) +\
             input_shape[1:]

        t = t.view(*intermediate_shape)
        t = t.transpose(1, 2).contiguous()
    t = t.view(*input_shape)

    return t


def fix_query_key_value_ordering(model, checkpoint_version):
    """Fix up query/key/value matrix ordering if checkpoint
    version is smaller than 2.0
    """
    if checkpoint_version < 2.0:
        if isinstance(model, list):
            assert len(model)==1
            model = model[0]
        for name, param in model.named_parameters():
            if name.endswith(('.query_key_value.weight', '.query_key_value.bias')):
                if checkpoint_version == 0:
                    fixed_param = _transpose_first_dim(param.data, 3, True, model)
                elif checkpoint_version == 1.0:
                    fixed_param = _transpose_first_dim(param.data, 3, False, model)
                else:
                    print_rank_0(f"Invalid checkpoint version {checkpoint_version}.")
                    sys.exit()
                param.data.copy_(fixed_param)
            if name.endswith(('.key_value.weight', '.key_value.bias')):
                if checkpoint_version == 0:
                    fixed_param = _transpose_first_dim(param.data, 2, True, model)
                elif checkpoint_version == 1.0:
                    fixed_param = _transpose_first_dim(param.data, 2, False, model)
                else:
                    print_rank_0(f"Invalid checkpoint version {checkpoint_version}.")
                    sys.exit()
                param.data.copy_(fixed_param)
        print_rank_0(" succesfully fixed query-key-values ordering for"
                     " checkpoint version {}".format(checkpoint_version))


def _load_base_checkpoint(load_dir, rank0=False):
    """ Load the base state_dict from the given directory

    If rank0 is true, just loads rank 0 checkpoint, ignoring arguments.
    """

    # Read the tracker file and set the iteration.
    tracker_filename = get_checkpoint_tracker_filename(load_dir)

    # If no tracker file, return nothing
    if not os.path.isfile(tracker_filename):
        if not rank0:
            print_rank_0('WARNING: could not find the metadata file {} '.format(
                tracker_filename))
            print_rank_0('    will not load any checkpoints and will start from '
                         'random')
        return None, "", False

    # Otherwise, read the tracker file and either set the iteration or
    # mark it as a release checkpoint.
    iteration, release = read_metadata(tracker_filename)

    # Checkpoint.
    if rank0:
        checkpoint_name = find_checkpoint_rank_0(load_dir, iteration, release)
    else:
        checkpoint_name = get_checkpoint_name(load_dir, iteration, release)
        if release:
            print_rank_0(f' loading release checkpoint from {load_dir}')
        else:
            print_rank_0(f' loading checkpoint from {load_dir} at iteration {iteration}')

    # Load the checkpoint.
    try:
        state_dict = torch.load(checkpoint_name, map_location='cpu')
    except ModuleNotFoundError:
        from megatron.fp16_deprecated import loss_scaler
        # For backward compatibility.
        if not rank0:
            print_rank_0(' > deserializing using the old code structure ...')
        sys.modules['fp16.loss_scaler'] = sys.modules[
            'megatron.fp16_deprecated.loss_scaler']
        sys.modules['megatron.fp16.loss_scaler'] = sys.modules[
            'megatron.fp16_deprecated.loss_scaler']
        state_dict = torch.load(checkpoint_name, map_location='cpu')
        sys.modules.pop('fp16.loss_scaler', None)
        sys.modules.pop('megatron.fp16.loss_scaler', None)
    except BaseException as e:
        print_rank_0('could not load the checkpoint')
        print_rank_0(e)
        sys.exit()

    return state_dict, checkpoint_name, release


def load_args_from_checkpoint(args, load_arg='load'):
    """Set required arguments from the checkpoint specified in the
    arguments.

    Will overwrite arguments that have a non-None default value, but
    will leave any arguments that default to None as set.

    Returns the same args NameSpace with the new values added/updated.

    If no checkpoint is specified in args, or if the checkpoint is
    there but invalid, the arguments will not be modified

    """
    load_dir = getattr(args, load_arg)

    if load_dir is None:
        print_rank_0('No load directory specified, using provided arguments.')
        return args

    state_dict, checkpoint_name, release = _load_base_checkpoint(load_dir, rank0=True)

    # Args.
    if not state_dict:
        print_rank_0('Checkpoint not found to provide arguments, using provided arguments.')
        return args

    if 'args' not in state_dict:
        print_rank_0('Checkpoint provided does not have arguments saved, using provided arguments.')
        return args

    checkpoint_args = state_dict['args']
    checkpoint_version = state_dict.get('checkpoint_version', 0)
    args.iteration = state_dict['iteration']

    # One-off conversion for foundation models
    if hasattr(checkpoint_args, 'disable_bias_linear'):
        setattr(checkpoint_args, 'add_bias_linear', not getattr(checkpoint_args, 'disable_bias_linear'))

    def _set_arg(arg_name, old_arg_name=None, force=False):
        if not force and getattr(args, arg_name, None) is not None:
            return

        if old_arg_name is not None:
            checkpoint_value = getattr(checkpoint_args, old_arg_name, None)
        else:
            checkpoint_value = getattr(checkpoint_args, arg_name, None)

        if checkpoint_value is not None:
            print_rank_0(f"Setting {arg_name} to {checkpoint_value} from checkpoint")
            setattr(args, arg_name, checkpoint_value)
        else:
            print_rank_0(f"Checkpoint did not provide arguments {arg_name}")

    _set_arg('num_layers')
    _set_arg('hidden_size')
    _set_arg('ffn_hidden_size')
    _set_arg('seq_length')
    _set_arg('num_attention_heads')
    _set_arg('num_query_groups', force=True)
    _set_arg('group_query_attention', force=True)
    _set_arg('kv_channels')
    _set_arg('max_position_embeddings')
    _set_arg('position_embedding_type', force=True)
    _set_arg('add_position_embedding', force=True)
    _set_arg('use_rotary_position_embeddings', force=True)
    _set_arg('rotary_percent', force=True)
    _set_arg('add_bias_linear', force=True)
    _set_arg('swiglu', force=True)
    _set_arg('untie_embeddings_and_output_weights', force=True)
    _set_arg('apply_layernorm_1p', force=True)
    _set_arg('normalization', force=True)
    _set_arg('tokenizer_type')
    _set_arg('padded_vocab_size')
    if checkpoint_version < 3.0:
        _set_arg('tensor_model_parallel_size',
                 'model_parallel_size')
    else:
        _set_arg('tensor_model_parallel_size', force=True)
        _set_arg('pipeline_model_parallel_size', force=True)
        _set_arg('virtual_pipeline_model_parallel_size', force=True)
        _set_arg('num_layers_per_virtual_pipeline_stage')
    return args, checkpoint_args


def load_checkpoint(model, optimizer, opt_param_scheduler, load_arg='load', strict=True):
    """Load a model checkpoint and return the iteration.
    strict (bool): whether to strictly enforce that the keys in
        :attr:`state_dict` of the checkpoint match the names of
        parameters and buffers in model.
    """
    args = get_args()
    load_dir = getattr(args, load_arg)

    model = unwrap_model(model)

    state_dict, checkpoint_name, release = _load_base_checkpoint(load_dir, rank0=False)

    # Checkpoint not loaded.
    if state_dict is None:

        # Conditionally exit at this point.
        if args.exit_on_missing_checkpoint:
            print_rank_0(">> '--exit-on-missing-checkpoint' set ... exiting. <<")
            torch.distributed.barrier()
            sys.exit()

        # Iteration defaults to 0.
        return 0

    # Set checkpoint version.
    set_checkpoint_version(state_dict.get('checkpoint_version', 0))

    # Set iteration.
    if args.finetune or release:
        iteration = 0
    else:
        try:
            iteration = state_dict['iteration']
        except KeyError:
            try:  # Backward compatible with older checkpoints
                iteration = state_dict['total_iters']
            except KeyError:
                print_rank_0('A metadata file exists but unable to load '
                             'iteration from checkpoint {}, exiting'.format(
                                 checkpoint_name))
                sys.exit()

    # Check arguments.
    assert args.consumed_train_samples == 0
    assert args.consumed_valid_samples == 0
    if 'args' in state_dict and not args.finetune:
        checkpoint_args = state_dict['args']
        check_checkpoint_args(checkpoint_args)
        args.consumed_train_samples = getattr(checkpoint_args,
                                              'consumed_train_samples', 0)
        update_num_microbatches(consumed_samples=args.consumed_train_samples)
        args.consumed_valid_samples = getattr(checkpoint_args,
                                              'consumed_valid_samples', 0)
    else:
        print_rank_0('could not find arguments in the checkpoint ...')

    # Model.
    strict = False if args.retro_add_retriever else strict
    if len(model) == 1:
        model[0].load_state_dict(state_dict['model'], strict=strict)
    else:
        for i in range(len(model)):
            mpu.set_virtual_pipeline_model_parallel_rank(i)
            model[i].load_state_dict(state_dict['model%d' % i], strict=strict)

    # Fix up query/key/value matrix ordering if needed.
    checkpoint_version = get_checkpoint_version()
    print_rank_0(f' checkpoint version {checkpoint_version}')
    fix_query_key_value_ordering(model, checkpoint_version)

    # Optimizer.
    if not release and not args.finetune and not args.no_load_optim:
        try:
            # Load state dict.
            if optimizer is not None:
                optimizer.load_state_dict(state_dict['optimizer'])

            # Load distributed optimizer's custom parameter state.
            if args.use_distributed_optimizer:
                tracker_filename = get_checkpoint_tracker_filename(load_dir)
                iteration, release = read_metadata(tracker_filename)
                model_checkpoint_name = \
                    get_checkpoint_name(load_dir, iteration, release)
                optim_checkpoint_name = \
                    get_distributed_optimizer_checkpoint_name(
                        model_checkpoint_name)
                optimizer.load_parameter_state(optim_checkpoint_name)

            # Load scheduler.
            if opt_param_scheduler is not None:
                if 'lr_scheduler' in state_dict: # backward compatbility
                    opt_param_scheduler.load_state_dict(state_dict['lr_scheduler'])
                else:
                    opt_param_scheduler.load_state_dict(state_dict['opt_param_scheduler'])
        except KeyError:
            print_rank_0('Unable to load optimizer from checkpoint {}. '
                         'Specify --no-load-optim or --finetune to prevent '
                         'attempting to load the optimizer state, '
                         'exiting ...'.format(checkpoint_name))
            sys.exit()
    else:
        if (args.fp16 or args.bf16) and optimizer is not None:
            optimizer.reload_model_params()

    # rng states.
    if not release and not args.finetune and not args.no_load_rng:
        try:
            if 'rng_state' in state_dict:
                # access rng_state for data parallel rank
                if args.data_parallel_random_init:
                    rng_state = state_dict['rng_state'][mpu.get_data_parallel_rank()]
                else:
                    rng_state = state_dict['rng_state'][0]
                random.setstate(rng_state['random_rng_state'])
                np.random.set_state(rng_state['np_rng_state'])
                torch.set_rng_state(rng_state['torch_rng_state'])
                torch.cuda.set_rng_state(rng_state['cuda_rng_state'])
                # Check for empty states array
                if not rng_state['rng_tracker_states']:
                    raise KeyError
                tensor_parallel.get_cuda_rng_tracker().set_states(
                    rng_state['rng_tracker_states'])
            else:  # backward compatability
                random.setstate(state_dict['random_rng_state'])
                np.random.set_state(state_dict['np_rng_state'])
                torch.set_rng_state(state_dict['torch_rng_state'])
                torch.cuda.set_rng_state(state_dict['cuda_rng_state'])
                # Check for empty states array
                if not state_dict['rng_tracker_states']:
                    raise KeyError
                tensor_parallel.get_cuda_rng_tracker().set_states(
                    state_dict['rng_tracker_states'])
        except KeyError:
            print_rank_0('Unable to load rng state from checkpoint {}. '
                         'Specify --no-load-rng or --finetune to prevent '
                         'attempting to load the rng state, '
                         'exiting ...'.format(checkpoint_name))
            sys.exit()

    # Some utilities want to load a checkpoint without distributed being initialized
    if torch.distributed.is_initialized():
        torch.distributed.barrier()

    print_rank_0(f'  successfully loaded checkpoint from {args.load} '
                 f'at iteration {iteration}')

    return iteration


def load_biencoder_checkpoint(model, only_query_model=False,
                              only_context_model=False, custom_load_path=None):
    """
    selectively load retrieval models for indexing/retrieving
    from saved checkpoints
    """

    args = get_args()

    model = unwrap_model(model)

    load_path = custom_load_path if custom_load_path is not None else args.load

    tracker_filename = get_checkpoint_tracker_filename(load_path)
    with open(tracker_filename, 'r') as f:
        iteration = int(f.read().strip())

    checkpoint_name = get_checkpoint_name(load_path, iteration,
                                          args.use_distributed_optimizer,
                                          release=False)

    if mpu.get_data_parallel_rank() == 0:
        print('global rank {} is loading checkpoint {}'.format(
            torch.distributed.get_rank(), checkpoint_name))

    state_dict = torch.load(checkpoint_name, map_location='cpu')
    ret_state_dict = state_dict['model']

    if only_query_model:
        ret_state_dict.pop('context_model')
    if only_context_model:
        ret_state_dict.pop('query_model')

    assert len(model) == 1
    model[0].load_state_dict(ret_state_dict)
    torch.distributed.barrier()

    if mpu.get_data_parallel_rank() == 0:
        print(' successfully loaded {}'.format(checkpoint_name))

    return model
