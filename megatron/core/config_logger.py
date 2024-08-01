# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

import dataclasses
import json
import os

import torch
import torch.nn as nn

from megatron.core import parallel_state


def get_config_logger_path(config):
    return getattr(config, 'config_logger_dir', '')


def has_config_logger_enabled(config):
    return get_config_logger_path(config) != ''


# For each prefix, holds a counter and increases it every time we dump with this
# prefix.
__config_logger_path_counts = {}


def get_path_count(path):
    """
    keeps tracks of number of times we've seen the input `path` and return count-1
    """
    global __config_logger_path_counts
    if not path in __config_logger_path_counts:
        __config_logger_path_counts[path] = 0
    count = __config_logger_path_counts[path]
    __config_logger_path_counts[path] += 1
    return count


def get_path_with_count(path):
    """
    calls get_path_count and appends returned value to path
    """
    return f'{path}.iter{get_path_count(path)}'


class JSONEncoderWithMcoreTypes(json.JSONEncoder):
    def default(self, o):
        if type(o).__name__ in ['function', 'ProcessGroup']:
            return str(o)
        if type(o).__name__ in ['dict', 'OrderedDict']:
            return {k: self.default(v) for k, v in o.items()}
        if type(o).__name__ in ['list', 'ModuleList']:
            return [self.default(val) for val in o]
        if type(o).__name__ == 'UniqueDescriptor':
            return {
                attr: self.default(getattr(o, attr))
                for attr in filter(lambda x: not x.startswith('__'), dir(o))
            }
        if type(o) is torch.dtype:
            return str(o)
        # if it's a Float16Module, add "Float16Module" to the output dict
        if type(o).__name__ == 'Float16Module':
            return {'Float16Module': {'module': self.default(o.module)}}
        # If it's a nn.Module subchild, either print its children or itself if leaf.
        if issubclass(type(o), nn.Module):
            if len(getattr(o, '_modules', {})) > 0:
                return {key: self.default(val) for key, val in o._modules.items()}
            else:
                return str(o)
        if type(o).__name__ in ['ABCMeta', 'type', 'AttnMaskType']:
            return str(o)
        if dataclasses.is_dataclass(o) or type(o).__name__ in ['ModuleSpec', 'TransformerConfig']:
            return dataclasses.asdict(o)
        try:
            return super().default(o)
        except:
            return str(o)


def log_config_to_disk(config, dict_data, prefix=''):
    """
    Encodes the input dict (dict_data) using the JSONEncoderWithMcoreTypes
    and dumps to disk, as specified via path
    """
    path = get_config_logger_path(config)
    assert path is not None, 'Expected config_logger_dir to be non-empty in config.'

    if 'self' in dict_data:
        if prefix == '':
            prefix = type(dict_data['self']).__name__
        del dict_data['self']

    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)

    rank = parallel_state.get_all_ranks()
    path = get_path_with_count(os.path.join(path, f'{prefix}.rank_{rank}'))
    if type(dict_data).__name__ == 'OrderedDict':
        torch.save(dict_data, f'{path}.pth')
    else:
        with open(f'{path}.json', 'w') as fp:
            json.dump(dict_data, fp, cls=JSONEncoderWithMcoreTypes)


__all__ = ['has_config_logger_enabled', 'log_config_to_disk']
