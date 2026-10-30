# Copyright (c) 2025, NVIDIA CORPORATION.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import dataclasses
import json
import os

import torch
import torch.nn as nn

from megatron.core import parallel_state
from megatron.core.process_groups_config import warn_global_process_group_fallback

# The groups whose ranks name a dump, in the order of parallel_state.get_all_ranks(): tensor,
# GTP-remat, data (with GTP-remat peers), context, pipeline and expert parallel.
_RANK_STRING_GROUPS = ('tp', 'gtp_remat', 'dp_gtp_remat', 'cp', 'pp', 'ep')


def get_config_logger_path(config):
    """Get the path to the config logger directory."""
    return getattr(config, 'config_logger_dir', '')


def has_config_logger_enabled(config):
    """Check if config logger is enabled."""
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
    """
    Custom JSON encoder that serializes according to types in mcore.
    """

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


def _rank_string(groups):
    """Join this rank's ranks in ``groups``; a None group, for an axis that is off, counts as 0."""
    return '_'.join('0' if group is None else str(group.rank()) for group in groups)


def log_config_to_disk(config, dict_data, prefix='', rank_str='', *, pg_collection=None):
    """
    Encodes the input dict (dict_data) using the JSONEncoderWithMcoreTypes
    and dumps to disk, as specified via path

    The file name ends in ``rank_str``. By default it is this rank's
    tp_gtp_dp_cp_pp_ep ranks in the groups of ``pg_collection``, the process
    groups of the model that logs. When ``pg_collection`` is omitted, or does
    not set all of these groups, the ranks come from the global parallel state;
    this fallback is deprecated and emits a ``ProcessGroupFallbackWarning``.
    """
    path = get_config_logger_path(config)
    assert path is not None, 'Expected config_logger_dir to be non-empty in config.'

    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)

    if 'self' in dict_data:
        if prefix == '':
            prefix = type(dict_data['self']).__name__
        del dict_data['self']

    # the caller of the funcion can decide the most informative string
    # rank_str defaults to '0_0_0_0_0_0' format (tp_gtp_dp_cp_pp_ep ranks)
    if rank_str == '':
        groups = {} if pg_collection is None else vars(pg_collection)
        if all(name in groups for name in _RANK_STRING_GROUPS):
            rank_str = _rank_string(groups[name] for name in _RANK_STRING_GROUPS)
        else:
            # A partial collection could give two processes the same file name, so it falls back
            # to the global ranks too.
            warn_global_process_group_fallback(
                'log_config_to_disk', deprecated_in='0.21', removed_in='0.23'
            )
            rank_str = parallel_state.get_all_ranks()

    path = get_path_with_count(os.path.join(path, f'{prefix}.rank_{rank_str}'))
    if type(dict_data).__name__ == 'OrderedDict':
        torch.save(dict_data, f'{path}.pth')
    else:
        with open(f'{path}.json', 'w') as fp:
            json.dump(dict_data, fp, cls=JSONEncoderWithMcoreTypes)


__all__ = ['has_config_logger_enabled', 'log_config_to_disk']
