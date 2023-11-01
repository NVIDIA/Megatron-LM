# Copyright (c) 2022-2023, NVIDIA CORPORATION.  All rights reserved.

""" Utilities for operating with dicts and lists. """

from collections import defaultdict
from typing import Any, Callable, Iterable, Optional, Tuple, Union

import torch


def extract_matching_values(
    x: Union[dict, list], predicate: Callable
) -> Tuple[Union[dict, list], Union[dict, list]]:
    """ Return matching and nonmatching values. Keeps hierarchy. """
    if isinstance(x, dict):
        matching_vals = {}
        nonmatching_vals = {}
        for k, v in x.items():
            if isinstance(v, (list, dict)):
                match, nonmatch = extract_matching_values(v, predicate)
                if match:
                    matching_vals[k] = match
                if nonmatch or not v:
                    nonmatching_vals[k] = nonmatch
            elif predicate(v):
                matching_vals[k] = v
            else:
                nonmatching_vals[k] = v
    else:
        assert isinstance(x, list)
        matching_vals = []
        nonmatching_vals = []
        for v in x:
            if isinstance(v, (list, dict)) and v:
                match, nonmatch = extract_matching_values(v, predicate)
                if match:
                    matching_vals.append(match)
                if nonmatch or not v:
                    nonmatching_vals.append(nonmatch)
            elif predicate(v):
                matching_vals.append(v)
            else:
                nonmatching_vals.append(v)
    return matching_vals, nonmatching_vals


def diff(x1: Any, x2: Any, prefix: Tuple = ()) -> Tuple[list, list, list]:
    mismatch = []
    if isinstance(x1, dict) and isinstance(x2, dict):
        only_left = [prefix + (k,) for k in x1.keys() - x2.keys()]
        only_right = [prefix + (k,) for k in x2.keys() - x1.keys()]
        for k in x2.keys() & x1.keys():
            _left, _right, _mismatch = diff(x1[k], x2[k], prefix + (k,))
            only_left.extend(_left)
            only_right.extend(_right)
            mismatch.extend(_mismatch)
    elif isinstance(x1, list) and isinstance(x2, list):
        only_left = list(range(len(x1) - 1, len(x2) - 1, -1))
        only_right = list(range(len(x1) - 1, len(x2) - 1, -1))
        for i, (v1, v2) in enumerate(zip(x1, x2)):
            _left, _right, _mismatch = diff(v1, v2, prefix + (i,))
            only_left.extend(_left)
            only_right.extend(_right)
            mismatch.extend(_mismatch)
    else:
        only_left = []
        only_right = []
        if isinstance(x1, torch.Tensor) and isinstance(x2, torch.Tensor):
            _is_mismatch = not torch.all(x1 == x2)
        else:
            try:
                _is_mismatch = bool(x1 != x2)
            except RuntimeError:
                _is_mismatch = True

        if _is_mismatch:
            mismatch.append((prefix, type(x1), type(x2)))

    return only_left, only_right, mismatch


def inspect_keys_types(d: dict, prefix: Tuple = (), indent: int = 4):
    print_indent = lambda: print(' ' * indent * len(prefix), end='')
    for k, v in d.items():
        if isinstance(v, dict):
            print_indent()
            print(f'> {k}:')
            inspect_keys_types(v, prefix + (k,), indent)
        else:
            print_indent()
            if isinstance(v, torch.Tensor):
                print(f'> {k}: {type(v)} of shape {v.shape}')
            else:
                print(f'> {k}: {type(v)}')


def inspect_types(x: Any, prefix: Tuple = (), indent: int = 4):
    print_indent = lambda: print(' ' * indent * len(prefix), end='')
    if isinstance(x, dict):
        print()
        for k, v in x.items():
            print_indent()
            print(f'> {k}: ', end='')
            inspect_types(v, prefix + (k,), indent)
    elif isinstance(x, list):
        print()
        for i, v in enumerate(x):
            print_indent()
            print(f'- {i}: ', end='')
            inspect_types(v, prefix + (i,), indent)
    else:
        if isinstance(x, torch.Tensor):
            print(f'Tensor of shape {x.shape}')
        else:
            try:
                x_str = str(x)
            except:
                x_str = '<no string repr>'
            if len(x_str) > 30:
                x_str = x_str[:30] + '... (truncated)'
            print(f'[{type(x)}]: {x_str}')


def nested_values(x: Union[dict, list]):
    x_iter = x.values() if isinstance(x, dict) else x
    for v in x_iter:
        if isinstance(v, (dict, list)):
            yield from nested_values(v)
        else:
            yield v


def nested_items_iter(x: Union[dict, list]):
    x_iter = x.items() if isinstance(x, dict) else enumerate(x)
    for k, v in x_iter:
        if isinstance(v, (dict, list)):
            yield from nested_items_iter(v)
        else:
            yield x, k, v


def dict_map(f: Callable, d: dict):
    for sub_d, k, v in nested_items_iter(d):
        sub_d[k] = f(v)


def dict_map_with_key(f: Callable, d: dict):
    for sub_d, k, v in nested_items_iter(d):
        sub_d[k] = f(k, v)


def dict_list_map_inplace(f: Callable, x: Union[dict, list]):
    if isinstance(x, dict):
        for k, v in x.items():
            x[k] = dict_list_map_inplace(f, v)
    elif isinstance(x, list):
        x[:] = (dict_list_map_inplace(f, v) for v in x)
    else:
        return f(x)
    return x


def dict_list_map_outplace(f: Callable, x: Union[dict, list]):
    if isinstance(x, dict):
        return {k: dict_list_map_outplace(f, v) for k, v in x.items()}
    elif isinstance(x, list):
        return [dict_list_map_outplace(f, v) for v in x]
    else:
        return f(x)


def merge(x1: dict, x2: dict):
    if isinstance(x1, dict) and isinstance(x2, dict):
        for k, v2 in x2.items():
            if k not in x1:
                x1[k] = v2
            else:
                x1[k] = merge(x1[k], v2)
    elif isinstance(x1, list) and isinstance(x2, list):
        if len(x1) != len(x2):
            raise ValueError('Cannot merge two lists with different lengths')
        for i, v2 in enumerate(x2):
            x1[i] = merge(x1[i], v2)
    else:
        raise ValueError(f'Duplicate non-dict and non-list values encountered: `{x1}` and `{x2}`')
    return x1


def map_reduce(
    xs: Iterable,
    key_fn: Callable = lambda x: x,
    value_fn: Callable = lambda x: x,
    reduce_fn: Callable = lambda x: x,
) -> dict:
    res = defaultdict(list)
    for x in xs:
        res[key_fn(x)].append(value_fn(x))
    for k in res:
        res[k] = reduce_fn(res[k])
    return dict(res)
