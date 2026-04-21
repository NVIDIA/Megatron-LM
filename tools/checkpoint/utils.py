# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
import psutil
import torch


def chunk_bias(bias, parallel_mode, tp_size=1, ep_size=1):
    assert parallel_mode in ["row", "column"]
    if bias.dim() == 2:
        num_experts, hidden_size = bias.shape
        if parallel_mode == 'column':
            bias = bias.reshape(ep_size, num_experts // ep_size, tp_size, hidden_size // tp_size)
            bias = bias.permute(0, 2, 1, 3) # (ep_size, tp_size, local_eps, hidden_size)
        else:
            bias = bias.reshape(ep_size, num_experts // ep_size, hidden_size) # (ep_size, local_eps, hidden_size)
        return bias
    else:
        hidden_size = bias.shape
        if parallel_mode == "column":
            bias = bias.reshape(tp_size, hidden_size[0] // tp_size) # (tp_size, hidden_size)
        return bias


def chunk_weight(weight, parallel_mode, tp_size=1, ep_size=1):
    assert parallel_mode in ["row", "column"]
    if weight.dim() == 3:
        num_experts, out_features, in_features = weight.shape
        if parallel_mode == "column":
            weight = weight.reshape(ep_size, num_experts // ep_size, tp_size, out_features // tp_size, in_features)
            weight = weight.permute(0, 2, 1, 3, 4)
        else:
            weight = weight.reshape(ep_size, num_experts // ep_size, out_features, tp_size, in_features // tp_size)
            weight = weight.permute(0, 3, 1, 2, 4)
        return weight # (ep_size, tp_size, local_eps, output_features, in_features)
    else:
        out_features, in_features = weight.shape
        if parallel_mode == "column":
            weight = weight.reshape(tp_size, out_features // tp_size, in_features)
        else:
            weight = weight.reshape(out_features, tp_size, in_features // tp_size).permute(1, 0, 2)
        return weight # (tp_size, output_features, in_features)


def combine_in_proj(tensors, d_inner, ngroups, d_state, nheads, tp_size=1):
    xs = []; zs = []; Bs = []; Cs = []; dts = []
    for tensor in tensors:
        x, z, B, C, dt = torch.split(tensor, [d_inner // tp_size,
                                              d_inner // tp_size,
                                              (ngroups // tp_size) * d_state,
                                              (ngroups // tp_size) * d_state,
                                              nheads // tp_size], dim=0)
        xs.append(x); zs.append(z); Bs.append(B); Cs.append(C); dts.append(dt)

    for ii in range(len(Bs)):
        Bs[ii] = torch.reshape(Bs[ii], (-1, d_state, Bs[ii].shape[-1]))
        Cs[ii] = torch.reshape(Cs[ii], (-1, d_state, Cs[ii].shape[-1]))
    B = torch.cat(Bs, dim=0); C = torch.cat(Cs, dim=0)
    x = torch.cat(xs, dim=0); z = torch.cat(zs, dim=0); dt = torch.cat(dts, dim=0)

    return torch.cat([x, z, B.flatten(0, 1), C.flatten(0, 1), dt], dim=0)


def combine_conv1d(tensors, key, d_inner, ngroups, d_state, tp_size=1):
    xs = []; Bs = []; Cs = []
    for tensor in tensors:
        x, B, C = torch.split(tensor, [d_inner//tp_size,
                                       (ngroups // tp_size) * d_state,
                                       (ngroups // tp_size) * d_state], dim=0)
        xs.append(x); Bs.append(B); Cs.append(C)

    for ii in range(len(Bs)):
        if 'weight' in key:
            Bs[ii] = torch.reshape(Bs[ii], (-1, d_state, Bs[ii].shape[-2], Bs[ii].shape[-1]))
            Cs[ii] = torch.reshape(Cs[ii], (-1, d_state, Cs[ii].shape[-2], Cs[ii].shape[-1]))
        elif 'bias' in key:
            Bs[ii] = torch.reshape(Bs[ii], (-1, d_state))
            Cs[ii] = torch.reshape(Cs[ii], (-1, d_state))
        else:
            raise Exception("Unknown key")
    B = torch.cat(Bs, dim=0); C = torch.cat(Cs, dim=0)
    x = torch.cat(xs, dim=0)

    return torch.cat([x, B.flatten(0, 1), C.flatten(0, 1)], dim=0)


def split_in_proj(tensor, d_inner, ngroups, d_state, nheads, tp_size):
    x, z, B, C, dt = torch.split(tensor, [d_inner, d_inner,
                                                  ngroups * d_state,
                                                  ngroups * d_state,
                                                  nheads], dim=0)
    B = torch.reshape(B, (-1, d_state, B.shape[-1]))
    C = torch.reshape(C, (-1, d_state, C.shape[-1]))

    B_sliced = torch.chunk(B, tp_size, dim=0)
    C_sliced = torch.chunk(C, tp_size, dim=0)
    x_sliced = torch.chunk(x, tp_size, dim=0)
    z_sliced = torch.chunk(z, tp_size, dim=0)
    dt_sliced = torch.chunk(dt, tp_size, dim=0)

    tensor_sliced = []
    for (x, z, B, C, dt) in zip(x_sliced, z_sliced, B_sliced, C_sliced, dt_sliced):
        tensor_sliced.append(torch.cat((x, z, B.flatten(0, 1), C.flatten(0, 1), dt), dim=0))

    return tensor_sliced


def split_conv1d(tensor, key, d_inner, ngroups, d_state, tp_size):
    x, B, C = torch.split(tensor, [d_inner,
                                           ngroups * d_state,
                                           ngroups * d_state], dim=0)
    if 'weight' in key:
        B = torch.reshape(B, (-1, d_state, B.shape[-2], B.shape[-1]))
        C = torch.reshape(C, (-1, d_state, C.shape[-2], C.shape[-1]))
    elif 'bias' in key:
        B = torch.reshape(B, (-1, d_state))
        C = torch.reshape(C, (-1, d_state))
    else:
        raise Exception("Unknown key")

    B_sliced = torch.chunk(B, tp_size, dim=0)
    C_sliced = torch.chunk(C, tp_size, dim=0)
    x_sliced = torch.chunk(x, tp_size, dim=0)

    tensor_sliced = []
    for (x, B, C) in zip(x_sliced, B_sliced, C_sliced):
        tensor_sliced.append(torch.cat((x, B.flatten(0, 1), C.flatten(0, 1)), dim=0))

    return tensor_sliced


def print_memory_usage(key, rank, num_ranks):
    '''Print memory usage.'''
    process = psutil.Process()
    mem_info = process.memory_info()
    print("> memory usage: '%s', rank %d / %d, mem %.1f/%.1f gb." % (
        key,
        rank,
        num_ranks,
        mem_info.rss / 1024**3,
        100 * mem_info.rss / process.memory_percent() / 1024**3,
    ))

class _ConverterFakeProcessGroup:
    def __init__(self, rank=0, size=1):
        self._rank = rank
        self._size = size

    def rank(self):
        return self._rank

    def size(self):
        return self._size

    def set_rank(self, rank):
        self._rank = rank

    def set_size(self, size):
        self._size = size
