# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from types import SimpleNamespace
from unittest import mock

import torch

from megatron.core.distributed.fsdp.src.megatron_fsdp import utils


def test_find_megatron_fsdp_preserves_v1_wrapper(monkeypatch):
    wrapper = object()
    monkeypatch.setattr(utils, "_find_megatron_fsdp_v1", lambda _model: wrapper)
    monkeypatch.setattr(utils, "_find_megatron_fsdp_v2_root", mock.Mock())

    assert utils.find_megatron_fsdp(torch.nn.Module()) is wrapper
    utils._find_megatron_fsdp_v2_root.assert_not_called()


def test_find_megatron_fsdp_returns_cached_v2_compat_proxy(monkeypatch):
    root = torch.nn.Module()
    root._fsdp_param_groups = [SimpleNamespace(sharding_strategy="optim_grads_params")]
    monkeypatch.setattr(utils, "_find_megatron_fsdp_v1", lambda _model: None)
    monkeypatch.setattr(utils, "_find_megatron_fsdp_v2_root", lambda _model: root)

    proxy = utils.find_megatron_fsdp(root)

    assert proxy is utils.find_megatron_fsdp(root)
    assert proxy.ddp_config.data_parallel_sharding_strategy == "optim_grads_params"

    with (
        mock.patch(
            "megatron.core.distributed.fsdp.src.megatron_fsdp.v2.hooks." "mfsdp_pre_backward_setup"
        ) as pre_backward,
        mock.patch(
            "megatron.core.distributed.fsdp.src.megatron_fsdp.v2.hooks."
            "mfsdp_post_backward_final_callback"
        ) as post_backward,
        mock.patch(
            "megatron.core.distributed.fsdp.src.megatron_fsdp.v2.hooks." "mfsdp_post_forward_hook"
        ) as post_forward_release,
        mock.patch(
            "megatron.core.distributed.fsdp.src.megatron_fsdp.v2.hooks." "mfsdp_post_backward_hook"
        ) as post_backward_release,
    ):
        layer = torch.nn.Module()
        proxy._replace_param_with_raw_if_needed()
        proxy.pre_backward()
        proxy.post_backward()
        proxy.post_forward_release_module(layer)
        proxy.post_backward_release_module(layer)

    pre_backward.assert_called_once_with(root, skip_final_callback=True)
    post_backward.assert_called_once_with(root)
    post_forward_release.assert_called_once_with(layer)
    post_backward_release.assert_called_once_with(layer)
