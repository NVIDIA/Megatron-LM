# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import torch.nn as nn

from megatron.lite.model.deepseek_v4.lite import protocol
from megatron.lite.model.protocol_utils import set_cross_entropy_fusion


class _ForwardRecorder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.kwargs = None

    def forward(self, **kwargs):
        self.kwargs = kwargs
        return {}


class _Wrapper(nn.Module):
    def __init__(self, module: nn.Module) -> None:
        super().__init__()
        self.module = module

    def forward(self, **kwargs):
        return self.module(**kwargs)


def test_cross_entropy_fusion_reaches_wrapped_ds4_forward(monkeypatch) -> None:
    native = _ForwardRecorder()
    set_cross_entropy_fusion([native], True)
    wrapper = _Wrapper(native)
    monkeypatch.setattr(protocol, "_prepare_model_forward_kwargs", lambda _model, _batch: {})

    protocol._forward_step(wrapper, object())

    assert native.kwargs == {"use_fused_kernels": True}


def test_impl_config_accepts_cross_entropy_fusion() -> None:
    assert protocol.ImplConfig(cross_entropy_fusion=True).cross_entropy_fusion is True
