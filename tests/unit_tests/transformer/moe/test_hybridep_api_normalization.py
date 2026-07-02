# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

import warnings

from megatron.core.transformer.moe import fused_a2a


class _HybridEPBufferWithoutFusedArgs:
    def dispatch_with_permute(self, hidden):
        return hidden


class _HybridEPBufferWithFusedArgs:
    def dispatch_with_permute(self, hidden, fuse_permute_dispatch=False):
        return hidden, fuse_permute_dispatch


def _reset_cache():
    fused_a2a._hybrid_ep_supports_fused_dispatch = None
    fused_a2a._hybrid_ep_warned_unsupported_fused_dispatch = False


def test_hybridep_dispatch_options_drop_unsupported_kwargs(monkeypatch):
    _reset_cache()
    monkeypatch.setattr(fused_a2a, "HybridEPBuffer", _HybridEPBufferWithoutFusedArgs)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        normalized = fused_a2a._normalize_hybrid_ep_dispatch_options(True, 128, 256)
        normalized_again = fused_a2a._normalize_hybrid_ep_dispatch_options(True, 128, 256)

    assert normalized == (False, None, None)
    assert normalized_again == (False, None, None)
    assert len(caught) == 1
    assert "does not support fused permute dispatch" in str(caught[0].message)


def test_hybridep_dispatch_options_keep_supported_kwargs(monkeypatch):
    _reset_cache()
    monkeypatch.setattr(fused_a2a, "HybridEPBuffer", _HybridEPBufferWithFusedArgs)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        normalized = fused_a2a._normalize_hybrid_ep_dispatch_options(True, 128, 256)

    assert normalized == (True, 128, 256)
    assert caught == []
