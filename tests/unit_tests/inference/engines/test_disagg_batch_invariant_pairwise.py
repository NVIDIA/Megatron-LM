# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
from functools import partial

import pytest
import torch
import torch.distributed as dist
import transformer_engine.pytorch.module.linear as te_linear

from megatron.core.transformer import attention
from megatron.core.transformer.custom_layers import batch_invariant_kernels as bik
from megatron.core.transformer.enums import AttnBackend
from tests.unit_tests.inference.engines import disagg_test_utils as du
from tests.unit_tests.inference.engines import test_dynamic_engine as dynamic_tests
from tests.unit_tests.inference.engines.test_disagg_pairwise import transport_world  # noqa: F401


def _observe(engine, active):
    records, wrapper = ([], [], []), engine.controller.inference_wrapped_model
    original = wrapper.run_one_forward_step

    def forward(inference_input, *args, **kwargs):
        context = engine.context
        ids = context.request_ids[: context.total_request_count].tolist()
        if 101 not in ids:
            return original(inference_input, *args, **kwargs)
        row = ids.index(101)
        active[0] = records
        try:
            output = original(inference_input)
        finally:
            active[0] = None
        count = context.active_token_count
        owned = context.gpu_view.token_to_request_idx[:count] == row
        records[0].append(
            (
                inference_input["tokens"][0, :count][owned].detach().clone(),
                inference_input["position_ids"][0, :count][owned].detach().clone(),
                output.detach().clone(),
                row,
                (count, inference_input["tokens"].shape[1]),
                bool(context.request_in_prefill_status_tensor[row]),
            )
        )
        return output

    wrapper.run_one_forward_step = forward
    return records


def _install_factory(monkeypatch):
    constructor = dynamic_tests.TransformerConfig
    rounder = dynamic_tests.set_rounder
    active = [None]

    def trace(owner, name, index, match=None):
        original = getattr(owner, name)

        def traced(*args, **kwargs):
            if active[0] is not None and (match is None or args[0] is match):
                active[0][index].append(match or original)
            return original(*args, **kwargs)

        monkeypatch.setattr(owner, name, traced)

    fa3 = attention._flash_attn_forward
    trace(te_linear, "general_gemm", 1)
    trace(type(fa3), "__call__", 2, fa3)
    trace(attention, "flash_attn3_with_kvcache", 2)
    monkeypatch.setattr(dynamic_tests, "set_rounder", lambda _: rounder(64))
    monkeypatch.setattr(
        dynamic_tests,
        "TransformerConfig",
        partial(
            constructor,
            attention_backend=AttnBackend.flash,
            attention_dropout=0.0,
            batch_invariant_mode=True,
        ),
    )
    return active


def _assert_target_equal(left, right):
    assert all(torch.equal(a, b) for a, b in zip(left[:2], right[:2]))
    assert torch.isfinite(left[2]).all() and torch.isfinite(right[2]).all()
    assert torch.equal(left[2][0, left[3]], right[2][0, right[3]])


@torch.inference_mode()
def test_disaggregated_prefill_is_batch_invariant(transport_world, monkeypatch):
    active = _install_factory(monkeypatch)
    config = du.disagg_config(flash_attention_version=3, context_max_requests=128)
    tokens, neighbor_tokens, params = du.prompt(33), du.prompt(66)[1:], du.sampling(2)
    source = dist.get_rank() % 2 == 0
    try:
        bik.enable_batch_invariant_mode(backend="te_native", collective="ordered")
        with du.real_engine(config) as reference:
            weights = du.canonical_weights(reference)
            baseline, _, _ = _observe(reference, active)
            expected = du.run_to_completion(
                reference, reference.add_request(101, tokens, params)
            ).generated_tokens

        role = "prefill" if source else "decode"
        with du.real_engine(config, role=role, weights=weights) as engine:
            assert engine.context.batch_invariant_mode and engine.context.TOKEN_ROUNDER == 64
            assert bik.get_batch_invariant_backend() == "te_native"
            witness, gemms, kernels = _observe(engine, active)
            metadata = state = None
            if source:
                neighbor = engine.add_request(102, neighbor_tokens, du.sampling(1))
                future = engine.add_request(101, tokens, du.sampling(1, do_kv_handoff=True))
                request = du.run_to_completion(engine, future)
                assert du.run_to_completion(engine, neighbor).generated_tokens
                metadata, state = du.snapshot_source(engine, request)
            peer = du.exchange((metadata, state) if source else None, transport_world)
            if not source:
                metadata, state = peer
            pending, future = du.complete_transfer(
                engine, metadata, tokens, params, transport_world
            )
            if not source:
                assert len(pending.local_blocks) == 3 and pending.resume_tokens == expected[:1]
                du.assert_import_equal(engine, pending, state, len(tokens))
                du.admit_import(engine)
                assert du.run_to_completion(engine, future).generated_tokens == expected
            if source:
                engine._poll_pending_kv_pushes()
                engine.release_handoff_blocks(101)
            du.assert_released(engine)

        assert len(baseline) == 2 and baseline[0][4] == (33, 64)
        assert all((gemms, kernels))
        if source:
            assert len(witness) == 1 and witness[0][3:5] == (1, (98, 128))
            with pytest.raises(AssertionError):
                assert torch.equal(baseline[0][2][0, 0], witness[0][2][0, 0])
            _assert_target_equal(baseline[0], witness[0])
        else:
            assert len(witness) == 1 and not witness[0][5] and witness[0][4] == (1, 64)
            _assert_target_equal(baseline[1], witness[0])
    finally:
        bik.disable_batch_invariant_mode()
