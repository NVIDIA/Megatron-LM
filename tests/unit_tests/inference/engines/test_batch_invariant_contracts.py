# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Output, terminal, residency, and explicit unsupported BI contracts."""

import pytest
import torch

from megatron.core.inference.config import KVCacheManagementMode as KVMode
from megatron.core.transformer.enums import AttnBackend
from megatron.core.transformer.transformer_config import TransformerConfig
from tests.unit_tests.inference.engines import batch_invariant_test_utils as fixture


@pytest.mark.parametrize(
    "invalid,reason",
    [
        ({"params_dtype": torch.float16}, "BF16 model parameters"),
        ({"flash_attention_version": 2}, "flash-attention-version 3 or 4"),
        ({"context_parallel_size": 2}, "does not support context parallelism"),
        ({"attention_dropout": 0.1}, "does not support attention dropout"),
    ],
)
def test_batch_invariant_support_boundary(invalid, reason):
    # Negative support-only: deliberately no model execution or positive pair credit.
    kwargs = dict(
        num_layers=2,
        hidden_size=128,
        num_attention_heads=4,
        batch_invariant_mode=True,
        params_dtype=torch.bfloat16,
        attention_backend=AttnBackend.flash,
        flash_attention_version=3,
        attention_dropout=0.0,
    )
    kwargs.update(invalid)
    with pytest.raises(AssertionError, match=reason):
        TransformerConfig(**kwargs)


def test_batch_invariant_rejects_flashinfer_rope():
    # Main rejects this numerical path; it is no longer a supported positive pair.
    case = fixture.Case("fused-rope", context={"use_flashinfer_fused_rope": True})
    with fixture.invariant_runtime(case) as (backend, version):
        with pytest.raises(ValueError, match="does not support FlashInfer fused RoPE"):
            fixture.build_engine(case, backend, version)


@pytest.mark.parametrize("output", ["top-n", "skip-prompt", "total-length", "bos"])
def test_batch_invariant_output_contract(output):
    options = dict(top_n_logprobs=3)
    if output == "skip-prompt":
        options["skip_prompt_log_probs"] = True
    if output == "total-length":
        options.update(num_tokens_to_generate=None, num_tokens_total=23)
    if output == "bos":
        options["add_BOS"] = True
    case = fixture.Case("output", sampling=options)
    with (
        fixture.invariant_runtime(case) as (backend, version),
        pytest.MonkeyPatch.context() as patch,
    ):
        if output == "bos":
            original_prompt = fixture.target_prompt
            patch.setattr(
                fixture, "target_prompt", lambda n: " ".join(map(str, original_prompt(n)))
            )
        reference = fixture.run_order(case, backend, version, "solo")
        actual = fixture.run_order(case, backend, version, "back")
        fixture.assert_same_target(reference, actual)
        request = actual[0]
        assert request.succeeded() and request.generated_length == 6
        assert len(request.generated_tokens) == len(request.generated_log_probs) == 6
        assert len(request.generated_top_n_logprobs) == 6
        assert all(len(row) == 3 for row in request.generated_top_n_logprobs)
        assert [step["token"] for step in actual[1].sample_steps] == request.generated_tokens
        for token, score, top_n in zip(
            request.generated_tokens, request.generated_log_probs, request.generated_top_n_logprobs
        ):
            assert top_n[str(token)] == score
        if output == "skip-prompt":
            assert not request.prompt_log_probs and not request.prompt_top_n_logprobs
        else:
            assert len(request.prompt_log_probs) == len(request.prompt_tokens) - 1
            assert len(request.prompt_top_n_logprobs) == len(request.prompt_tokens) - 1
        if output == "bos":
            assert request.prompt_tokens[0] == 1 and len(request.prompt_tokens) == 18
        if output == "total-length":
            assert len(request.prompt_tokens) + request.generated_length == 23
        assert actual[1].sample_steps[0]["requests"] == 65


@pytest.mark.parametrize(
    "mode,static_pointers",
    [(m, s) for m in KVMode for s in (False, True) if s or m != KVMode.OFFLOAD],
)
def test_batch_invariant_suspend_resume(mode, static_pointers):
    # Production RL validation rejects OFFLOAD without persistent graph pointers.
    uses_uvm = static_pointers and mode == KVMode.RECOMPUTE
    case = fixture.Case(
        "suspend",
        context={
            "kv_cache_management_mode": mode,
            "static_kv_memory_pointers": static_pointers,
            "unified_memory_level": int(uses_uvm),
            "num_cuda_graphs": 4,
            "max_tokens": 128,
        },
    )
    forwards = []
    with (
        fixture.invariant_runtime(case) as (backend, version),
        pytest.MonkeyPatch.context() as patch,
    ):
        reference = fixture.run_order(case, backend, version, "solo")
        build = fixture.build_engine

        def resumed_engine(*args, **kwargs):
            engine = build(*args, **kwargs)
            original = engine.step_modern
            context = engine.context
            suspended = False
            resumed = False

            def active_ids():
                active = context.request_ids[
                    context.paused_request_count : context.total_request_count
                ]
                return tuple(map(int, active.tolist()))

            def observe_forward():
                ids = active_ids()
                if fixture.TARGET in ids and context.is_decode_only():
                    forwards.append((resumed, ids))

            engine._bi_after_forward = observe_forward

            def step(*args, **kwargs):
                nonlocal resumed, suspended
                result = original(*args, **kwargs)
                if not suspended and forwards and fixture.TARGET in engine.requests:
                    request = engine.get_request(fixture.TARGET)
                    if len(request.generated_tokens or []) >= 2:
                        before = tuple(request.generated_tokens)
                        live = forwards[-1][1]
                        assert fixture.TARGET in live and set(live) - {fixture.TARGET}
                        assert set(active_ids()) - {fixture.TARGET}
                        pointer = context.memory_buffer.data_ptr()
                        record = engine.requests[fixture.TARGET].record
                        segments = len(record.requests)
                        target_idx = int((context.request_ids == fixture.TARGET).nonzero()[0])
                        block = int(context.request_to_kv_block_ids[target_idx, 0])
                        stored = len(request.prompt_tokens) + len(before) - 1
                        kv = context.memory_buffer[:, :, block, :stored].cpu().clone()
                        physical = torch.cuda.mem_get_info()[0]
                        engine.suspend()
                        torch.cuda.synchronize()
                        suspended_free = torch.cuda.mem_get_info()[0]
                        assert not context.is_tensor_state_allocated
                        if mode == KVMode.OFFLOAD:
                            assert context._uses_torch_memory_saver
                            assert suspended_free > physical
                            print("BI_OFFLOAD_WITNESS", block, stored, suspended_free)
                        if mode == KVMode.RECOMPUTE:
                            assert len(record.requests) == segments + 1
                        engine.resume()
                        resumed = True
                        merged = record.merge()
                        assert tuple(merged.generated_tokens or []) == before
                        if mode == KVMode.RECOMPUTE:
                            assert fixture.TARGET in engine.waiting_request_ids
                            assert not engine.get_request(fixture.TARGET).generated_tokens
                        else:
                            assert torch.equal(
                                context.memory_buffer[:, :, block, :stored].cpu(), kv
                            )
                        if static_pointers or mode == KVMode.PERSIST:
                            assert context.memory_buffer.data_ptr() == pointer
                        print("BI_RESIDENCY_WITNESS", mode.value, live, len(before))
                        suspended = True
                return result

            patch.setattr(engine, "step_modern", step)
            return engine

        patch.setattr(fixture, "build_engine", resumed_engine)
        actual = fixture.run_order(case, backend, version, "back")
        fixture.assert_same_target(reference, actual)
        decode_steps = [step for step in actual[1].steps if step["decode"]]
        assert len(forwards) == len(decode_steps), "missing target residency forward observation"
        assert any(
            resumed and step["replay"] and step["captured_graphs"]
            for (resumed, _), step in zip(forwards, decode_steps)
        ), "no target-owned resumed replay with captured backend provenance"
