# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Targeted battery for the transactional async-scheduling decode pipeline.

The spine of this battery is :func:`assert_async_equals_serial`, which runs the
same prompts/seed twice -- once with ``enable_async_scheduling=False`` (the serial
ground truth) and once ``True`` -- and asserts identical generated token ids. The
serial path is the oracle; the async path must reproduce it token-for-token.

In the earliest commits the flag is a no-op, so the equivalence check passes
trivially and exists to lock in the harness. As the async pipeline is built up
(launch-before-commit, closed survivor set, hybrid single-bank, EP, MTP), the same
check becomes the load-bearing correctness gate.

Run (8 GPUs, in its own ``torch.distributed.run`` invocation)::

    /opt/venv/bin/python -m torch.distributed.run --nproc-per-node 8 -m pytest -v \
        tests/unit_tests/inference/test_async_sched_txn.py
"""

import argparse

import pytest
import torch

from megatron.core.inference.config import InferenceConfig
from megatron.core.transformer.cuda_graphs import delete_cuda_graphs
from megatron.core.utils import is_fa_min_version
from tests.unit_tests.inference.engines.test_dynamic_engine import (
    DynamicEngineTestConfig,
    DynamicInferenceEngineTestBase,
    set_rounder,
)
from tests.unit_tests.test_utilities import Utils


def _to_token_list(tokens):
    """Normalize a generated-token container (list or tensor) to a list of ints."""
    if tokens is None:
        return []
    if isinstance(tokens, torch.Tensor):
        return tokens.detach().cpu().tolist()
    return list(tokens)


def _collect_outputs(env):
    """Map request_id -> (status, generated token ids) after a completed run."""
    outputs = {}
    for request in env.requests:
        outputs[request.request_id] = (
            request.status,
            _to_token_list(getattr(request, "output", None)),
        )
    return outputs


class AsyncSchedTxnTestBase(DynamicInferenceEngineTestBase):
    """Shared helpers for the async-scheduling battery.

    Reuses the model/context/engine fixtures from the dynamic-engine test base so
    the async path is exercised against the exact same tiny GPT / hybrid configs.
    """

    @classmethod
    def assert_async_equals_serial(cls, **test_config_kwargs):
        """Token-exact ``async == serial`` equivalence.

        Builds and runs the engine twice from identical seeds/config -- once serial
        (``enable_async_scheduling=False``) and once async (``True``) -- and asserts
        every request produced the identical status and generated token ids.

        Returns the (serial_env, async_env) pair for any further assertions.
        """
        # Disallow the caller overriding the toggle; this helper owns it.
        test_config_kwargs.pop("enable_async_scheduling", None)

        serial_env = cls._run_test(enable_async_scheduling=False, **test_config_kwargs)
        serial_outputs = _collect_outputs(serial_env)

        async_env = cls._run_test(enable_async_scheduling=True, **test_config_kwargs)
        async_outputs = _collect_outputs(async_env)

        assert set(serial_outputs.keys()) == set(async_outputs.keys()), (
            f"request id sets differ: serial={sorted(serial_outputs)} "
            f"async={sorted(async_outputs)}"
        )

        mismatches = []
        for request_id, (serial_status, serial_tokens) in serial_outputs.items():
            async_status, async_tokens = async_outputs[request_id]
            if serial_status != async_status or serial_tokens != async_tokens:
                mismatches.append(
                    f"  request {request_id}: "
                    f"serial=({serial_status}, {serial_tokens}) "
                    f"async=({async_status}, {async_tokens})"
                )
        assert not mismatches, "async != serial for:\n" + "\n".join(mismatches)

        return serial_env, async_env


@pytest.mark.skipif(
    not is_fa_min_version("2.7.3"), reason="need latest flash attn for dynamic batching"
)
class TestAsyncSchedTxnFlag(AsyncSchedTxnTestBase):
    """C0: opt-in flag plumbing + the serial-equivalence spine (flag is a no-op here)."""

    @classmethod
    def setup_class(cls):
        Utils.initialize_model_parallel(
            tensor_model_parallel_size=1,
            pipeline_model_parallel_size=1,
            expert_model_parallel_size=1,
            expert_tensor_parallel_size=1,
        )

    @classmethod
    def teardown_class(cls):
        delete_cuda_graphs()
        set_rounder(64)
        Utils.destroy_model_parallel()

    @pytest.mark.internal
    def test_arg_maps_to_config_dest(self) -> None:
        """The Megatron arg parses to the documented dest, defaulting to False."""
        from megatron.training.arguments import _add_inference_args

        parser = argparse.ArgumentParser()
        _add_inference_args(parser)

        # Default is off.
        default_args = parser.parse_args([])
        assert default_args.inference_dynamic_batching_async_scheduling is False

        # Enable flag.
        on_args = parser.parse_args(["--inference-dynamic-batching-async-scheduling"])
        assert on_args.inference_dynamic_batching_async_scheduling is True

        # BooleanOptionalAction provides an explicit --no- variant.
        off_args = parser.parse_args(["--no-inference-dynamic-batching-async-scheduling"])
        assert off_args.inference_dynamic_batching_async_scheduling is False

    @pytest.mark.internal
    def test_config_field_defaults_false(self) -> None:
        """The config field exists and defaults to False (disabled == legacy)."""
        assert InferenceConfig().enable_async_scheduling is False
        assert InferenceConfig(enable_async_scheduling=True).enable_async_scheduling is True

    @pytest.mark.internal
    def test_flag_plumbs_to_context(self) -> None:
        """The flag reaches the live DynamicInferenceContext both off and on."""
        set_rounder(4)
        for enabled in (False, True):
            env = self._build_test_env(
                DynamicEngineTestConfig(model_provider="gpt", enable_async_scheduling=enabled)
            )
            assert env.engine.context.enable_async_scheduling is enabled

    @pytest.mark.internal
    def test_async_equals_serial_gpt_decode(self) -> None:
        """Spine: tiny GPT greedy decode is identical with the flag off vs on.

        In C0 the flag is a no-op, so this is a determinism + harness check; the
        same assertion becomes the real correctness gate in later commits.
        """
        self.assert_async_equals_serial(
            model_provider="gpt",
            num_tokens_to_generate=8,
            num_requests=4,
            num_gap_steps=1,
        )
