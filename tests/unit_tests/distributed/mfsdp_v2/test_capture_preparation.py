# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""FSDP readiness and rank agreement before full-iteration graph capture."""

import unittest
from contextlib import ExitStack, nullcontext
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from megatron.core.distributed.fsdp.src.megatron_fsdp.experimental.context import (
    FsdpContext,
    GradientFinalizationMode,
)
from megatron.core.distributed.fsdp.src.megatron_fsdp.experimental.execution_order import (
    ExecutionEvent,
)
from megatron.core.distributed.fsdp.src.megatron_fsdp.experimental.phase import Phase
from megatron.core.distributed.fsdp.src.megatron_fsdp.experimental.runtime_policy import (
    PrefetchOrder,
    RuntimePolicy,
)
from megatron.core.full_cuda_graph import FullCudaGraphWrapper


class _Status:
    def __init__(self, values, **_):
        self.value = values[0]

    def item(self):
        return self.value


class TestContextCapturePreparation(unittest.TestCase):
    def setUp(self):
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.stack.enter_context(
            patch.object(torch.cuda, "is_current_stream_capturing", return_value=False)
        )
        self.stack.enter_context(patch.object(torch.cuda, "Stream", return_value=Mock()))

    def _context(self, order=PrefetchOrder.RECORDED):
        context = FsdpContext(
            device=torch.device("cuda"),
            runtime_policy=RuntimePolicy(prefetch_order=order),
            gradient_finalization_mode=GradientFinalizationMode.REDUCE_OUTER,
        )
        owner = torch.nn.Identity()
        owner.parameter_groups = (Mock(_unshard_events={}),)
        owner._schedule_policy = SimpleNamespace(register_hooks=True)
        owner.reshard = Mock()
        context.register_module(owner)
        context.finalize()
        return context, owner

    def test_recorded_training_requires_a_complete_eager_window(self):
        context, owner = self._context()
        with self.assertRaisesRegex(RuntimeError, "complete execution window"):
            context.prepare_for_cuda_graph_capture()
        for phase in (Phase.FORWARD, Phase.BACKWARD):
            context.execution_order.append(ExecutionEvent(owner, phase))
        context.post_backward()
        view = context.execution_order_view
        view.advance()
        with self.assertRaisesRegex(RuntimeError, "complete execution window"):
            context.prepare_for_cuda_graph_capture()
        view.advance()
        # The U-only hint has wrapped, but the complete replay still needs its boundary.
        for _ in range(2):
            context.prefetch_order_view.advance()
        self.assertEqual(context.prefetch_order_view.position, 0)
        with self.assertRaisesRegex(RuntimeError, "complete execution window"):
            context.prepare_for_cuda_graph_capture()
        context.post_backward()
        context.prepare_for_cuda_graph_capture()

    def test_registration_evaluation_and_clean_capture_boundaries(self):
        context, owner = self._context(PrefetchOrder.REGISTRATION)
        context.prepare_for_cuda_graph_capture()
        for recorded in (False, True):
            context, owner = self._context()
            if recorded:
                for phase in (Phase.FORWARD, Phase.BACKWARD):
                    context.execution_order.append(ExecutionEvent(owner, phase))
                context.post_backward()
                context.prefetch_order_view.advance()
            owner.eval()
            position = context.prefetch_order_view.position
            context.prepare_for_cuda_graph_capture()
            self.assertEqual(context.prefetch_order_view.position, position)
            self.assertEqual(context._recording_execution_order, not recorded)
        context, owner = self._context(PrefetchOrder.REGISTRATION)
        for attribute, dirty in (
            ("_is_finalized", False),
            ("_post_autograd_hook_registered", True),
            ("_delayed_post_backward_callbacks", [(owner, Mock())]),
        ):
            with self.subTest(attribute=attribute), patch.object(context, attribute, dirty):
                with self.assertRaises(RuntimeError):
                    context.prepare_for_cuda_graph_capture()
        with patch.object(torch.cuda, "is_current_stream_capturing", return_value=True):
            with self.assertRaisesRegex(RuntimeError, "before entering"):
                context.prepare_for_cuda_graph_capture()
        owner.parameter_groups[0]._unshard_events = {Phase.FORWARD: object()}
        with self.assertRaisesRegex(RuntimeError, "gathered weights"):
            context.prepare_for_cuda_graph_capture()


class TestGraphCapturePreparation(unittest.TestCase):
    def setUp(self):
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.wrapper = FullCudaGraphWrapper.__new__(FullCudaGraphWrapper)
        self.wrapper.control_group = object()
        self.prefix = "megatron.core.full_cuda_graph"
        self.stack.enter_context(patch(self.prefix + ".torch.tensor", side_effect=_Status))
        self.agreement = self.stack.enter_context(
            patch(self.prefix + ".torch.distributed.all_reduce")
        )
        self.context = SimpleNamespace(prepare_for_cuda_graph_capture=Mock())
        self.model = SimpleNamespace(_context=self.context)

    def test_stage_without_context_participates_in_agreement(self):
        self.wrapper._prepare_fsdp_capture(SimpleNamespace())
        self.agreement.assert_called_once()

    def test_shared_context_is_prepared_once(self):
        self.wrapper._prepare_fsdp_capture([self.model, self.model])
        self.context.prepare_for_cuda_graph_capture.assert_called_once()
        self.agreement.assert_called_once()

    def test_local_failure_is_agreed_before_raise(self):
        self.context.prepare_for_cuda_graph_capture.side_effect = ValueError("invalid schedule")
        with self.assertRaisesRegex(ValueError, "invalid schedule"):
            self.wrapper._prepare_fsdp_capture(self.model)
        self.agreement.assert_called_once()

    def test_peer_failure_rejects_stage_without_context(self):
        self.agreement.side_effect = lambda status, **_: setattr(status, "value", 1)
        with self.assertRaisesRegex(RuntimeError, "another rank"):
            self.wrapper._prepare_fsdp_capture(SimpleNamespace())

    def _run_wrapper(self, *, training, capture):
        stage = "training" if training else "validation"
        graph = SimpleNamespace(replay=Mock())
        rerun = SimpleNamespace(
            capture_deferred_validations=lambda _: nullcontext(), run_deferred_validations=Mock()
        )
        self.wrapper.cuda_graph_warmup_steps = 1
        self.wrapper.use_single_mempool = False
        self.wrapper.data_read = Mock(return_value=[])
        self.wrapper.forward_backward_func = Mock(return_value="result")
        self.wrapper._prepare_fsdp_capture = Mock()
        self.stack.enter_context(
            patch.object(FullCudaGraphWrapper, "curr_iteration", {stage: 1 if capture else 2})
        )
        self.stack.enter_context(
            patch.object(FullCudaGraphWrapper, "cuda_graph", {stage: None if capture else graph})
        )
        self.stack.enter_context(patch.object(FullCudaGraphWrapper, "result", {stage: None}))
        for target in (
            "torch.distributed.barrier",
            "torch.cuda.synchronize",
            "prime_cuda_rng_states_for_graph_capture",
        ):
            self.stack.enter_context(patch(self.prefix + "." + target))
        self.stack.enter_context(
            patch(
                self.prefix + ".torch.autograd.graph.set_override_stale_capture_stream", create=True
            )
        )
        self.stack.enter_context(patch(self.prefix + ".torch.cuda.CUDAGraph", return_value=graph))
        self.stack.enter_context(
            patch(self.prefix + ".torch.cuda.graph", side_effect=lambda *_, **__: nullcontext())
        )
        self.stack.enter_context(
            patch(self.prefix + ".capture_constants", side_effect=lambda _: nullcontext())
        )
        self.stack.enter_context(
            patch(self.prefix + ".get_rerun_state_machine", return_value=rerun)
        )
        self.stack.enter_context(patch(self.prefix + ".get_shared_capture_stream"))
        self.stack.enter_context(patch(self.prefix + ".get_graph_pool"))
        self.stack.enter_context(patch(self.prefix + ".symmetric_wgrad_pool"))
        self.stack.enter_context(
            patch(self.prefix + ".cudagraph_needs_generator_registration", return_value=False)
        )
        self.wrapper(
            model=self.model,
            data_iterator=None,
            num_microbatches=1,
            seq_length=1,
            forward_only=not training,
        )
        graph.replay.assert_called_once_with()
        self.agreement.assert_not_called()
        return self.wrapper._prepare_fsdp_capture.call_args_list

    def test_training_capture_prepares_once_before_initial_replay(self):
        calls = self._run_wrapper(training=True, capture=True)
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0].args, (self.model,))
        self.assertEqual(calls[0].kwargs, {})

    def test_later_replay_does_not_prepare_again(self):
        calls = self._run_wrapper(training=True, capture=False)
        self.assertEqual(calls, [])

    def test_evaluation_capture_prepares_once(self):
        calls = self._run_wrapper(training=False, capture=True)
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0].args, (self.model,))
