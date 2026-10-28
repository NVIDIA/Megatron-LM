# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Native PP/VPP graph replay parity with the same Dist-MoE eager trajectory."""

from contextlib import contextmanager
from unittest.mock import patch

import pytest

from tests.unit_tests.transformer.moe.test_dist_moe_pipeline_numeric import _run_pipeline_comparison


class PipelineGraphAudit:
    """Read captured diagnostic clones after replay, never synchronizing in capture.

    The numerical fixture's hooks run during capture and emit GPU clone nodes for
    layer/MoE/router outputs and gradients. Each native runner owns those clones.
    These wrappers only observe the real runner; they do not replace its kernels,
    graph calls, slot selection, ownership callbacks, or scheduling.
    """

    active = None

    def __init__(self, models, case):
        from megatron.core.transformer.cuda_graphs import _CudagraphReplayNode, _CudaGraphRunner

        assert PipelineGraphAudit.active is None
        PipelineGraphAudit.active = self
        self.case = case
        self.chunks = {id(layer): model.module for model in models for layer in model.module.layers}
        self.layers = [layer for model in models for layer in model.module.layers]
        self.observations = {}
        self.prefixes = {}
        self.replays = {"forward": 0, "backward": 0}
        self.contexts = None
        self.phases = []
        self.capture_counts = []
        self.recording = True
        self.patches = []

        @contextmanager
        def capture(runner):
            chunk = self.chunks[id(runner.base_module)]
            assert chunk.capture_observations is None
            chunk.capture_observations = self.observations.setdefault(runner, {})
            try:
                yield
            finally:
                chunk.capture_observations = None

        def wrap_capture(original):
            def call(runner, *args, **kwargs):
                with capture(runner):
                    return original(runner, *args, **kwargs)

            return call

        original_replay = _CudaGraphRunner.replay_graph_capture

        def replay(runner, *args, **kwargs):
            result = original_replay(runner, *args, **kwargs)
            chunk = self.chunks[id(runner.base_module)]
            self.prefixes[runner] = f"{chunk.phase}/mb{chunk.microbatch}"
            self.read(runner, backward=False)
            chunk._save_updated_parameters(runner.base_module)
            self.replays["forward"] += 1
            return result

        original_backward = _CudagraphReplayNode.backward

        def backward(ctx, *grads):
            result = original_backward(ctx, *grads)
            self.read(ctx.runner, backward=True)
            self.replays["backward"] += 1
            return result

        for name in ("create_fwd_graph", "create_bwd_graph"):
            self.patches.append(
                patch.object(_CudaGraphRunner, name, wrap_capture(getattr(_CudaGraphRunner, name)))
            )
        self.patches.extend(
            [
                patch.object(_CudaGraphRunner, "replay_graph_capture", replay),
                patch.object(_CudagraphReplayNode, "backward", staticmethod(backward)),
            ]
        )
        for item in self.patches:
            item.start()

    def read(self, runner, backward):
        chunk = self.chunks[id(runner.base_module)]
        for name, value in self.observations[runner].items():
            is_gradient = name.endswith(("dgrad", "router_score_grad"))
            if is_gradient == backward:
                chunk._save(f"{self.prefixes[runner]}/{name}", value)

    def check_capture(self):
        from megatron.core.transformer.cuda_graphs import _CudagraphGlobalRecord

        assert _CudagraphGlobalRecord.cudagraph_created, "Native schedule did not create graphs"
        assert self.observations
        for runner, values in self.observations.items():
            assert runner.cudagraph_created and runner.fwd_graph and runner.bwd_graph
            required = {
                "Y",
                "dgrad",
                "moe_Y",
                "moe_dgrad",
                "routing_map",
                "router_scores",
                "router_score_grad",
            }
            assert {name.split("/", 1)[1] for name in values} == required, values.keys()
            owners = runner.base_module.mlp.megakernel_experts.owners
            assert all(any(owner is p for p in runner.params_to_backprop) for owner in owners)
        bindings = self._bindings()
        if self.contexts is not None:
            assert bindings == self.contexts, "Runtime storage changed during recapture"
        self.contexts = bindings
        self.capture_counts.append(len(self.observations))
        self.recording = False

    def _bindings(self):
        return tuple(
            (
                id(layer.mlp.megakernel_experts.context),
                layer.mlp.megakernel_experts._graph_storage_binding(),
            )
            for layer in self.layers
        )

    def check_idle(self, phase):
        assert all(not layer.mlp.megakernel_experts._live for layer in self.layers)
        if self.contexts is not None:
            assert self._bindings() == self.contexts, "Captured runtime storage changed"
            if not self.recording:
                self.phases.append(phase)

    def finish(self):
        expected = self.case.steps * self.case.microbatches * len(self.layers)
        assert self.replays == {"forward": expected, "backward": expected}, self.replays
        assert self.phases == ["train0", "eval"] + [
            f"train{step}" for step in range(1, self.case.steps)
        ]
        assert len(self.capture_counts) == (2 if self.case.graph_recapture else 1)
        return {
            "captured_runners_per_cycle": self.capture_counts,
            "replays": self.replays,
            "expected_replays_per_direction": expected,
            "context_and_storage_stable_across": self.phases,
        }

    def release_graphs(self):
        from megatron.core.transformer.cuda_graphs import delete_cuda_graphs

        delete_cuda_graphs()
        for layer in self.layers:
            assert not layer.mlp.megakernel_experts._graph_bound
        self.observations.clear()
        self.prefixes.clear()
        self.recording = True

    def close(self):
        try:
            self.release_graphs()
        finally:
            for item in reversed(self.patches):
                item.stop()
            PipelineGraphAudit.active = None


@pytest.mark.internal
@pytest.mark.launch_on_gb200
@pytest.mark.parametrize("precision", ["bf16", "mxfp8"])
@pytest.mark.parametrize("vpp", [1, 2, 3], ids=["pp2", "pp2_vpp2", "pp2_vpp3"])
def test_dist_moe_pipeline_graph_numeric(precision, vpp):
    """Compare default-Adam optimizer trajectories, including all internal observations."""
    _run_pipeline_comparison(precision, vpp, cuda_graph=True)
