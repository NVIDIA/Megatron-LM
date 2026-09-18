# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from types import SimpleNamespace

from megatron.core.transformer.moe import experts


def test_megamoe_expert_wgrads_split_across_p2p_windows(monkeypatch):
    calls = []

    class FakeExpert:
        def __init__(self, name, delay_wgrad=True):
            self.name = name
            self.config = SimpleNamespace(delay_megamoe_wgrad=delay_wgrad)

        def _backward_dw_fc2(self):
            calls.append(f"{self.name}.fc2")

        def _backward_dw_fc1(self):
            calls.append(f"{self.name}.fc1")

    class FakeModel:
        def __init__(self, modules):
            self._modules = modules

        def modules(self):
            return iter(self._modules)

    monkeypatch.setattr(experts, "TEGroupedMLP", FakeExpert)
    model = FakeModel(
        [
            object(),
            FakeExpert("first"),
            FakeExpert("not-delayed", delay_wgrad=False),
            FakeExpert("last"),
        ]
    )

    manager = experts.DelayedMegaMoeWgradManager([model])
    manager.on_backward_p2p_launched(0)
    assert calls == ["last.fc1", "first.fc1"]

    manager.on_forward_p2p_launched()
    assert calls == ["last.fc1", "first.fc1", "last.fc2", "first.fc2"]
