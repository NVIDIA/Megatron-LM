from typing import final, override

import torch

from megatron.core.metrics import collector, forwarding_collector


@final
class SimpleCollector(collector.MetricCollector):
    """Unit test collector that stores collected metrics in a list.

    Attributes:
        collected: A list of tuples where each tuple contains a module and the metrics reported, in
            the order they were collected.
    """

    def __init__(self):
        self.collected: list[tuple[torch.nn.Module, dict[str, torch.Tensor]]] = []

    @override
    def collect(self, module: torch.nn.Module, **params: torch.Tensor) -> None:
        self.collected.append((module, params))


class DummyModule(torch.nn.Module):
    """A no-op module that collects a dummy metric during its forward pass."""

    def __init__(self, metric_collector: collector.MetricCollector):
        super().__init__()
        self._metric_collector = metric_collector

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """A dummy forward method that simply collects the input as a metric and returns it."""
        self._metric_collector.collect(self, dummy_metric=x)
        return x


def test_collector_collects_metrics():
    collector_instance = SimpleCollector()
    module = DummyModule(metric_collector=collector_instance)

    input_tensor = torch.tensor([1.0, 2.0, 3.0])
    output_tensor = module(input_tensor)

    assert torch.equal(output_tensor, input_tensor), "Output tensor should match input tensor."

    assert collector_instance.collected == [
        (module, {"dummy_metric": input_tensor})
    ], "Collector should have recorded the metric."


def test_noop_collector_does_nothing():
    collector_instance = collector.NoopMetricCollector()
    module = DummyModule(metric_collector=collector_instance)

    input_tensor = torch.tensor([1.0, 2.0, 3.0])
    output_tensor = module(input_tensor)

    assert torch.equal(output_tensor, input_tensor), "Output tensor should match input tensor."


def test_forward_collector_forwards_metrics():
    test_collector_1 = SimpleCollector()
    test_collector_2 = SimpleCollector()
    collector_instance = forwarding_collector.ForwardingCollector(
        test_collector_1, test_collector_2
    )
    module = DummyModule(metric_collector=collector_instance)

    input_tensor = torch.tensor([1.0, 2.0, 3.0])
    output_tensor = module(input_tensor)

    assert torch.equal(output_tensor, input_tensor), "Output tensor should match input tensor."

    assert test_collector_1.collected == [
        (module, {"dummy_metric": input_tensor})
    ], "Collector 1 should have recorded the metric."

    assert test_collector_2.collected == [
        (module, {"dummy_metric": input_tensor})
    ], "Collector 2 should have recorded the metric."

    test_collector_3 = SimpleCollector()
    collector_instance.add_subscriber(test_collector_3)

    output_tensor = module(input_tensor)

    assert test_collector_1.collected == [
        (module, {"dummy_metric": input_tensor}),
        (module, {"dummy_metric": input_tensor}),
    ], "Collector 1 should have recorded the metric again."
    assert test_collector_3.collected == [
        (module, {"dummy_metric": input_tensor})
    ], "Collector 3 should have recorded the metric."
