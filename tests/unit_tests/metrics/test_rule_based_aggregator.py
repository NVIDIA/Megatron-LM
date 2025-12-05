import collections
from unittest import mock

import pytest
import torch
from torch import testing as torch_testing

from megatron.core import process_groups_config
from megatron.core.metrics import (
    collector,
    every_n_reporter,
    forwarding_collector,
    rule_based_aggregator,
)


class Multiply(torch.nn.Module):

    def __init__(self, scalar: float, metric_collector: collector.MetricCollector):
        super().__init__()
        self._multiply_by = scalar
        self._metric_collector = metric_collector

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self._metric_collector.collect(self, multiplied=x)
        return x * self._multiply_by


class Add(torch.nn.Module):

    def __init__(self, scalar: float, metric_collector: collector.MetricCollector):
        super().__init__()
        self._add_value = scalar
        self._metric_collector = metric_collector

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self._metric_collector.collect(self, added=x)
        return x + self._add_value


class ChildModule(torch.nn.Module):

    def __init__(self, metric_collector: collector.MetricCollector):
        super().__init__()

        self.first = Multiply(2.0, metric_collector)
        self.second = Add(3.0, metric_collector)
        self._metric_collector = metric_collector

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.first(x)
        z = self.second(y)
        self._metric_collector.collect(self, y=y, z=z)
        return z


class SplitModule(torch.nn.Module):
    def __init__(self, splits: int, metric_collector: collector.MetricCollector):
        super().__init__()
        self.layers = torch.nn.ModuleList([ChildModule(metric_collector) for _ in range(splits)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        outputs = []
        for layer, shard in zip(self.layers, x.split(x.size(0) // len(self.layers))):
            outputs.append(layer(shard))
        return torch.cat(outputs, dim=0)


class SumRule(rule_based_aggregator.AggregationRule):
    """Aggregation rule that returns whatever params were passed in collect and sums them in aggregate."""

    @classmethod
    def collect(cls, **params: torch.Tensor) -> dict[str, torch.Tensor]:
        """Collects raw metrics and sums their values."""
        return {k: v.sum() for k, v in params.items()}

    @classmethod
    def aggregate(
        cls,
        collected_metrics: list[dict[str, torch.Tensor]],
        pg_collection: process_groups_config.ProcessGroupCollection,
    ) -> dict[str, float]:
        """Aggregates collected metrics by summing them across forward passes."""
        del pg_collection  # unused in this simple rule
        assert len(collected_metrics) == 1, "SumRule expects a single entry per module instance."
        return {k: float(v.item()) for k, v in collected_metrics[0].items()}


def test_aggregator_collects_metrics():
    collector = forwarding_collector.ForwardingCollector()
    root = SplitModule(splits=2, metric_collector=collector)

    # Unused in this test
    pg_collection = mock.create_autospec(spec=process_groups_config.ProcessGroupCollection)

    aggregator = rule_based_aggregator.RuleBasedAggregator(
        root, {ChildModule: SumRule, Multiply: SumRule, Add: SumRule}, pg_collection=pg_collection
    )
    collector.add_subscriber(aggregator)

    result = root(torch.tensor([[1.0, 2.0], [3.0, 4.0]]))
    torch_testing.assert_close(result, torch.tensor([[5.0, 7.0], [9.0, 11.0]]))

    assert aggregator.report(prefix='root/') == {
        'root/ModuleList.0.ChildModule.0.Multiply.0.multiplied': pytest.approx(3.0),
        'root/ModuleList.0.ChildModule.0.Add.1.added': pytest.approx(6.0),
        'root/ModuleList.0.ChildModule.0.y': pytest.approx(6.0),
        'root/ModuleList.0.ChildModule.0.z': pytest.approx(12.0),
        'root/ModuleList.0.ChildModule.1.Multiply.0.multiplied': pytest.approx(7.0),
        'root/ModuleList.0.ChildModule.1.Add.1.added': pytest.approx(14.0),
        'root/ModuleList.0.ChildModule.1.y': pytest.approx(14.0),
        'root/ModuleList.0.ChildModule.1.z': pytest.approx(20.0),
    }


class AverageAcrossPasses(rule_based_aggregator.AggregationRule):
    """Aggregation rule that returns whatever params were passed in collect and sums them in aggregate."""

    @classmethod
    def collect(cls, **params: torch.Tensor) -> dict[str, torch.Tensor]:
        """Collects raw metrics and sums their values."""
        return {k: v.sum() for k, v in params.items()}

    @classmethod
    def aggregate(
        cls,
        collected_metrics: list[dict[str, torch.Tensor]],
        pg_collection: process_groups_config.ProcessGroupCollection,
    ) -> dict[str, float]:
        """Averages collected metrics by summing them across forward passes."""
        del pg_collection  # unused in this simple rule
        totals: dict[str, list[float]] = collections.defaultdict(list)
        for entry in collected_metrics:
            for k, v in entry.items():
                totals[k].append(float(v.item()))
        return {k: sum(vs) / len(vs) for k, vs in totals.items()}


def test_aggregator_aggregates_across_passes():
    collector = forwarding_collector.ForwardingCollector()
    root = SplitModule(splits=2, metric_collector=collector)

    # Unused in this test
    pg_collection = mock.create_autospec(spec=process_groups_config.ProcessGroupCollection)

    aggregator = rule_based_aggregator.RuleBasedAggregator(
        root,
        {ChildModule: AverageAcrossPasses, Multiply: AverageAcrossPasses, Add: AverageAcrossPasses},
        pg_collection=pg_collection,
    )
    collector.add_subscriber(aggregator)
    reporter = every_n_reporter.EveryNReporter(aggregator, n=3)

    result = root(torch.tensor([[1.0, 2.0], [3.0, 4.0]]))
    torch_testing.assert_close(result, torch.tensor([[5.0, 7.0], [9.0, 11.0]]))
    assert reporter.report(prefix='root/') == {}

    result = root(torch.tensor([[0, -2.0], [-3.0, -5.0]]))
    torch_testing.assert_close(result, torch.tensor([[3.0, -1.0], [-3.0, -7.0]]))
    assert reporter.report(prefix='root/') == {}

    result = root(torch.tensor([[0.0, 4.0], [0.0, 0.0]]))
    torch_testing.assert_close(result, torch.tensor([[3.0, 11.0], [3.0, 3.0]]))
    assert reporter.report(prefix='root/') == {
        'root/ModuleList.0.ChildModule.0.Multiply.0.multiplied': pytest.approx(5.0 / 3.0),
        'root/ModuleList.0.ChildModule.0.Add.1.added': pytest.approx(10.0 / 3.0),
        'root/ModuleList.0.ChildModule.0.y': pytest.approx(10.0 / 3.0),
        'root/ModuleList.0.ChildModule.0.z': pytest.approx(28.0 / 3.0),
        'root/ModuleList.0.ChildModule.1.Multiply.0.multiplied': pytest.approx(-1.0 / 3.0),
        'root/ModuleList.0.ChildModule.1.Add.1.added': pytest.approx(-2.0 / 3.0),
        'root/ModuleList.0.ChildModule.1.y': pytest.approx(-2.0 / 3.0),
        'root/ModuleList.0.ChildModule.1.z': pytest.approx(16.0 / 3.0),
    }
