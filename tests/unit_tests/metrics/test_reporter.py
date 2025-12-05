from typing import cast, final, override

import pytest
import torch

from megatron.core.metrics import collector, every_n_reporter, metrics, reporter


@final
class SimpleReporter(collector.MetricCollector, reporter.MetricReporter):
    """Unit test collector/reporter that stores collected metrics in a list."""

    def __init__(self):
        self._collected: list[dict[str, float]] = []

    @override
    def collect(self, module: torch.nn.Module, **params: torch.Tensor) -> None:
        del module  # unused
        self._collected.append({k: cast(float, v.sum()) for k, v in params.items()})

    @override
    def report(self, *, prefix: str = '') -> metrics.Metrics:
        final_metrics: dict[str, float] = {}
        for sums in self._collected:
            for k, v in sums.items():
                k = f'{prefix}{k}'
                if k in final_metrics:
                    final_metrics[k] += v
                else:
                    final_metrics[k] = v
        self._collected.clear()
        return final_metrics


class DummyModule(torch.nn.Module):
    """A no-op module that collects a dummy metric during its forward pass."""

    def __init__(self, metric_collector: collector.MetricCollector):
        super().__init__()
        self._metric_collector = metric_collector

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """A dummy forward method that simply collects the input as a metric and returns it."""
        self._metric_collector.collect(self, dummy_metric=x)
        return x


def test_reporter_reports_empty_metrics():
    reporter_instance = SimpleReporter()

    assert (
        reporter_instance.report(prefix='model/') == {}
    ), "Reporter should report empty metrics when nothing has been collected."


def test_reporter_reports_metrics():
    reporter_instance = SimpleReporter()
    module = DummyModule(metric_collector=reporter_instance)

    input_tensor = torch.tensor([1.0, 2.0, 3.0])
    output_tensor = module(input_tensor)

    assert torch.equal(output_tensor, input_tensor), "Output tensor should match input tensor."

    assert reporter_instance.report(prefix='model/') == {
        "model/dummy_metric": pytest.approx(6.0)
    }, "Reporter should have reported the metric."


def test_reporter_clears_metrics():
    reporter_instance = SimpleReporter()
    module = DummyModule(metric_collector=reporter_instance)

    module(torch.tensor([1.0, 2.0, 3.0]))

    assert reporter_instance.report(
        prefix='model/'
    ), "Reporter should have reported metrics on the first call."
    assert (
        reporter_instance.report(prefix='model/') == {}
    ), "Reporter should have cleared its metrics."


def test_reporter_merges_metrics():
    reporter_instance = SimpleReporter()
    module = DummyModule(metric_collector=reporter_instance)

    module(torch.tensor([1.0, 2.0, 3.0]))
    module(torch.tensor([4.0, 5.0]))

    assert reporter_instance.report(prefix='model/') == {
        "model/dummy_metric": pytest.approx(15.0)
    }, "Reporter should have reported the metric."


def test_noop_reporter_does_nothing():
    reporter_instance = reporter.NoopMetricReporter()

    assert (
        reporter_instance.report(prefix='model/') == {}
    ), "NoopReporter should always report empty metrics."


def test_every_n_reporter_reports_every_n_calls():
    inner_reporter = SimpleReporter()
    module = DummyModule(metric_collector=inner_reporter)
    every_3_reporter = every_n_reporter.EveryNReporter(inner_reporter, n=3)

    module(torch.tensor([1.0]))
    assert (
        every_3_reporter.report(prefix='model/') == {}
    ), "EveryNReporter should not report on the 1st call."

    module(torch.tensor([2.0]))
    assert (
        every_3_reporter.report(prefix='model/') == {}
    ), "EveryNReporter should not report on the 2nd call."

    module(torch.tensor([3.0]))
    assert every_3_reporter.report(prefix='model/') == {
        "model/dummy_metric": pytest.approx(6.0)
    }, "EveryNReporter should report on the 3rd call."

    module(torch.tensor([4.0]))
    assert (
        every_3_reporter.report(prefix='model/') == {}
    ), "EveryNReporter should not report on the 4th call."

    module(torch.tensor([5.0]))
    assert (
        every_3_reporter.report(prefix='model/') == {}
    ), "EveryNReporter should not report on the 5th call."

    module(torch.tensor([6.0]))
    assert every_3_reporter.report(prefix='model/') == {
        "model/dummy_metric": pytest.approx(15.0)
    }, "EveryNReporter should report on the 6th call."
