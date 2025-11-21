import abc
from typing import final, override

from . import metrics


class MetricReporter(abc.ABC):
    """Reports the metrics collected by individual layers during a previous forward pass."""

    @abc.abstractmethod
    def report(self, *, prefix: str = '') -> metrics.Metrics:
        """Report metrics collected from the model.

        Args:
            prefix: A string prefix to add to all reported metric names.
        """
        pass


@final
class NoopMetricReporter(MetricReporter):
    """A metrics reporter that does nothing."""

    @override
    def report(self, *, prefix: str = '') -> metrics.Metrics:
        del prefix  # Unused
        return {}
