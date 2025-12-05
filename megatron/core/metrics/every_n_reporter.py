from typing import final, override

from . import metrics, reporter


@final
class EveryNReporter(reporter.MetricReporter):
    """Returns metrics every N calls to report()."""

    def __init__(self, inner_reporter: reporter.MetricReporter, *, n: int):
        """Initializes the EveryNReporter.

        Args:
            inner_reporter: The inner reporter to delegate to every N calls.
            n: The frequency of calls to report() at which to delegate to the inner reporter.
        """
        self._n = n
        self._call_count = 0
        self._inner_reporter = inner_reporter

    @override
    def report(self, *, prefix: str = '') -> metrics.Metrics:
        """Report metrics collected from the model."""
        self._call_count += 1
        if self._call_count % self._n == 0:
            self._call_count = 0
            return self._inner_reporter.report(prefix=prefix)
        else:
            return {}
