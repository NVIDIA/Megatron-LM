import abc
from typing import final, override

import torch


class MetricCollector(abc.ABC):
    """Collects metrics from individual modules during a forward pass."""

    @abc.abstractmethod
    def collect(self, module: torch.nn.Module, **params: torch.Tensor) -> None:
        """Report metrics from a module.

        Args:
            module: The module reporting the metrics.
            **params: Arbitrary keyword arguments representing metric values.
        """
        pass


@final
class NoopMetricCollector(MetricCollector):
    """A metric collector that does nothing."""

    @override
    def collect(self, module: torch.nn.Module, **params: torch.Tensor) -> None:
        del module, params  # unused
