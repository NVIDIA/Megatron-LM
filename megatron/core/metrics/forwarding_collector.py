from typing import TYPE_CHECKING, override

from . import collector

if TYPE_CHECKING:
    import torch


class ForwardingCollector(collector.MetricCollector):
    """A metric collector that forwards calls to zero or more subscribed collectors."""

    def __init__(self, *collectors: collector.MetricCollector):
        """Initializes the ForwardingCollector.

        Args:
            *collectors: The collectors to which to forward calls.
        """
        self._collectors = list(collectors)

    def add_subscriber(self, subscriber: collector.MetricCollector) -> None:
        """Subscribes a new collector to receive forwarded calls.

        Args:
            subscriber: The collector to subscribe.
        """
        self._collectors.append(subscriber)

    @override
    def collect(self, module: 'torch.nn.Module', **params: 'torch.Tensor') -> None:
        """Collects metrics from a module during a forward pass and publishes them."""
        for c in self._collectors:
            c.collect(module, **params)
