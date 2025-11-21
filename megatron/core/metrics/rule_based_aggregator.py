"""Implements a rule-based metric aggregator for collecting and reporting metrics per layer.

RuleBasedAggregator is forwarded the metrics collected from a root module and all of its children
during forward passes. It then uses user-specified rules, on a per-module-type basis, to determine
which metrics to collect from each module and how to aggregate them over multiple forward passes.
"""

import abc
import collections
from typing import Mapping, final, override

import torch

from megatron.core import process_groups_config

from . import collector, metrics, reporter


def default_suffix(module_name: str, child_index: int) -> str:
    """Returns the default suffix to append to metric names collected from a child module.

    Args:
        module_name: The name of the child module.
        child_index: The index of the child module among its siblings.
    """
    return f'{module_name}.{child_index}.'


class AggregationRule(abc.ABC):
    """A rule for determining how to collect metrics from a layer.

    Each AggregationRule subclass defines how to process raw metrics reported by a layer during
    its forward pass, as well as how to aggregate those metrics over one or more consecutive
    forward passes. `collect` may not use any distributed primitives, but `aggregate` may do so in
    order to compute global metrics across all instances of that particular layer.

    AggregationRule can also be used to customize the naming scheme for metrics collected from
    child modules of a given type by overriding the `suffix` method. If only this behavior is
    desired, then `collect` and `aggregate` may be implemented as returning empty dictionaries to
    skip metric tracking for those layers.
    """

    @classmethod
    @abc.abstractmethod
    def collect(cls, **params: torch.Tensor) -> dict[str, torch.Tensor]:
        """Collects raw metrics from a layer's forward pass and returns processed metrics.

        Args:
            **params: The raw metrics reported by the layer during its forward pass.

        Returns:
            A dictionary mapping metric names to their processed values, as calculated from the
            provided raw metrics.
        """

    @classmethod
    def suffix(cls, module_name: str, child_index: int) -> str:
        """Returns a suffix to append to metric names collected from a child module.

        This is used to override the default naming scheme if overridden.

        Args:
            module_name: The name of the child module.
            child_index: The index of the child module among its siblings.

        Returns:
            A string suffix to append to metric names.
        """
        return default_suffix(module_name, child_index)

    @classmethod
    @abc.abstractmethod
    def aggregate(
        cls,
        collected_metrics: list[dict[str, torch.Tensor]],
        pg_collection: process_groups_config.ProcessGroupCollection,
    ) -> metrics.Metrics:
        """Aggregates collected metrics over multiple steps and returns the final metrics to report.

        Aggregation is done on a per-instance basis - that is, all metrics in collected_metrics were
        from the same module instance over multiple forward passes.

        Args:
            collected_metrics: A list of dictionaries, each corresponding to the output of a call to
                collect() during a forward pass, in order.
            pg_collection: The process group collection used for distributed aggregation.

        Returns:
            A dictionary mapping metric names to their aggregated values.
        """


@final
class RuleBasedAggregator(collector.MetricCollector, reporter.MetricReporter):
    """Aggregates metrics per layer using rules to determine which metrics to collect and report."""

    def __init__(
        self,
        root_module: torch.nn.Module,
        rules: Mapping[type[torch.nn.Module], type[AggregationRule]],
        pg_collection: process_groups_config.ProcessGroupCollection,
    ):
        """Initializes the RuleBasedAggregator.

        Args:
            root_module: The root module from which metrics are being collected. All metrics
                forwarded to this aggregator should come from this module or its descendants.
            rules: A mapping from module types to aggregation rules that determine how to collect
                and aggregate metrics from modules of that type.
            pg_collection: The process group collection used for distributed aggregation.

        """
        self._root_module = root_module
        self._rules = dict(rules)
        self._pg_collection = pg_collection
        self._collected_metrics: dict[torch.nn.Module, list[dict[str, torch.Tensor]]] = (
            collections.defaultdict(list)
        )

    @override
    def collect(self, module: torch.nn.Module, **params: torch.Tensor) -> None:
        """Collects metrics from a module during a forward pass.

        If there is a rule for the module's type, uses it to process the reported metrics and
        stores the result for later aggregation. Otherwise, does nothing.

        Args:
            module: The module reporting the metrics.
            **params: The raw metrics reported by the module during its forward pass.
        """
        rule_type = self._rules.get(type(module))
        if rule_type is None:
            return
        with torch.no_grad():
            collected = rule_type.collect(**params)
        self._collected_metrics[module].append(collected)

    def _report_from(self, module: torch.nn.Module, prefix: str) -> metrics.Metrics:
        """Recursively reports metrics from a module and its children.

        Args:
            module: The current module from which to report metrics.
            prefix: The prefix to prepend to metric names collected from this module.
        """
        my_raw_metrics = self._collected_metrics.get(module)
        rule_type = self._rules.get(type(module))
        if my_raw_metrics:
            assert rule_type is not None
            gathered_metrics = {
                f'{prefix}{k}': v
                for k, v in rule_type.aggregate(my_raw_metrics, self._pg_collection).items()
            }
        else:
            gathered_metrics = {}
        suffix_rule = rule_type.suffix if rule_type is not None else default_suffix
        for i, module in enumerate(module.children()):
            gathered_metrics |= self._report_from(
                module, prefix + suffix_rule(type(module).__name__, i)
            )
        return gathered_metrics

    @override
    def report(self, *, prefix: str = '') -> metrics.Metrics:
        """Report metrics collected from the model.

        Each call to report() aggregates all metrics collected since the last call to report().
        Only metrics from modules which (1) are descendants of the root_module, and (2) have a rule
        registered for their type, are reported.

        Args:
            prefix: A string prefix to prepend to all reported metric names.

        Returns:
            A dictionary mapping metric names to their aggregated values.
        """
        with torch.no_grad():
            final_metrics = self._report_from(self._root_module, prefix)
        self._collected_metrics.clear()
        return final_metrics
