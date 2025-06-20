# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

"""
Provide base functionality for quantization purposes.

Usage comes from a user-provide YAML file, for example:

configs:
  nvfp4:
    $payload1
  mxfp8:
    $payload2

matchers:
  fc1:
    config: "nvfp4"
    type: "glob"
    pattern: "*fc1*"
    enabled: True
  fc2:
    config: "nvfp4"
    type: "glob"
    pattern: "*fc2*"
    enabled: True
  default:
    config: "mxfp8"
    type: "glob"
    pattern: "*"
    enabled: True

The user-passed configuration is split into 2 distinct pieces:
 * A set of quantization configs, describing *how* a given operator will be quantized
   Note: This is consumed by the operator(s), and the particular operators being instantiated
     are responsible for parsing this configuration if they support configurable quantization.
 * An ordered collection of matchers that determine what quantization config (if any) is
   applied to a given operator. The first matcher in the collection that successfully matches
   the context determines the key from the configs dict. If a matcher doesn't match, the rest
   of the matchers in the list are tested against.
   Matchers define a type, or style of matching - "glob" is bash-style, but this
   can be extended by inheriting from the abstract Matcher class to define a new match type.

The idea here is to provide an ability to define arbitrarily-complicated recipes in as
friendly a way as possible.
"""
import fnmatch
import logging
from abc import ABC, abstractmethod
from copy import deepcopy
from dataclasses import dataclass
from typing import Dict, List, Optional

import yaml

logger = logging.getLogger(__name__)


@dataclass
class MatchContext:
    """Layer context that can be matched to a quantization config."""

    module_path: str
    layer_number: Optional[int]


class QuantizationConfig:
    """Wrapper around configuration dictionary for layer's numerics."""

    def __init__(self, config: dict, match_input: MatchContext, config_key: str):
        """
        Initialize the quantization config.

        The configuration dictionary is copied to defend against modules that
        mutate the configuration corrupting the configuration of other modules.
        """
        self.config = deepcopy(config)
        self.match_input = match_input
        self.config_key = config_key

    def __repr__(self) -> str:
        return (
            f'{type(self).__name__}(config={self.config}, '
            f'match_input={self.match_input}, config_key={self.config_key})'
        )


class Matcher(ABC):
    """Matcher interface to select layers."""

    @abstractmethod
    def match(self, context: MatchContext) -> Optional[str]:
        """
        Match a layer based on its qualified name.

        If it does not match, return None. If it matches,
        return the configuration key to select for the layer.
        """
        return None


class GlobMatcher(Matcher):
    """Pattern based matcher using fnmatch to compare the module path against a pattern.
    fnmatch supplies glob-style matching similar to that used in bash, allowing for matches like:

    match_str="*fc2*" - match anything which includes "fc2" anywhere in the string.
    match_str="*fc2" - match anything which includes "fc2" at the end of the string.
    match_str="*layers.10*" - match anything with "layers.10" (layer #) in the string.
    """

    def __init__(self, pattern: str, config_key: str):
        self.pattern = pattern
        self.config_key = config_key

    def match(self, context: MatchContext) -> Optional[str]:
        """Pattern based match."""
        if fnmatch.fnmatch(context.module_path, self.pattern):
            return self.config_key
        return None

    def __repr__(self) -> str:
        return f'{type(self).__name__}(pattern={self.pattern}, config_key={self.config_key})'


class RecipeConfig:
    """Hold recipe information (matcher_fn) -> Configs)"""

    def __init__(self, matchers: List[Matcher], config_dict: Dict[str, Dict]):
        self.configs = config_dict
        self.matchers = matchers

    @staticmethod
    def _build_matchers(matchers_dict: Dict) -> List[Matcher]:
        # NOTE(slayton): We rely on order for matchers because it allows us to specify an
        # override ordering from the yaml structure. Process matchers in order of
        # definition, so we can have fallthrus.
        matchers: List[Matcher] = []

        for name, matcher in matchers_dict.items():
            enabled = matcher.get("enabled", False)

            if not enabled:
                continue

            match_type = matcher.get("type", None)
            assert match_type is not None, f"Matcher must specify a \"type\" field"

            if match_type == "glob":
                pattern = matcher.get("pattern", None)
                config = matcher.get("config", None)

                assert pattern is not None, f"GlobMatcher must specify \"pattern\" field"
                assert config is not None, f"GlobMatcher must specify \"config\" field"

                m = GlobMatcher(pattern, config)
            else:
                raise NotImplementedError(f"Match type '{match_type}' not implemented")

            matchers.append(m)

        return matchers

    @staticmethod
    def from_yaml_file(recipe_yaml_path: str) -> "RecipeConfig":
        """Loads recipe from yaml configuration."""

        with open(recipe_yaml_path, "r") as f:
            config = yaml.load(f, Loader=yaml.FullLoader)

        return RecipeConfig.from_config_dict(config)

    @staticmethod
    def from_config_dict(config: Dict) -> "RecipeConfig":
        """Loads recipe from dict configuration."""

        matchers_config = config.get("matchers", None)
        matchers = RecipeConfig._build_matchers(matchers_config)
        config_dict = config.get("configs", None)

        return RecipeConfig(matchers, config_dict)

    def match_to_config_key(self, operator_context: MatchContext) -> str | None:
        """
        Gives an operator's context, return a configuration key if
        necessary, or sentinel (None) denoting no matchers matched.
        """
        for matcher in self.matchers:
            config_key = matcher.match(operator_context)
            if config_key is not None:
                logger.info(
                    f'Context ({operator_context}) matched to quant config \"{config_key}\"'
                )
                return config_key
        logger.info(f'No config key match found for Context ({operator_context})')
        return None

    def match(self, operator_context: MatchContext) -> QuantizationConfig | None:
        """
        Gives an operator's context, return a QuantizationConfig if
        necessary, or sentinel (None) denoting no matchers matched.
        """
        config_key = self.match_to_config_key(operator_context)
        if config_key is not None:
            return QuantizationConfig(
                self.configs[config_key], match_input=operator_context, config_key=config_key
            )
        return None

    def __repr__(self) -> str:
        s = f'{type(self).__name__}(\n'
        for matcher in self.matchers:
            s += f'  matcher({repr(matcher)}\n'
        s += ')'
        return s
