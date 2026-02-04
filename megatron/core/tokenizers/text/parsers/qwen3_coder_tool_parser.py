# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import ast
import json
import logging
import uuid
from types import SimpleNamespace
from typing import Any

import re

from megatron.core.tokenizers.text.parsers.base_parser import BaseParser

logger = logging.getLogger(__name__)

# These map to vLLM types but we just use dictionaries for now
ToolCall = dict[str, Any]
FunctionCall = dict[str, Any]
ChatCompletionToolsParam = dict[str, Any]
ChatCompletionRequest = dict[str, Any]
ExtractedToolCallInformation = dict


class _Qwen3CoderToolParser:

    # Sentinel tokens for streaming mode
    tool_call_start_token: str = "<tool_call>"
    tool_call_end_token: str = "</tool_call>"
    tool_call_prefix: str = "<function="

    # Regex patterns
    tool_call_complete_regex = re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL)
    tool_call_regex = re.compile(r"<tool_call>(.*?)</tool_call>|<tool_call>(.*?)$", re.DOTALL)
    tool_call_function_regex = re.compile(r"<function=(.*?)</function>|<function=(.*)$", re.DOTALL)
    tool_call_parameter_regex = re.compile(
        r"<parameter=(.*?)(?:</parameter>|(?=<parameter=)|(?=</function>)|$)", re.DOTALL
    )

    def _generate_tool_call_id(self) -> str:
        """Generate a unique tool call ID."""
        return f"call_{uuid.uuid4().hex[:24]}"

    def _get_arguments_config(
        self, func_name: str, tools: list[ChatCompletionToolsParam] | None
    ) -> dict:
        """Extract argument configuration for a function."""
        if tools is None:
            return {}
        for config in tools:
            config = SimpleNamespace(**config)  # Convert to SimpleNamespace for ease of access
            if not hasattr(config, "type") or not (
                hasattr(config, "function") and hasattr(config.function, "name")
            ):
                continue
            if config.type == "function" and config.function.name == func_name:
                if not hasattr(config.function, "parameters"):
                    return {}
                params = config.function.parameters
                if isinstance(params, dict) and "properties" in params:
                    return params["properties"]
                elif isinstance(params, dict):
                    return params
                else:
                    return {}
        logger.debug("Tool '%s' is not defined in the tools list.", func_name)
        return {}

    def _convert_param_value(
        self, param_value: str, param_name: str, param_config: dict, func_name: str
    ) -> Any:
        """Convert parameter value based on its type in the schema."""
        # Handle null value for any type
        if param_value.lower() == "null":
            return None

        if param_name not in param_config:
            if param_config != {}:
                logger.debug(
                    "Parsed parameter '%s' is not defined in the tool "
                    "parameters for tool '%s', directly returning the "
                    "string value.",
                    param_name,
                    func_name,
                )
            return param_value

        if isinstance(param_config[param_name], dict) and "type" in param_config[param_name]:
            param_type = str(param_config[param_name]["type"]).strip().lower()
        else:
            param_type = "string"
        if param_type in ["string", "str", "text", "varchar", "char", "enum"]:
            return param_value
        elif (
            param_type.startswith("int")
            or param_type.startswith("uint")
            or param_type.startswith("long")
            or param_type.startswith("short")
            or param_type.startswith("unsigned")
        ):
            try:
                return int(param_value)
            except (ValueError, TypeError):
                logger.debug(
                    "Parsed value '%s' of parameter '%s' is not an "
                    "integer in tool '%s', degenerating to string.",
                    param_value,
                    param_name,
                    func_name,
                )
                return param_value
        elif param_type.startswith("num") or param_type.startswith("float"):
            try:
                float_param_value = float(param_value)
                return (
                    float_param_value
                    if float_param_value - int(float_param_value) != 0
                    else int(float_param_value)
                )
            except (ValueError, TypeError):
                logger.debug(
                    "Parsed value '%s' of parameter '%s' is not a float "
                    "in tool '%s', degenerating to string.",
                    param_value,
                    param_name,
                    func_name,
                )
                return param_value
        elif param_type in ["boolean", "bool", "binary"]:
            param_value = param_value.lower()
            if param_value not in ["true", "false"]:
                logger.debug(
                    "Parsed value '%s' of parameter '%s' is not a boolean "
                    "(`true` or `false`) in tool '%s', degenerating to "
                    "false.",
                    param_value,
                    param_name,
                    func_name,
                )
            return param_value == "true"
        else:
            if (
                param_type in ["object", "array", "arr"]
                or param_type.startswith("dict")
                or param_type.startswith("list")
            ):
                try:
                    param_value = json.loads(param_value)
                    return param_value
                except (json.JSONDecodeError, TypeError, ValueError):
                    logger.debug(
                        "Parsed value '%s' of parameter '%s' cannot be "
                        "parsed with json.loads in tool '%s', will try "
                        "other methods to parse it.",
                        param_value,
                        param_name,
                        func_name,
                    )
            try:
                param_value = ast.literal_eval(param_value)  # safer
            except (ValueError, SyntaxError, TypeError):
                logger.debug(
                    "Parsed value '%s' of parameter '%s' cannot be "
                    "converted via Python `ast.literal_eval()` in tool "
                    "'%s', degenerating to string.",
                    param_value,
                    param_name,
                    func_name,
                )
            return param_value

    def _parse_xml_function_call(
        self, function_call_str: str, tools: list[ChatCompletionToolsParam] | None
    ) -> ToolCall | None:
        # Extract function name
        end_index = function_call_str.index(">")
        function_name = function_call_str[:end_index]
        param_config = self._get_arguments_config(function_name, tools)
        parameters = function_call_str[end_index + 1 :]
        param_dict = {}
        for match_text in self.tool_call_parameter_regex.findall(parameters):
            idx = match_text.index(">")
            param_name = match_text[:idx]
            param_value = str(match_text[idx + 1 :])
            # Remove prefix and trailing \n
            if param_value.startswith("\n"):
                param_value = param_value[1:]
            if param_value.endswith("\n"):
                param_value = param_value[:-1]

            param_dict[param_name] = self._convert_param_value(
                param_value, param_name, param_config, function_name
            )
        return ToolCall(
            type="function",
            id=self._generate_tool_call_id(),
            function=FunctionCall(
                name=function_name, arguments=json.dumps(param_dict, ensure_ascii=False)
            ),
        )

    def _get_function_calls(self, model_output: str) -> list[str]:
        # Find all tool calls
        matched_ranges = self.tool_call_regex.findall(model_output)
        raw_tool_calls = [match[0] if match[0] else match[1] for match in matched_ranges]

        # Back-off strategy if no tool_call tags found
        if len(raw_tool_calls) == 0:
            raw_tool_calls = [model_output]

        raw_function_calls = []
        for tool_call in raw_tool_calls:
            raw_function_calls.extend(self.tool_call_function_regex.findall(tool_call))

        function_calls = [match[0] if match[0] else match[1] for match in raw_function_calls]
        return function_calls

    def extract_tool_calls(
        self, model_output: str, tools: list[ChatCompletionToolsParam] | None
    ) -> ExtractedToolCallInformation:
        """Extracts the tool calls from the text using <tool_call>...</tool_call> tags."""
        # Quick check to avoid unnecessary processing
        if self.tool_call_prefix not in model_output:
            return ExtractedToolCallInformation(
                tools_called=False, tool_calls=[], content=model_output
            )

        try:
            function_calls = self._get_function_calls(model_output)
            if len(function_calls) == 0:
                return ExtractedToolCallInformation(
                    tools_called=False, tool_calls=[], content=model_output
                )

            tool_calls = [
                self._parse_xml_function_call(function_call_str, tools)
                for function_call_str in function_calls
            ]

            # Extract content before tool calls
            content_index = model_output.find(self.tool_call_start_token)
            idx = model_output.find(self.tool_call_prefix)
            content_index = content_index if content_index >= 0 else idx
            content = model_output[:content_index]  # .rstrip()

            return ExtractedToolCallInformation(
                tools_called=(len(tool_calls) > 0),
                tool_calls=tool_calls,
                content=content if content else None,
            )

        except Exception:
            logger.exception("Error in extracting tool call from response.")
            return ExtractedToolCallInformation(
                tools_called=False, tool_calls=[], content=model_output
            )


class Qwen3CoderToolParser(BaseParser):
    """Parser for Qwen3 Coder style tool calls."""

    @staticmethod
    def parse(text: str, **kwargs) -> tuple[str, dict[str, list[dict]]]:
        """
        Extracts the tool calls from the text using <tool_call>...</tool_call> tags.
        Uses the _Qwen3CoderToolParser class (copied from vLLM) to extract the tool calls.

        Args:
            text (str): The text to parse.

        Returns:
            tuple[str, dict[str, str]]: A tuple containing the unprocessed text
            and a dictionary with the extracted tool calls.
        """

        information = _Qwen3CoderToolParser().extract_tool_calls(
            text, tools=kwargs.get("tools", [])
        )
        if information.get("tools_called", False):
            return information.get("content", ""), {"tool_calls": information.get("tool_calls", [])}
        else:
            return text, {}
