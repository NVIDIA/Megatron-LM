# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Qwen3-Coder XML tool-call parser.

The model emits tool calls as::

    <tool_call>
    <function=NAME>
    <parameter=KEY>VALUE</parameter>
    </function>
    </tool_call>

Parsing here mirrors vLLM's parser engine so that a model served through the
Megatron dynamic text-generation server yields the same parsed tool calls as
the same model served through vLLM (vLLM 0.25 routes ``qwen3_coder`` to
``Qwen3EngineToolParser``). RL pipelines that mix the two engines rely on this
parity because tool-use rewards are computed from the parsed arguments.

Three behaviours are taken from vLLM:

- Parameter values are ``str.strip()``-ed (``vllm/parser/qwen3.py``,
  ``_qwen3_arg_converter``). The previous implementation removed at most one
  leading and one trailing newline, so multi-line code arguments kept their
  indentation while vLLM dropped it, and ``<parameter=x>  v  </parameter>``
  parsed differently in the two servers.
- Content preceding the tool calls is ``str.strip()``-ed and an empty result
  becomes ``None`` (``ParserEngine._strip_content_whitespace`` with
  ``strip_content_whitespace_with_tools=True``).
- Schema-based type coercion follows ``vllm/tool_parsers/utils.py``
  (``extract_types_from_schema`` + ``coerce_to_schema_type``): candidate types
  are tried in the order null > integer > number > boolean > object > array >
  string, ``"1"``/``"0"`` are accepted as booleans, a value that matches none of
  the schema types falls back to ``json.loads`` and otherwise stays a string.
  The previous implementation degraded unparseable booleans to ``False``,
  accepted Python literals for objects via ``ast.literal_eval`` and returned
  ``None`` for ``"null"`` regardless of the schema.
"""

import json
import logging
import math
import re
import uuid
from typing import Any

from megatron.core.tokenizers.text.parsers.base_parser import BaseParser

logger = logging.getLogger(__name__)

# These map to vLLM types but we just use dictionaries for now
ToolCall = dict[str, Any]
FunctionCall = dict[str, Any]
ChatCompletionToolsParam = dict[str, Any]
ChatCompletionRequest = dict[str, Any]
ExtractedToolCallInformation = dict

# Mirrors vllm/tool_parsers/utils.py::_TYPE_ALIASES.
_TYPE_ALIASES: dict[str, str] = {
    "str": "string",
    "text": "string",
    "varchar": "string",
    "char": "string",
    "enum": "string",
    "int": "integer",
    "int32": "integer",
    "int64": "integer",
    "uint": "integer",
    "uint32": "integer",
    "uint64": "integer",
    "long": "integer",
    "short": "integer",
    "unsigned": "integer",
    "float": "number",
    "float32": "number",
    "float64": "number",
    "double": "number",
    "bool": "boolean",
    "dict": "object",
    "arr": "array",
    "list": "array",
    "sequence": "array",
}

# Priority in which candidate schema types are tried during coercion.
_TYPE_PRIORITY = ("null", "integer", "number", "boolean", "object", "array", "string")


def _extract_types_from_schema(schema: Any) -> list[str]:
    """Extract all possible type strings from a JSON Schema definition.

    Handles ``type`` (string or list), ``enum`` value inference, and recursive
    ``anyOf``/``oneOf``/``allOf``. Returns ``["string"]`` when no type
    information can be determined. Mirrors
    ``vllm/tool_parsers/utils.py::extract_types_from_schema``.
    """
    if schema is None or not isinstance(schema, dict):
        return ["string"]

    types: set[str] = set()

    if "type" in schema:
        type_value = schema["type"]
        if isinstance(type_value, str):
            types.add(type_value)
        elif isinstance(type_value, list):
            for t in type_value:
                if isinstance(t, str):
                    types.add(t)

    if "enum" in schema and isinstance(schema["enum"], list) and schema["enum"]:
        for value in schema["enum"]:
            if value is None:
                types.add("null")
            elif isinstance(value, bool):
                types.add("boolean")
            elif isinstance(value, int):
                types.add("integer")
            elif isinstance(value, float):
                types.add("number")
            elif isinstance(value, str):
                types.add("string")
            elif isinstance(value, list):
                types.add("array")
            elif isinstance(value, dict):
                types.add("object")

    for choice_field in ("anyOf", "oneOf", "allOf"):
        if choice_field in schema and isinstance(schema[choice_field], list):
            for choice in schema[choice_field]:
                types.update(_extract_types_from_schema(choice))

    return list(types) if types else ["string"]


def _is_json_finite(obj: Any) -> bool:
    """Whether ``obj`` serializes to valid JSON (no ``inf``/``nan`` anywhere inside)."""
    try:
        json.dumps(obj, allow_nan=False)
        return True
    except (ValueError, TypeError):
        return False


def _coerce_to_schema_type(value: str, schema_type: str | list[str]) -> Any:
    """Best-effort coercion of a raw string value to a JSON Schema type.

    Tries each type in priority order (null > integer > number > boolean >
    object > array > string) and returns the first successful coercion. When
    no schema type matches, the value is parsed with ``json.loads`` and falls
    back to the original string. Mirrors
    ``vllm/tool_parsers/utils.py::coerce_to_schema_type``.
    """
    if isinstance(schema_type, str):
        schema_type = [schema_type]

    normalized_types = {
        _TYPE_ALIASES.get(key, key) for t in schema_type for key in [t.strip().lower()]
    }

    for candidate_type in _TYPE_PRIORITY:
        if candidate_type not in normalized_types:
            continue

        if candidate_type == "null":
            if value.lower() == "null":
                return None
            continue
        if candidate_type == "string":
            return value
        if candidate_type == "integer":
            try:
                return int(value)
            except (ValueError, TypeError):
                continue
        if candidate_type == "number":
            try:
                val = float(value)
            except (ValueError, TypeError):
                continue
            if not math.isfinite(val):
                # inf/-inf/nan are not valid JSON numbers; keep the raw string.
                continue
            return val if val != int(val) else int(val)
        if candidate_type == "boolean":
            lower_val = value.lower().strip()
            if lower_val in ("true", "1"):
                return True
            if lower_val in ("false", "0"):
                return False
            continue
        if candidate_type in ("object", "array"):
            try:
                parsed = json.loads(value)
            except (json.JSONDecodeError, ValueError, TypeError):
                continue
            if _is_json_finite(parsed):
                return parsed
            continue

    try:
        parsed = json.loads(value)
    except (json.JSONDecodeError, ValueError):
        return value
    if not _is_json_finite(parsed):
        return value
    return parsed


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
        """Return the ``properties`` schema of ``func_name``, or ``{}``.

        Mirrors ``vllm/tool_parsers/utils.py::find_tool_properties``: only the
        ``parameters.properties`` mapping drives coercion; a tool without it, or a
        function that is not in ``tools``, leaves every argument a string.
        """
        if not tools:
            return {}
        for config in tools:
            if not isinstance(config, dict):
                continue
            fn = config.get("function", {})
            if not isinstance(fn, dict):
                continue
            if config.get("type") != "function" or fn.get("name") != func_name:
                continue
            params = fn.get("parameters") or {}
            if not isinstance(params, dict):
                return {}
            properties = params.get("properties", {})
            return properties if isinstance(properties, dict) else {}
        logger.debug("Tool '%s' is not defined in the tools list.", func_name)
        return {}

    def _convert_param_value(
        self, param_value: str, param_name: str, param_config: dict, func_name: str
    ) -> Any:
        """Convert a parameter value according to its JSON Schema, like vLLM does.

        Parameters absent from the schema (or whose schema is not a mapping) are
        returned unchanged as strings; everything else goes through
        ``_coerce_to_schema_type``.
        """
        schema = param_config.get(param_name) if param_config else None
        if not isinstance(schema, dict):
            if param_config:
                logger.debug(
                    "Parsed parameter '%s' is not defined in the tool "
                    "parameters for tool '%s', directly returning the "
                    "string value.",
                    param_name,
                    func_name,
                )
            return param_value
        return _coerce_to_schema_type(param_value, _extract_types_from_schema(schema))

    def _parse_xml_function_call(
        self, function_call_str: str, tools: list[ChatCompletionToolsParam] | None
    ) -> ToolCall | None:
        # Extract function name
        end_index = function_call_str.find(">")
        if end_index == -1:
            return None
        function_name = function_call_str[:end_index]
        param_config = self._get_arguments_config(function_name, tools)
        parameters = function_call_str[end_index + 1 :]
        param_dict = {}
        for match_text in self.tool_call_parameter_regex.findall(parameters):
            idx = match_text.find(">")
            # Malformed parameter block with no name/value delimiter, e.g. truncated tool call.
            if idx == -1:
                continue
            param_name = match_text[:idx]
            # vLLM strips all surrounding whitespace from the value (the model wraps
            # values in newlines), not just a single leading/trailing newline.
            param_value = str(match_text[idx + 1 :]).strip()

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
            tool_calls = [tc for tc in tool_calls if tc is not None]

            # Extract content before tool calls
            content_index = model_output.find(self.tool_call_start_token)
            idx = model_output.find(self.tool_call_prefix)
            content_index = content_index if content_index >= 0 else idx
            # vLLM strips the content around tool calls
            # (ParserEngineConfig.strip_content_whitespace_with_tools=True).
            content = model_output[:content_index].strip()

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

    implicit_reasoning_end_markers = ("<tool_call>",)
    streaming_markers = ("<tool_call>", "<function=")

    @staticmethod
    def parse(text: str, **kwargs) -> tuple[str, dict[str, list[dict]]]:
        """
        Extracts the tool calls from the text using <tool_call>...</tool_call> tags.
        Uses the _Qwen3CoderToolParser class (behaviourally aligned with vLLM's
        Qwen3 parser engine) to extract the tool calls.

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
