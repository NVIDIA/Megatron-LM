# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
import logging
import math
import re
import uuid
from typing import Any, Optional

from megatron.core.tokenizers.text.parsers.base_parser import BaseParser

logger = logging.getLogger(__name__)

# Mirrors _TYPE_ALIASES in vllm/tool_parsers/utils.py.
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

# Priority order from coerce_to_schema_type; the first type that converts wins.
_TYPE_PRIORITY = ("null", "integer", "number", "boolean", "object", "array", "string")


def _is_json_finite(obj: Any) -> bool:
    """Whether a parsed JSON value is free of inf/nan.

    json.dumps renders those as `Infinity`/`NaN`, which is not valid JSON.
    """
    if isinstance(obj, float):
        return math.isfinite(obj)
    if isinstance(obj, dict):
        return all(_is_json_finite(v) for v in obj.values())
    if isinstance(obj, list):
        return all(_is_json_finite(v) for v in obj)
    return True


def _extract_types_from_schema(schema: Any) -> list[str]:
    """Collect every JSON Schema type a property may take.

    Port of extract_types_from_schema: handles `type` as a string or list,
    infers types from `enum` members, recurses through `anyOf`/`oneOf`/`allOf`,
    and falls back to ["string"] when nothing can be determined.
    """
    if schema is None or not isinstance(schema, dict):
        return ["string"]

    types: set[str] = set()

    type_value = schema.get("type")
    if isinstance(type_value, str):
        types.add(type_value)
    elif isinstance(type_value, list):
        types.update(t for t in type_value if isinstance(t, str))

    enum_values = schema.get("enum")
    if isinstance(enum_values, list) and enum_values:
        for value in enum_values:
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
        choices = schema.get(choice_field)
        if isinstance(choices, list):
            for choice in choices:
                types.update(_extract_types_from_schema(choice))

    return list(types) if types else ["string"]


def _coerce_to_schema_type(value: str, schema_type: str | list[str]) -> Any:
    """Best-effort coercion of a raw string to a JSON Schema type.

    Port of coerce_to_schema_type. Tries each declared type in priority order
    and returns the first that converts, then falls back to a plain JSON parse,
    then to the raw string. Note this means a "null" literal only becomes JSON
    null when the property actually admits null -- for a string-typed property
    it stays the four characters the model wrote.
    """
    if isinstance(schema_type, str):
        schema_type = [schema_type]

    normalized = {_TYPE_ALIASES.get(key, key) for key in (t.strip().lower() for t in schema_type)}

    for candidate in _TYPE_PRIORITY:
        if candidate not in normalized:
            continue

        if candidate == "null":
            if value.lower() == "null":
                return None
            continue
        if candidate == "string":
            return value
        if candidate == "integer":
            try:
                return int(value)
            except (ValueError, TypeError):
                continue
        if candidate == "number":
            try:
                parsed = float(value)
            except (ValueError, TypeError):
                continue
            if not math.isfinite(parsed):
                # int(float("inf")) raises and json.dumps(inf) is invalid JSON.
                continue
            return parsed if parsed != int(parsed) else int(parsed)
        if candidate == "boolean":
            lowered = value.lower().strip()
            if lowered in ("true", "1"):
                return True
            if lowered in ("false", "0"):
                return False
            continue
        if candidate in ("object", "array"):
            try:
                parsed = json.loads(value)
            except (json.JSONDecodeError, ValueError, TypeError):
                continue
            if _is_json_finite(parsed):
                return parsed
            continue

    try:
        parsed = json.loads(value)
    except (json.JSONDecodeError, ValueError, TypeError):
        return value
    return parsed if _is_json_finite(parsed) else value


# These map to vLLM types but we just use dictionaries for now
ToolCall = dict[str, Any]
FunctionCall = dict[str, Any]
ChatCompletionToolsParam = dict[str, Any]
ChatCompletionRequest = dict[str, Any]
ExtractedToolCallInformation = dict

_TOOL_CALL_START = "<tool_call>"
_TOOL_CALL_END = "</tool_call>"
_PARAM_OPEN_RE = re.compile(r"<\s*parameter\s*=[^>]*>")
_PARAM_CLOSE_RE = re.compile(r"<\s*/\s*parameter\s*>")


def _find_matching_tool_call_end(text: str, start: int) -> Optional[int]:
    """Find the index just past the `</tool_call>` matching the call that begins
    at `start` (the position right after its opening `<tool_call>` tag).

    A `<parameter=...>...</parameter>` block is treated as atomic, so a
    `</tool_call>` occurring inside a parameter value does not terminate the
    call early -- the same intent the regex this replaces had, but without its
    exponential worst case. That regex was `(?:PARAM_BLOCK|ANY_CHAR)*` wrapped
    in an outer repetition: when no `</tool_call>` ever arrives (truncated
    generation, or every streaming call before the terminator arrives), the
    engine tries every way to partition the input between the two alternatives
    before giving up -- confirmed locally to exceed 3s on `<function=f>` plus
    20 `<parameter=a>x</parameter>` blocks with no closing `</tool_call>`.

    This scans `text` once. The `i += 1` / atomic-skip loop below is O(n): `i`
    only ever moves forward. The one part that could reintroduce quadratic
    behavior -- searching for each parameter's closing tag -- is done with a
    single upfront `finditer` pass plus a pointer into it that also only moves
    forward, rather than a fresh search per parameter block.

    Returns None if `</tool_call>` never appears -- mirrors the second
    alternative of the old regex (`|<tool_call>(.*?)$`), which salvages a call
    whose closing tag never arrived.
    """
    closes = [m.start() for m in _PARAM_CLOSE_RE.finditer(text, start)]
    close_ptr = 0
    i = start
    n = len(text)
    while i < n:
        if text.startswith(_TOOL_CALL_END, i):
            return i + len(_TOOL_CALL_END)
        m = _PARAM_OPEN_RE.match(text, i)
        if m is not None:
            while close_ptr < len(closes) and closes[close_ptr] < m.end():
                close_ptr += 1
            if close_ptr < len(closes):
                close_match = _PARAM_CLOSE_RE.match(text, closes[close_ptr])
                i = close_match.end()
                close_ptr += 1
                continue
        i += 1
    return None


def _extract_tool_call_bodies(model_output: str) -> list[str]:
    """Linear-time replacement for the old `tool_call_regex.findall(model_output)`.

    Returns the text between each `<tool_call>` and its matching `</tool_call>`
    (or, for a call with no closing tag, everything to the end of the string --
    same fallback the old regex's second alternative provided).
    """
    bodies = []
    pos = 0
    while True:
        start = model_output.find(_TOOL_CALL_START, pos)
        if start == -1:
            break
        body_start = start + len(_TOOL_CALL_START)
        end = _find_matching_tool_call_end(model_output, body_start)
        if end is None:
            bodies.append(model_output[body_start:])
            break
        bodies.append(model_output[body_start : end - len(_TOOL_CALL_END)])
        pos = end
    return bodies


class _Qwen3CoderToolParser:

    # Sentinel tokens for streaming mode
    tool_call_start_token: str = "<tool_call>"
    tool_call_end_token: str = "</tool_call>"
    tool_call_prefix: str = "<function="

    # Regex patterns
    tool_call_complete_regex = re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL)
    # A `</tool_call>` occurring INSIDE a parameter value must not terminate the
    # call. vLLM's engine is a state machine that knows it is inside a parameter;
    # the equivalent here is to consume complete `<parameter=...>...</parameter>`
    # blocks atomically, so any `</tool_call>` within one is absorbed. This used
    # to be a single `(?:PARAM_BLOCK|ANY_CHAR)*` regex, which was exponential on
    # an unterminated call with many parameter blocks (no closing `</tool_call>`
    # -- the case a truncated generation or a mid-stream chunk always is until
    # the closing tag arrives). `_extract_tool_call_bodies` / `_find_matching_
    # tool_call_end` above replace it with a linear scan carrying the identical
    # semantics, including the "salvage everything to end of string" fallback
    # for a call whose `</tool_call>` never arrived.
    tool_call_function_regex = re.compile(r"<function=(.*?)</function>|<function=(.*)$", re.DOTALL)
    # Mirrors vLLM's `_PARAM_RE` in `vllm/parser/qwen3.py`. Three properties are
    # load-bearing and each was previously wrong:
    #   * the tag tolerates whitespace -- `< parameter = a >`, `</ parameter >`
    #   * the name is captured separately, so `\s*` after `=` eats leading
    #     whitespace while `[^>]*` keeps trailing whitespace (vLLM does not
    #     strip the name)
    #   * a parameter is recognised ONLY when closed by `</parameter>` or
    #     followed by another `<parameter=`. vLLM has no `</function>` or
    #     end-of-string terminator, so an unclosed parameter is dropped rather
    #     than salvaged.
    tool_call_parameter_regex = re.compile(
        r"<\s*parameter\s*=\s*([^>]*)>"
        r"(.*?)"
        r"(?:<\s*/\s*parameter\s*>|(?=<\s*parameter\s*=))",
        re.DOTALL,
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
            if not isinstance(config, dict):
                continue
            fn = config.get("function", {})
            if not isinstance(fn, dict):
                continue
            if config.get("type") != "function" or fn.get("name") != func_name:
                continue
            params = fn.get("parameters", {})
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
        """Convert a parameter value using its declared schema type."""
        schema = param_config.get(param_name) if isinstance(param_config, dict) else None
        if schema is None and param_config:
            logger.debug(
                "Parsed parameter '%s' is not defined in the tool parameters "
                "for tool '%s', treating it as a string.",
                param_name,
                func_name,
            )
        return _coerce_to_schema_type(param_value, _extract_types_from_schema(schema))

    def _parse_xml_function_call(
        self,
        function_call_str: str,
        tools: list[ChatCompletionToolsParam] | None,
        finished: bool = True,
    ) -> ToolCall | None:
        # Extract function name. When the opening tag was never closed -- e.g.
        # generation stopped mid-name, leaving `<function=f` -- vLLM still emits
        # a call named after the remaining text. Returning None here instead
        # left `<tool_call>` in the post-parse content, which NeMo-Gym reads as
        # `is_invalid_tool_call` and turns into a -5.0 advantage, so a merely
        # truncated call was being punished as a malformed one.
        #
        # Gated on `finished`: a streaming caller re-parses the whole
        # accumulated text on every chunk, so `<function=get` mid-stream is not
        # generation having stopped -- it is `<function=get_weather` that just
        # has not fully arrived yet. StreamingChatParser emits a tool call's
        # name delta exactly once, the first time it sees a non-None name, so
        # firing this fallback mid-stream would permanently lock in the
        # truncated prefix. Only treat a missing `>` as truly truncated once
        # the caller confirms the response itself is finished.
        end_index = function_call_str.find(">")
        if end_index == -1:
            if not finished:
                return None
            function_name = function_call_str.strip()
            if not function_name:
                return None
            return ToolCall(
                type="function",
                id=self._generate_tool_call_id(),
                function=FunctionCall(name=function_name, arguments="{}"),
            )
        function_name = function_call_str[:end_index]
        param_config = self._get_arguments_config(function_name, tools)
        parameters = function_call_str[end_index + 1 :]
        param_dict = {}
        for param_name, param_value in self.tool_call_parameter_regex.findall(parameters):
            # vLLM's _qwen3_arg_converter strips the value before coercion, so
            # whitespace around a value never reaches the tool. The NAME is not
            # stripped: `\s*` in the pattern already removed anything leading,
            # and vLLM preserves trailing whitespace inside `[^>]*`.
            param_value = str(param_value).strip()

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
        raw_tool_calls = _extract_tool_call_bodies(model_output)

        # Back-off strategy if no tool_call tags found
        if len(raw_tool_calls) == 0:
            raw_tool_calls = [model_output]

        raw_function_calls = []
        for tool_call in raw_tool_calls:
            raw_function_calls.extend(self.tool_call_function_regex.findall(tool_call))

        function_calls = [match[0] if match[0] else match[1] for match in raw_function_calls]
        return function_calls

    def extract_tool_calls(
        self,
        model_output: str,
        tools: list[ChatCompletionToolsParam] | None,
        finished: bool = True,
    ) -> ExtractedToolCallInformation:
        """Extracts the tool calls from the text using <tool_call>...</tool_call> tags.

        `finished` gates the truncated-function-name fallback in
        `_parse_xml_function_call`: see that method's docstring.
        """
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
                self._parse_xml_function_call(function_call_str, tools, finished=finished)
                for function_call_str in function_calls
            ]
            tool_calls = [tc for tc in tool_calls if tc is not None]

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

    implicit_reasoning_end_markers = ("<tool_call>",)
    streaming_markers = ("<tool_call>", "<function=")

    @staticmethod
    def parse(text: str, **kwargs) -> tuple[str, dict[str, list[dict]]]:
        """
        Extracts the tool calls from the text using <tool_call>...</tool_call> tags.
        Uses the _Qwen3CoderToolParser class (copied from vLLM) to extract the tool calls.

        Args:
            text (str): The text to parse.
            finished (bool): Whether `text` is the complete, final response rather
                than a partial chunk being re-parsed mid-stream. Defaults to True,
                since most callers hand this a finished response; the streaming
                path is the one exception and passes this explicitly on every
                call. Gates whether a truncated `<function=name` with no closing
                `>` is treated as "generation actually stopped here" (finished)
                versus "the rest just has not arrived in this chunk yet" (not
                finished) -- see `_Qwen3CoderToolParser._parse_xml_function_call`.

        Returns:
            tuple[str, dict[str, str]]: A tuple containing the unprocessed text
            and a dictionary with the extracted tool calls.
        """

        information = _Qwen3CoderToolParser().extract_tool_calls(
            text, tools=kwargs.get("tools", []), finished=kwargs.get("finished", True)
        )
        if information.get("tools_called", False):
            return information.get("content", ""), {"tool_calls": information.get("tool_calls", [])}
        else:
            return text, {}
