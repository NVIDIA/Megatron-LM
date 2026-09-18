# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
import logging
import math
import re
import uuid
from typing import Any

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


class _Qwen3CoderToolParser:

    # Sentinel tokens for streaming mode
    tool_call_start_token: str = "<tool_call>"
    tool_call_end_token: str = "</tool_call>"
    tool_call_prefix: str = "<function="

    # Regex patterns
    tool_call_complete_regex = re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL)
    # Used only to locate boundaries (never to capture/backtrack over the body
    # of a call), so each of these stays a simple, non-catastrophic pattern.
    _tool_call_start_re = re.compile(r"<tool_call>")
    _tool_call_end_re = re.compile(r"</tool_call>")
    _parameter_start_re = re.compile(r"<\s*parameter\s*=")
    _parameter_end_re = re.compile(r"<\s*/\s*parameter\s*>")
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
        r"<\s*parameter\s*=\s*([^>]*)>" r"(.*?)" r"(?:<\s*/\s*parameter\s*>|(?=<\s*parameter\s*=))",
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
        # Only do this once generation is actually finished. Mid-stream,
        # `<function=get` is not "the model stopped after `get`" -- it is
        # "`get_weather` has not fully arrived in this chunk yet", and the
        # streaming caller emits a tool call's name delta exactly once, on the
        # first non-None name it sees. Firing here on a partial chunk would
        # permanently lock in the truncated name.
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

    def _find_tool_call_end(self, text: str, pos: int) -> int | None:
        """Find the `</tool_call>` that closes the call body starting at `pos`.

        A `</tool_call>`-looking substring inside a parameter's value must not
        end the call early -- vLLM's engine is a state machine that knows it is
        inside a parameter. Doing this with a single regex (trying, at every
        position, both "this looks like a parameter block" and "this is one
        plain character" as alternatives) is what caused catastrophic
        backtracking: with no `</tool_call>` to match, the engine tried
        exponentially many ways to partition the text between those two
        alternatives before giving up. This walks forward with bounded `re`
        searches instead -- `pos` only ever advances and each search is bounded
        to a small window (the next parameter, if any), so the total work
        across the whole scan stays linear in the length of `text` even when
        many parameters in a row never close.

        Per `tool_call_parameter_regex`'s grammar, a parameter is only
        recognised when closed by `</parameter>` or followed by another
        `<parameter=`; an unclosed parameter with nothing after it is simply
        dropped. Two cases follow from that:
          * A parameter that DOES eventually close absorbs everything in its
            value, including any `</tool_call>`-looking text in it, no matter
            how far away that close is.
          * A parameter that is instead abandoned for another `<parameter=`
            (never closed) does NOT retroactively absorb a `</tool_call>` that
            appeared in its dropped span -- that occurrence is the real
            terminator, since the parameter that would have "explained it away"
            never actually got recognised.

        Returns:
            The index of `</tool_call>`, or None if the call is not (yet) closed.
        """
        while True:
            param_match = self._parameter_start_re.search(text, pos)
            end_match = self._tool_call_end_re.search(text, pos)
            if end_match is None:
                return None
            if param_match is None or end_match.start() < param_match.start():
                return end_match.start()

            # A parameter starts before this `</tool_call>` candidate. Find
            # its own resolution -- another `<parameter=` bounds how far a
            # legitimate `</parameter>` close can still count.
            next_param_match = self._parameter_start_re.search(text, param_match.end())
            close_limit = next_param_match.start() if next_param_match else len(text)
            close_match = self._parameter_end_re.search(text, param_match.end(), close_limit)

            if close_match is not None:
                # Cleanly closed -- skip the whole block atomically. Any
                # `</tool_call>`-looking text inside it (including the
                # candidate found above, if it fell within this span) is part
                # of the value, not a real terminator.
                pos = close_match.end()
                continue

            # Abandoned (another `<parameter=` starts before this one's own
            # close) or never resolved. A `</tool_call>` found strictly within
            # this parameter's unresolved span -- from its own start up to
            # wherever it gets abandoned, or end of string -- is the real
            # terminator.
            scan_limit = next_param_match.start() if next_param_match else len(text)
            bounded_end_match = self._tool_call_end_re.search(text, param_match.start(), scan_limit)
            if bounded_end_match is not None:
                return bounded_end_match.start()
            if next_param_match is not None:
                pos = next_param_match.start()
                continue
            return None

    def _get_function_calls(self, model_output: str) -> list[str]:
        # Find all tool calls.
        raw_tool_calls = []
        pos = 0
        while True:
            start_match = self._tool_call_start_re.search(model_output, pos)
            if start_match is None:
                break
            body_start = start_match.end()
            end_index = self._find_tool_call_end(model_output, body_start)
            if end_index is None:
                # Unclosed -- salvage the rest of the string, same as the old
                # regex's second alternative, and stop (nothing legitimate can
                # follow an unterminated call).
                raw_tool_calls.append(model_output[body_start:])
                break
            raw_tool_calls.append(model_output[body_start:end_index])
            pos = end_index + len(self.tool_call_end_token)

        # Back-off strategy if no tool_call tags found
        if len(raw_tool_calls) == 0:
            raw_tool_calls = [model_output]

        raw_function_calls = []
        for tool_call in raw_tool_calls:
            raw_function_calls.extend(self.tool_call_function_regex.findall(tool_call))

        function_calls = [match[0] if match[0] else match[1] for match in raw_function_calls]
        return function_calls

    def extract_tool_calls(
        self, model_output: str, tools: list[ChatCompletionToolsParam] | None, finished: bool = True
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

    streaming_markers = ("<tool_call>", "<function=")

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
            text, tools=kwargs.get("tools", []), finished=kwargs.get("finished", True)
        )
        if information.get("tools_called", False):
            return information.get("content", ""), {"tool_calls": information.get("tool_calls", [])}
        else:
            return text, {}


class Qwen3CoderToolCombinedParser(Qwen3CoderToolParser):
    """Qwen3 Coder tool parser where <tool_call> also ends an open reasoning block.

    `qwen3-coder-tool` leaves a tool call emitted before `</think>` inside the
    reasoning text, like vLLM's standalone reasoning parsers paired with the
    qwen3_coder tool parser. This variant treats `<tool_call>` as an implicit
    end of reasoning and parses it, like vLLM's combined qwen3 parser.
    """

    implicit_reasoning_end_markers = ("<tool_call>",)
