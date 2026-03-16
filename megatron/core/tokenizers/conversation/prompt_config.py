# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

from dataclasses import dataclass, field
from typing import Dict, Optional

# ---------------------------------------------------------------------------
# Library-agnostic helpers for resolving token attributes
# ---------------------------------------------------------------------------


def _token_to_id(tokenizer, token: str) -> int:
    """Resolve a token string to its ID, working with any tokenizer library.

    All text library tokenizers have ``token_to_id`` (concrete default on the
    abstract base class), so that is the primary branch.  The
    ``convert_tokens_to_ids`` fallback covers the multimodal path, which passes
    a raw HuggingFace ``AutoTokenizer`` directly.
    """
    if hasattr(tokenizer, 'token_to_id'):
        return tokenizer.token_to_id(token)
    if hasattr(tokenizer, 'convert_tokens_to_ids'):
        return tokenizer.convert_tokens_to_ids(token)
    raise TypeError(f"Cannot resolve token to ID for {type(tokenizer)}")


def _get_pad_token_id(tokenizer) -> Optional[int]:
    return getattr(tokenizer, 'pad_token_id', None) or getattr(tokenizer, 'pad_id', None)


def _get_bos_token_id(tokenizer) -> Optional[int]:
    return getattr(tokenizer, 'bos_token_id', None) or getattr(tokenizer, 'bos_id', None)


def _get_eos_token_id(tokenizer) -> Optional[int]:
    return getattr(tokenizer, 'eos_token_id', None) or getattr(tokenizer, 'eos_id', None)


def _get_unk_token_id(tokenizer) -> Optional[int]:
    return getattr(tokenizer, 'unk_token_id', None) or getattr(tokenizer, 'unk_id', None)


def _get_chat_template(tokenizer) -> Optional[str]:
    return getattr(tokenizer, 'chat_template', None)


@dataclass
class PromptConfig:
    """Config options for different prompt formats.

    Controls how conversations are tokenized and how target masking is applied
    for supervised fine-tuning (SFT) and multimodal training.
    """

    # How many tokens are used for the assistant prefix, e.g. "<|im_start|>assistant\n".
    # Used for masking the assistant prefix.
    assistant_prefix_len: int
    # Padding token ID.
    pad_token_id: int
    # For overriding the default chat format template.
    custom_chat_template: Optional[str]
    # If the tokenizer inserts BOS token by default.
    has_bos: bool
    # If the tokenizer supports a separate role for system messages.
    has_system_role: bool
    # Whether to force a specific system message.
    force_system_message: bool = False
    system_default: Optional[dict] = None
    # Whether to validate that IMAGE_TOKEN is not in assistant content.
    validate_no_image_in_assistant: bool = False
    # Whether to capitalize role names (e.g. for nemotron5-aligned format).
    capitalize_roles: bool = False
    # Whether to skip target masking entirely (e.g. for SFT "default" format).
    skip_masking: bool = False
    # Whether to include "tool" role in the set of masked (non-training) roles.
    allow_tool_role: bool = False


# ---------------------------------------------------------------------------
# Chat template strings
# ---------------------------------------------------------------------------

# SFT templates
# fmt: off
nemotron_h_aligned_custom_template = """{% for message in messages %}{% if message['role'] == 'system' %}{{ '<SPECIAL_10>System\n' + message['content'].strip() + '\n' }}{% elif message['role'] == 'user' %}{{ '<SPECIAL_11>User\n' + message['content'].strip() + '\n' + '<SPECIAL_11>Assistant\n' }}{% elif message['role'] == 'assistant' %}{{ message['content'].strip() + '\n' }}{% endif %}{% endfor %}"""  # pylint: disable=line-too-long
nemotron_nano_v2_custom_template = """{% for message in messages %}{% set content = message['content'] %}{% if message['role'] == 'system' %}{{ '<SPECIAL_10>System\n' + content.replace('/think', '').replace('/no_think', '').strip() + '\n' }}{% elif message['role'] == 'user' %}{{ '<SPECIAL_11>User\n' + content.replace('/think', '').replace('/no_think', '').strip() + '\n' }}{% elif message['role'] == 'assistant' %}{{ '<SPECIAL_11>Assistant\n' + content.strip() + '\n<SPECIAL_12>\n' }}{% endif %}{% endfor %}"""  # pylint: disable=line-too-long
identity_template = """{% for message in messages %}{{ message['content'] }}{% endfor %}"""
# fmt: on

# Multimodal templates
# The default mistral template raises exceptions so we use a custom one.
mistral_custom_template = """
{{- bos_token }}
{%- for message in messages %}
    {%- if message['role'] == 'user' %}
        {{- '[INST] ' + message['content'] + '[/INST]' }}
    {%- elif message['role'] == 'assistant' %}
        {{- ' ' + message['content'] + eos_token}}
    {%- endif %}
{%- endfor %}
{% if add_generation_prompt %}{{ ' ' }}{% endif %}
"""

nvlm_yi_34b_template = "{{- bos_token }}{% for message in messages %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"  # pylint: disable=line-too-long

qwen2p0_custom_template = "{% for message in messages %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"  # pylint: disable=line-too-long

# Note: this is the same template as
# https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct/blob/main/tokenizer_config.json#L2053
# but we removed the forced system message.
llama3p1_chat_template = """{{- bos_token }}\n{%- if custom_tools is defined %}\n    {%- set tools = custom_tools %}\n{%- endif %}\n{%- if not tools_in_user_message is defined %}\n    {%- set tools_in_user_message = true %}\n{%- endif %}\n{%- if not date_string is defined %}\n    {%- set date_string = \"26 Jul 2024\" %}\n{%- endif %}\n{%- if not tools is defined %}\n    {%- set tools = none %}\n{%- endif %}\n\n{#- This block extracts the system message, so we can slot it into the right place. #}\n{%- if messages[0]['role'] == 'system' %}\n    {%- set system_message = messages[0]['content']|trim %}\n    {%- set messages = messages[1:] %}\n{%- else %}\n    {%- set system_message = none %}\n{%- endif %}\n\n{%- if system_message is not none %}{#- System message + builtin tools #}\n{{- \"<|start_header_id|>system<|end_header_id|>\\n\\n\" }}\n{%- if builtin_tools is defined or tools is not none %}\n    {{- \"Environment: ipython\\n\" }}\n{%- endif %}\n{%- if builtin_tools is defined %}\n    {{- \"Tools: \" + builtin_tools | reject('equalto', 'code_interpreter') | join(\", \") + \"\\n\\n\"}}\n{%- endif %}{%- if tools is not none and not tools_in_user_message %}\n    {{- \"You have access to the following functions. To call a function, please respond with JSON for a function call.\" }}\n    {{- 'Respond in the format {\"name\": function name, \"parameters\": dictionary of argument name and its value}.' }}\n    {{- \"Do not use variables.\\n\\n\" }}\n    {%- for t in tools %}\n        {{- t | tojson(indent=4) }}\n        {{- \"\\n\\n\" }}\n    {%- endfor %}\n{%- endif %}\n{{- system_message }}\n{{- \"<|eot_id|>\" }}\n\n{%-endif %}{#- Custom tools are passed in a user message with some extra guidance #}\n{%- if tools_in_user_message and not tools is none %}\n    {#- Extract the first user message so we can plug it in here #}\n    {%- if messages | length != 0 %}\n        {%- set first_user_message = messages[0]['content']|trim %}\n        {%- set messages = messages[1:] %}\n    {%- else %}\n        {{- raise_exception(\"Cannot put tools in the first user message when there's no first user message!\") }}\n{%- endif %}\n    {{- '<|start_header_id|>user<|end_header_id|>\\n\\n' -}}\n    {{- \"Given the following functions, please respond with a JSON for a function call \" }}\n    {{- \"with its proper arguments that best answers the given prompt.\\n\\n\" }}\n    {{- 'Respond in the format {\"name\": function name, \"parameters\": dictionary of argument name and its value}.' }}\n    {{- \"Do not use variables.\\n\\n\" }}\n    {%- for t in tools %}\n        {{- t | tojson(indent=4) }}\n        {{- \"\\n\\n\" }}\n    {%- endfor %}\n    {{- first_user_message + \"<|eot_id|>\"}}\n{%- endif %}\n\n{%- for message in messages %}\n    {%- if not (message.role == 'ipython' or message.role == 'tool' or 'tool_calls' in message) %}\n        {{- '<|start_header_id|>' + message['role'] + '<|end_header_id|>\\n\\n'+ message['content'] | trim + '<|eot_id|>' }}\n    {%- elif 'tool_calls' in message %}\n        {%- if not message.tool_calls|length == 1 %}\n            {{- raise_exception(\"This model only supports single tool-calls at once!\") }}\n        {%- endif %}\n        {%- set tool_call = message.tool_calls[0].function %}\n        {%- if builtin_tools is defined and tool_call.name in builtin_tools %}\n            {{- '<|start_header_id|>assistant<|end_header_id|>\\n\\n' -}}\n            {{- \"<|python_tag|>\" + tool_call.name + \".call(\" }}\n            {%- for arg_name, arg_val in tool_call.arguments | items %}\n                {{- arg_name + '=\"' + arg_val + '\"' }}\n                {%- if not loop.last %}\n                    {{- \", \" }}\n                {%- endif %}\n                {%- endfor %}\n            {{- \")\" }}\n        {%- else  %}\n            {{- '<|start_header_id|>assistant<|end_header_id|>\\n\\n' -}}\n            {{- '{\"name\": \"' + tool_call.name + '\", ' }}\n            {{- '\"parameters\": ' }}\n            {{- tool_call.arguments | tojson }}\n            {{- \"}\" }}\n        {%- endif %}\n        {%- if builtin_tools is defined %}\n            {#- This means we're in ipython mode #}\n            {{- \"<|eom_id|>\" }}\n        {%- else %}\n            {{- \"<|eot_id|>\" }}\n        {%- endif %}\n    {%- elif message.role == \"tool\" or message.role == \"ipython\" %}\n        {{- \"<|start_header_id|>ipython<|end_header_id|>\\n\\n\" }}\n        {%- if message.content is mapping or message.content is iterable %}\n            {{- message.content | tojson }}\n        {%- else %}\n            {{- message.content }}\n        {%- endif %}\n        {{- \"<|eot_id|>\" }}\n    {%- endif %}\n{%- endfor %}\n{%- if add_generation_prompt %}\n    {{- '<|start_header_id|>assistant<|end_header_id|>\\n\\n' }}\n{%- endif %}\n"""  # pylint: disable=line-too-long

nemotron_custom_template = "{{- bos_token }}{% for message in messages %}{{'<SPECIAL_14>' + message['role'] + '\n' + message['content'] + '<SPECIAL_15>' + '\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<SPECIAL_14>assistant\n' }}{% endif %}"  # pylint: disable=line-too-long

nemotron_aligned_custom_template = "{{- bos_token}}{% for message in messages %}{{message['role'] + '\n' + message['content'] + '\n' + '[PREFIX]'}}{% endfor %}{% if add_generation_prompt %}{{ 'Assistant\n' }}{% endif %}"  # pylint: disable=line-too-long


# ---------------------------------------------------------------------------
# Prompt format registry
# ---------------------------------------------------------------------------


def _build_sft_nemotron_nano_v2(tokenizer):
    return PromptConfig(
        assistant_prefix_len=3,
        pad_token_id=_token_to_id(tokenizer, "<unk>"),
        custom_chat_template=nemotron_nano_v2_custom_template,
        has_bos=False,
        has_system_role=True,
        allow_tool_role=True,
    )


def _build_sft_nemotron_h_aligned(tokenizer):
    return PromptConfig(
        assistant_prefix_len=0,
        pad_token_id=_token_to_id(tokenizer, "<SPECIAL_233>"),
        custom_chat_template=nemotron_h_aligned_custom_template,
        has_bos=False,
        has_system_role=True,
        allow_tool_role=True,
    )


def _build_sft_identity(tokenizer):
    return PromptConfig(
        assistant_prefix_len=0,
        pad_token_id=_token_to_id(tokenizer, "<unk>"),
        custom_chat_template=identity_template,
        has_bos=False,
        has_system_role=True,
        allow_tool_role=True,
    )


def _build_sft_default(tokenizer):
    pad = _get_pad_token_id(tokenizer)
    return PromptConfig(
        assistant_prefix_len=0,
        pad_token_id=(pad if pad is not None else _get_eos_token_id(tokenizer)),
        custom_chat_template=_get_chat_template(tokenizer),
        has_bos=_get_bos_token_id(tokenizer) is not None,
        has_system_role=True,
        skip_masking=True,
        allow_tool_role=True,
    )


def _build_multimodal_mistral(tokenizer):
    return PromptConfig(
        assistant_prefix_len=0,
        pad_token_id=_get_unk_token_id(tokenizer),
        custom_chat_template=mistral_custom_template,
        has_bos=True,
        has_system_role=False,
        validate_no_image_in_assistant=True,
    )


def _build_multimodal_llama3(tokenizer):
    return PromptConfig(
        assistant_prefix_len=4,
        pad_token_id=_token_to_id(tokenizer, "<|end_of_text|>"),
        custom_chat_template=None,
        has_bos=True,
        has_system_role=True,
        validate_no_image_in_assistant=True,
    )


def _build_multimodal_llama3p1(tokenizer):
    return PromptConfig(
        assistant_prefix_len=4,
        pad_token_id=_token_to_id(tokenizer, "<|finetune_right_pad_id|>"),
        custom_chat_template=llama3p1_chat_template,
        has_bos=True,
        has_system_role=True,
        validate_no_image_in_assistant=True,
    )


def _build_multimodal_nvlm_yi_34b(tokenizer):
    return PromptConfig(
        assistant_prefix_len=4,
        pad_token_id=_get_pad_token_id(tokenizer),
        custom_chat_template=nvlm_yi_34b_template,
        has_bos=True,
        has_system_role=True,
        validate_no_image_in_assistant=True,
    )


def _build_multimodal_chatml(tokenizer):
    return PromptConfig(
        assistant_prefix_len=3,
        pad_token_id=_get_pad_token_id(tokenizer),
        custom_chat_template=None,
        has_bos=False,
        has_system_role=True,
        validate_no_image_in_assistant=True,
    )


def _build_multimodal_nemotron5(tokenizer):
    return PromptConfig(
        assistant_prefix_len=3,
        pad_token_id=_token_to_id(tokenizer, "<SPECIAL_233>"),
        custom_chat_template=nemotron_custom_template,
        has_bos=True,
        has_system_role=True,
        validate_no_image_in_assistant=True,
    )


def _build_multimodal_nemotron5_aligned(tokenizer):
    return PromptConfig(
        assistant_prefix_len=2,
        pad_token_id=_token_to_id(tokenizer, "<SPECIAL_233>"),
        custom_chat_template=nemotron_aligned_custom_template,
        has_bos=True,
        has_system_role=True,
        capitalize_roles=True,
        validate_no_image_in_assistant=True,
    )


def _build_multimodal_qwen2(tokenizer, force_system_message=False):
    return PromptConfig(
        assistant_prefix_len=3,
        pad_token_id=_get_pad_token_id(tokenizer),
        custom_chat_template=qwen2p0_custom_template,
        has_bos=False,
        has_system_role=True,
        force_system_message=force_system_message,
        system_default={
            "role": "system",
            "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant.",
        },
        validate_no_image_in_assistant=True,
    )


# Registry mapping prompt format names to factory functions.
# Factory functions take a tokenizer and return a PromptConfig.
# Some entries map to the same factory (e.g. llama3p1 and llama3p2).
PROMPT_FORMAT_REGISTRY: Dict[str, callable] = {
    # SFT formats
    "nemotron-nano-v2": _build_sft_nemotron_nano_v2,
    "nemotron-h-aligned": _build_sft_nemotron_h_aligned,
    "identity": _build_sft_identity,
    "default": _build_sft_default,
    # Multimodal formats
    "mistral": _build_multimodal_mistral,
    "llama3": _build_multimodal_llama3,
    "llama3p1": _build_multimodal_llama3p1,
    "llama3p2": _build_multimodal_llama3p1,  # Same config as llama3p1
    "nvlm-yi-34b": _build_multimodal_nvlm_yi_34b,
    "chatml": _build_multimodal_chatml,
    "nemotron5": _build_multimodal_nemotron5,
    "nemotron5-aligned": _build_multimodal_nemotron5_aligned,
    "qwen2p0": _build_multimodal_qwen2,
    "qwen2p5": _build_multimodal_qwen2,
}
