# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

"""Multimodal tokenizer."""
from dataclasses import dataclass
import json
import os
import time
import uuid
from typing import Dict, List, Union, Optional

import numpy as np

from megatron.core.datasets.megatron_tokenizer import MegatronTokenizer

# Mark tokens that will be ignored in the loss function with this value.
# Same ignore_index in https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html
from megatron.core.models.multimodal.llava_model import IGNORE_INDEX, IMAGE_TOKEN

IMAGE_TAGS = {
    "nvlm": ("<Image>", "</Image>"),
    "internvl": ("<img>", "</img>"),
    "": None,  # Image tag not used.
}


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


nvlm_yi_34b_template = "{{- bos_token }}{% for message in messages %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"


qwen2p0_custom_template = "{% for message in messages %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"


# Note: this is the same template as https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct/blob/main/tokenizer_config.json#L2053
# but we removed the forced system message.
llama3p1_chat_template = """{{- bos_token }}\n{%- if custom_tools is defined %}\n    {%- set tools = custom_tools %}\n{%- endif %}\n{%- if not tools_in_user_message is defined %}\n    {%- set tools_in_user_message = true %}\n{%- endif %}\n{%- if not date_string is defined %}\n    {%- set date_string = \"26 Jul 2024\" %}\n{%- endif %}\n{%- if not tools is defined %}\n    {%- set tools = none %}\n{%- endif %}\n\n{#- This block extracts the system message, so we can slot it into the right place. #}\n{%- if messages[0]['role'] == 'system' %}\n    {%- set system_message = messages[0]['content']|trim %}\n    {%- set messages = messages[1:] %}\n{%- else %}\n    {%- set system_message = none %}\n{%- endif %}\n\n{%- if system_message is not none %}{#- System message + builtin tools #}\n{{- \"<|start_header_id|>system<|end_header_id|>\\n\\n\" }}\n{%- if builtin_tools is defined or tools is not none %}\n    {{- \"Environment: ipython\\n\" }}\n{%- endif %}\n{%- if builtin_tools is defined %}\n    {{- \"Tools: \" + builtin_tools | reject('equalto', 'code_interpreter') | join(\", \") + \"\\n\\n\"}}\n{%- endif %}{%- if tools is not none and not tools_in_user_message %}\n    {{- \"You have access to the following functions. To call a function, please respond with JSON for a function call.\" }}\n    {{- 'Respond in the format {\"name\": function name, \"parameters\": dictionary of argument name and its value}.' }}\n    {{- \"Do not use variables.\\n\\n\" }}\n    {%- for t in tools %}\n        {{- t | tojson(indent=4) }}\n        {{- \"\\n\\n\" }}\n    {%- endfor %}\n{%- endif %}\n{{- system_message }}\n{{- \"<|eot_id|>\" }}\n\n{%-endif %}{#- Custom tools are passed in a user message with some extra guidance #}\n{%- if tools_in_user_message and not tools is none %}\n    {#- Extract the first user message so we can plug it in here #}\n    {%- if messages | length != 0 %}\n        {%- set first_user_message = messages[0]['content']|trim %}\n        {%- set messages = messages[1:] %}\n    {%- else %}\n        {{- raise_exception(\"Cannot put tools in the first user message when there's no first user message!\") }}\n{%- endif %}\n    {{- '<|start_header_id|>user<|end_header_id|>\\n\\n' -}}\n    {{- \"Given the following functions, please respond with a JSON for a function call \" }}\n    {{- \"with its proper arguments that best answers the given prompt.\\n\\n\" }}\n    {{- 'Respond in the format {\"name\": function name, \"parameters\": dictionary of argument name and its value}.' }}\n    {{- \"Do not use variables.\\n\\n\" }}\n    {%- for t in tools %}\n        {{- t | tojson(indent=4) }}\n        {{- \"\\n\\n\" }}\n    {%- endfor %}\n    {{- first_user_message + \"<|eot_id|>\"}}\n{%- endif %}\n\n{%- for message in messages %}\n    {%- if not (message.role == 'ipython' or message.role == 'tool' or 'tool_calls' in message) %}\n        {{- '<|start_header_id|>' + message['role'] + '<|end_header_id|>\\n\\n'+ message['content'] | trim + '<|eot_id|>' }}\n    {%- elif 'tool_calls' in message %}\n        {%- if not message.tool_calls|length == 1 %}\n            {{- raise_exception(\"This model only supports single tool-calls at once!\") }}\n        {%- endif %}\n        {%- set tool_call = message.tool_calls[0].function %}\n        {%- if builtin_tools is defined and tool_call.name in builtin_tools %}\n            {{- '<|start_header_id|>assistant<|end_header_id|>\\n\\n' -}}\n            {{- \"<|python_tag|>\" + tool_call.name + \".call(\" }}\n            {%- for arg_name, arg_val in tool_call.arguments | items %}\n                {{- arg_name + '=\"' + arg_val + '\"' }}\n                {%- if not loop.last %}\n                    {{- \", \" }}\n                {%- endif %}\n                {%- endfor %}\n            {{- \")\" }}\n        {%- else  %}\n            {{- '<|start_header_id|>assistant<|end_header_id|>\\n\\n' -}}\n            {{- '{\"name\": \"' + tool_call.name + '\", ' }}\n            {{- '\"parameters\": ' }}\n            {{- tool_call.arguments | tojson }}\n            {{- \"}\" }}\n        {%- endif %}\n        {%- if builtin_tools is defined %}\n            {#- This means we're in ipython mode #}\n            {{- \"<|eom_id|>\" }}\n        {%- else %}\n            {{- \"<|eot_id|>\" }}\n        {%- endif %}\n    {%- elif message.role == \"tool\" or message.role == \"ipython\" %}\n        {{- \"<|start_header_id|>ipython<|end_header_id|>\\n\\n\" }}\n        {%- if message.content is mapping or message.content is iterable %}\n            {{- message.content | tojson }}\n        {%- else %}\n            {{- message.content }}\n        {%- endif %}\n        {{- \"<|eot_id|>\" }}\n    {%- endif %}\n{%- endfor %}\n{%- if add_generation_prompt %}\n    {{- '<|start_header_id|>assistant<|end_header_id|>\\n\\n' }}\n{%- endif %}\n"""

nemotron_custom_template = "{{- bos_token }}{% for message in messages %}{{'<SPECIAL_14>' + message['role'] + '\n' + message['content'] + '<SPECIAL_15>' + '\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<SPECIAL_14>assistant\n' }}{% endif %}"

nemotron_aligned_custom_template = "{{- bos_token}}{% for message in messages %}{{message['role'] + '\n' + message['content'] + '\n' + '[PREFIX]'}}{% endfor %}{% if add_generation_prompt %}{{ 'Assistant\n' }}{% endif %}"

llama_nemotron_template = "{%- if messages[0]['role'] == 'system' -%}{%- set system_message = 'detailed thinking off\n\n' + messages[0]['content'] | trim -%}{%- set messages = messages[1:] -%}{%- else -%}{%- set system_message = 'detailed thinking off' -%}{%- endif -%}{%- if tools is not none -%}{{- '<|begin_of_text|><|start_header_id|>system<|end_header_id|>' + '\n\n' + system_message -}} {{- '\n\n' if system_message else '' -}} {{- '<AVAILABLE_TOOLS>[' -}} {% for t in tools %}{{- (t.function if t.function is defined else t) | tojson() -}}{{- ', ' if not loop.last else '' -}}{%- endfor -%} {{- ']</AVAILABLE_TOOLS>' -}} {{- '<|eot_id|>' -}}{%- else -%}{{- '<|begin_of_text|><|start_header_id|>system<|end_header_id|>' + '\n\n' + system_message + '<|eot_id|>' -}}{%- endif -%}{%- for message in messages -%}{%- if message['role'] == 'user' -%}{{- '<|start_header_id|>user<|end_header_id|>' + '\n\n' + message['content'] | trim + '<|eot_id|>' -}}{%- elif message['role'] == 'tool' -%}{%- set tool_response = '<TOOL_RESPONSE>[' + message['content'] | trim + ']</TOOL_RESPONSE>' -%}{{- '<|start_header_id|>user<|end_header_id|>' + '\n\n' + tool_response + '<|eot_id|>' -}}{%- elif message['role'] == 'assistant' and message.get('tool_calls') is not none -%}{%- set tool_calls = message['tool_calls'] -%}{{- '<|start_header_id|>assistant<|end_header_id|>' + '\n\n' + '<TOOLCALL>[' -}}{%- for tool_call in tool_calls -%}{{ '{' + '\"name\": \"' + tool_call.function.name + '\", \"arguments\": ' + tool_call.function.arguments | tojson + '}' }}{%- if not loop.last -%}{{ ', ' }}{%- else -%}{{ ']</TOOLCALL>' + '<|eot_id|>' }}{%- endif -%}{%- endfor -%}{%- elif message['role'] == 'assistant' -%}{{- '<|start_header_id|>assistant<|end_header_id|>' + '\n\n' + message['content'] | trim + '<|eot_id|>' -}}{%- endif -%}{%- endfor -%}{%- if add_generation_prompt -%}{{ '<|start_header_id|>assistant<|end_header_id|>' + '\n\n' }}{%- endif -%}"

nemotron_h_reasoning_template = "{{ '<SPECIAL_10>System\n' }}{%- if messages and messages[0]['role'] == 'system' -%}{{ messages[0]['content'].strip() }}{%- endif -%}{% for message in (messages[1:] if messages[0]['role'] == 'system' else messages) %}{%- if message['role'] == 'user' -%}{{ '\n<SPECIAL_11>User\n' + message['content'].strip() + '\n<SPECIAL_11>Assistant\n' }}{%- if loop.last -%}{%- if messages[0]['role'] == 'system' -%}{%- if \"{'reasoning': True}\" in messages[0]['content'] -%}{{ '<think>\n' }}{%- elif \"{'reasoning': False}\" in messages[0]['content'] -%}{{ '<think></think>' }}{%- endif -%}{%- endif -%}{%- endif -%}{%- elif message['role'] == 'assistant' -%}{{ message['content'].strip() }}{%- endif -%}{%- endfor -%}<SPECIAL_11>"

nemotron_h_5p5_reasoning_template = "{% for message in messages %}{% set content = message['content'] %}{% if message['role'] == 'system' %}{{ '<SPECIAL_10>System\n' + content.replace('/think', '').replace('/no_think', '').strip() + '\n' }}{% elif message['role'] == 'user' %}{{ '<SPECIAL_11>User\n' + content.replace('/think', '').replace('/no_think', '').strip() + '\n' }}{% elif message['role'] == 'assistant' %}{{ '<SPECIAL_11>Assistant\n' + content.strip() + '\n<SPECIAL_12>\n' }}{% endif %}{% endfor %}"

nemotron_h_5p5_reasoning_inference_template = "{%- for message in messages %}{%- set content = message['content'] %}{%- if message['role'] == 'system' %}{{- '<SPECIAL_10>System\n' + content.replace('/think', '').replace('/no_think', '').strip() }}{%- if tools -%}{%- if content.replace('/think', '').replace('/no_think', '').strip() != '' -%}{{- '\n\n' -}}{%- endif -%}{{- 'You can use the following tools to assist the user if required:\n<AVAILABLE_TOOLS>[' -}}{%- for tool in tools -%}{{- (tool.function if tool.function is defined else tool) | tojson -}}{{- ', ' if not loop.last else '' -}}{%- endfor -%}{{- ']</AVAILABLE_TOOLS>\n\nIf you decide to call any tool(s), use the following format:\n<TOOLCALL>[{{\"name\": \"tool_name1\", \"arguments\": \"tool_args1\"}}, {{\"name\": \"tool_name2\", \"arguments\": \"tool_args2\"}}]</TOOLCALL>\n\nThe user will execute tool-calls and return responses from tool(s) in this format:\n<TOOL_RESPONSE>[{{\"tool_response1\"}}, {{\"tool_response2\"}}]</TOOL_RESPONSE>\n\nBased on the tool responses, you can call additional tools if needed, correct tool calls if any errors are found, or just respond to the user.' -}}{%- endif -%}{{- '\n' -}}{%- elif message['role'] == 'user' %}{{- '<SPECIAL_11>User\n' + content.replace('/think', '').replace('/no_think', '').strip() + '\n' }}{%- elif message['role'] == 'tool' %}{%- if loop.first or (messages[loop.index0 - 1].role != 'tool') -%}{{- '<SPECIAL_11>User\n' + '<TOOL_RESPONSE>[' }}{%- endif -%}{{- message.content -}}{{- ', ' if not loop.last and (messages[loop.index0 + 1].role == 'tool') else '' -}}{%- if loop.last or (messages[loop.index0 + 1].role != 'tool') -%}{{- ']</TOOL_RESPONSE>\n' -}}{%- endif -%}{%- elif message['role'] == 'assistant' %}{%- if '</think>' in content %}{%- set content = content.split('</think>')[1].strip() %}{%- endif %}{{- '<SPECIAL_11>Assistant\n' + content.strip() }}{%- if message.tool_calls -%}{%- if content.strip() != '' -%}{{- '\n\n' -}}{%- endif -%}{{- '<TOOLCALL>[' -}}{%- for call in message.tool_calls -%}{%- set fn = call.function if call.function is defined else call -%}{{- '{\"name\": \"' + fn.name + '\", \"arguments\": ' -}}{%- if fn.arguments is string -%}{{- fn.arguments -}}{%- else -%}{{- fn.arguments | tojson -}}{%- endif -%}{{- '}' + (', ' if not loop.last else '') -}}{%- endfor -%}{{- ']</TOOLCALL>' -}}{%- endif -%}{{- '\n<SPECIAL_12>\n' -}}{%- endif %}{%- endfor %}{%- set ns = namespace(enable_thinking=true) %}{%- for message in messages %}{%- set content = message['content'] %}{%- if message['role'] == 'user' or message['role'] == 'system' %}{%- if '/think' in content %}{%- set ns.enable_thinking = true %}{%- elif '/no_think' in content %}{%- set ns.enable_thinking = false %}{%- endif %}{%- endif %}{%- endfor %}{%- if add_generation_prompt %}{{- '<SPECIAL_11>Assistant\n' }}{%- if ns.enable_thinking is defined and ns.enable_thinking is false %}{{- '<think></think>' }}{%- else %}{{- '<think>\n' }}{%- endif %}{%- endif %}"

llama_nemotron_super_template = "{{- bos_token }}{%- if messages[0]['role'] == 'system' %}{%- set system_message = messages[0]['content']|trim %}{%- set messages = messages[1:] %}{%- else %}{%- set system_message = \"\" %}{%- endif %}{{- \"<|start_header_id|>system<|end_header_id|>\n\n\" }}{{- system_message }}{{- \"<|eot_id|>\" }}{%- for message in messages %}{%- if message['role'] == 'assistant' and '</think>' in message['content'] %}{%- set content = message['content'].split('</think>')[-1].lstrip() %}{%- else %}{%- set content = message['content'] %}{%- endif %}{{- '<|start_header_id|>' + message['role'] + '<|end_header_id|>\n\n' + content | trim + '<|eot_id|>' }}{%- endfor %}{%- if add_generation_prompt %}{{- '<|start_header_id|>assistant<|end_header_id|>\n\n' }}{%- endif %}"

# Llama Nemotron Super 1.5 chat template
llama_nemotron_super_1p5_template = """
{%- set bos = "<|begin_of_text|>" -%}
{%- set enable_thinking = true -%}
{%- set system_start_header = "<|start_header_id|>" -%}
{%- set system_end_header = "<|end_header_id|>\n\n" -%}
{%- set start_header = "<|start_header_id|>" -%}
{%- set end_header = "<|end_header_id|>\n\n" -%}
{%- set eot = "<|eot_id|>" -%}
{%- set system_token = "system" -%}
{%- set user_token = "user" -%}
{%- set assistant_token = "assistant" -%}
{%- set tool_token = "tool" -%}
{{- bos ~ system_start_header ~ system_token ~ system_end_header -}}
{%- if messages[0].role == 'system' and messages[0].content != '' -%}
    {%- set system_content = messages[0].content -%}
    {%- if '/no_think' in system_content -%}
        {%- set system_content = system_content.replace('/no_think', '')|trim -%}
        {%- set enable_thinking = false -%}
    {%- elif '/think' in system_content -%}
        {%- set system_content = system_content.replace('/think', '')|trim -%}
        {%- set enable_thinking = true -%}
    {%- endif -%}
    {{- system_content + '\n\n' -}}
{%- endif -%}
{%- if tools -%}
    {{- 'You can use the following tools to assist the user if required:\n<AVAILABLE_TOOLS>[' -}}
    {%- for tool in tools -%}
        {{- (tool.function if tool.function is defined else tool) | tojson -}}
        {{- ', ' if not loop.last else '' -}}
    {%- endfor -%}
    {{- ']</AVAILABLE_TOOLS>\n\nIf you decide to call any tool(s), use the following format:\n<TOOLCALL>[{{"name": "tool_name1", "arguments": "tool_args1"}}, {{"name": "tool_name2", "arguments": "tool_args2"}}]</TOOLCALL>\n\nResponse from tool(s) will be returned in this format:\n<TOOL_RESPONSE>[{{"response": "tool_response1"}}, {{"response": "tool_response2"}}]</TOOL_RESPONSE>\n\nBased on the results returned by the tool(s), you can call additional tools if needed, correct tool calls if any errors are found, or just respond with the answer to the user.' -}}
{%- endif -%}
{{- eot -}}
{%- for message in messages -%}
    {%- if message.role == user_token -%}
        {{- start_header ~ user_token ~ end_header -}}{{ message.content -}}{{ eot -}}
    {%- elif message.role == assistant_token -%}
        {%- if '</think>' in message.content -%}
            {%- set content = message.content.split('</think>')[-1].lstrip() -%}
        {%- else -%}
            {%- set content = message.content -%}
        {%- endif -%}
        {{- start_header ~ assistant_token ~ end_header -}}{{ content -}}
        {%- if message.tool_calls -%}
            {{- '<TOOLCALL>[' -}}
            {%- for call in message.tool_calls -%}
                {%- set fn = call.function if call.function is defined else call -%}
                {{- '{"name": "' + fn.name + '", "arguments": ' -}}
                {%- if fn.arguments is string -%}
                    {{- fn.arguments -}}
                {%- else -%}
                    {{- fn.arguments | tojson -}}
                {%- endif -%}
                {{- '}' + (', ' if not loop.last else '') -}}
            {%- endfor -%}
            {{- ']</TOOLCALL>' -}}
        {%- endif -%}
        {{- eot -}}
    {%- elif message.role == tool_token -%}
        {%- if loop.first or (messages[loop.index0 - 1].role != tool_token) -%}
            {{- start_header ~ tool_token ~ end_header -}}{{ '<TOOL_RESPONSE>[' -}}
        {%- endif -%}
        {{- message.content -}}
        {{- ', ' if not loop.last and (messages[loop.index0 + 1].role == tool_token) else '' -}}
        {%- if loop.last or (messages[loop.index0 + 1].role != tool_token) -%}
            {{- ']</TOOL_RESPONSE>' -}}{{ eot -}}
        {%- endif -%}
    {%- endif -%}
{%- endfor -%}
{%- if add_generation_prompt -%}
    {{- start_header ~ assistant_token ~ end_header -}}
    {%- if not enable_thinking -%}
        {{- '<think>\n\n</think>\n\n' -}}
    {%- endif -%}
{%- endif -%}
"""

@dataclass
class PromptConfig:
    """Config options for different prompt formats."""

    # How many tokens are used for the assistant prefix, e.g. "<|im_start|>assistant\n".
    # Used for masking the assistant prefix.
    assistant_prefix_len: int
    # Padding token ID.
    pad_token_id: int
    # For overriding the default chat format template.
    custom_chat_template: str
    # If the tokenizer inserts BOS token by default.
    has_bos: bool
    # If the tokenizer supports a separate role for system messages.
    has_system_role: bool
    # Wether to force a specific system message.
    force_system_message: bool = False
    system_default: dict = None


class MultimodalTokenizer(MegatronTokenizer):
    """Multimodal Tokenizer."""

    def __init__(
        self,
        tokenizer: MegatronTokenizer,
        prompt_format: str,
        special_tokens: List[str],
        image_tag_type: str,
        force_system_message: bool = False,
        keep_history_thinking: bool = False,
    ):
        """Tokenizer with a support for non-text inputs.

        Note: Currently, only HuggingFaceTokenizer is supported as the underlying text tokenizer.

        Args:
            tokenizer (MegatronTokenizer): Underlying tokenizer.
            prompt_format (str): Prompt format for the tokenizer.
            special_tokens (List[str]): Non-text tokens.
            image_tag_type (str): Image tag to apply, if any. For example <img><image></img>.
        """
        num_added_tokens = tokenizer.add_tokens(special_tokens, special_tokens=True)
        if prompt_format == "nemotron6-moe":
            # Special tokens already added in the tokenizer.
            assert num_added_tokens == 0
        else:
            assert num_added_tokens == len(
                special_tokens
            ), f"failed to add {len(special_tokens)} special tokens; only added {num_added_tokens}"
        self._vocab_size = len(tokenizer)

        self._tokenizer = tokenizer

        if prompt_format == "mistral":
            # Mistral format doesn't have prefix for the assistant message.
            self._prompt_config = PromptConfig(
                assistant_prefix_len=0,
                pad_token_id=tokenizer.unk_token_id,
                custom_chat_template=mistral_custom_template,
                has_bos=True,
                has_system_role=False,
            )
        elif prompt_format == "llama3":
            # "<|start_header_id|>assistant<|end_header|>\n\n" is the prefix for assistant messages.
            self._prompt_config = PromptConfig(
                assistant_prefix_len=4,
                pad_token_id=tokenizer.convert_tokens_to_ids("<|end_of_text|>"),
                custom_chat_template=None,
                has_bos=True,
                has_system_role=True,
            )
        elif prompt_format in ("llama3p1", "llama3p2"):
            # "<|start_header_id|>assistant<|end_header|>\n\n" is the prefix for assistant messages.
            # That occupies 4 tokens and can be masked in the target.
            self._prompt_config = PromptConfig(
                assistant_prefix_len=4,
                pad_token_id=tokenizer.convert_tokens_to_ids("<|finetune_right_pad_id|>"),
                custom_chat_template=llama3p1_chat_template,
                has_bos=True,
                has_system_role=True,
            )
        elif prompt_format == "llama_nemotron_8b":
            # "<|start_header_id|>assistant<|end_header|>\n\n" is the prefix for assistant messages.
            self._prompt_config = PromptConfig(
                assistant_prefix_len=4,
                pad_token_id=tokenizer.convert_tokens_to_ids("<|end_of_text|>"),
                custom_chat_template=llama_nemotron_template,
                has_bos=True,
                has_system_role=True,
            )
        elif prompt_format == "nvlm-yi-34b":
            self._prompt_config = PromptConfig(
                assistant_prefix_len=4,
                pad_token_id=tokenizer.pad_token_id,
                custom_chat_template=nvlm_yi_34b_template,
                has_bos=True,
                has_system_role=True,
            )
        elif prompt_format == "chatml":
            # "<|im_start|>assistant\n" is the prefix for assistant messages
            self._prompt_config = PromptConfig(
                assistant_prefix_len=3,
                pad_token_id=tokenizer.pad_token_id,
                custom_chat_template=None,
                has_bos=False,
                has_system_role=True,
            )
        elif prompt_format == "nemotron5":
            # "<|im_start|>assistant\n" is the prefix.
            self._prompt_config = PromptConfig(
                assistant_prefix_len=3,
                pad_token_id=tokenizer.convert_tokens_to_ids("<SPECIAL_233>"),
                custom_chat_template=nemotron_custom_template,
                has_bos=True,
                has_system_role=True,
            )
        elif prompt_format == "nemotron5-aligned":
            # "Assistant\n" is the prefix.
            self._prompt_config = PromptConfig(
                assistant_prefix_len=2,
                pad_token_id=tokenizer.convert_tokens_to_ids("<SPECIAL_233>"),
                custom_chat_template=nemotron_aligned_custom_template,
                has_bos=True,
                has_system_role=True,
            )
        elif prompt_format in ("qwen2p0", "qwen2p5"):
            # "<|im_start|>assistant\n" is the prefix for assistant messages
            self._prompt_config = PromptConfig(
                assistant_prefix_len=3,
                pad_token_id=tokenizer.pad_token_id,
                custom_chat_template=qwen2p0_custom_template,
                has_bos=False,
                has_system_role=True,
                force_system_message=force_system_message,
                system_default={"role": "system",
                                "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."}
            )
        elif prompt_format == "llama3p1":
            # "<|start_header_id|>assistant<|end_header|>\n\n" is the prefix for assistant messages.
            # That occupies 4 tokens and can be masked in the target.
            self._prompt_config = PromptConfig(
                assistant_prefix_len=4,
                pad_token_id=tokenizer.convert_tokens_to_ids("<|finetune_right_pad_id|>"),
                custom_chat_template=llama3p1_chat_template,
                has_bos=True,
                has_system_role=True,
            )
        elif prompt_format == "nemotron-h-reasoning":
            self._prompt_config = PromptConfig(
                assistant_prefix_len=3,
                pad_token_id=tokenizer.convert_tokens_to_ids("<SPECIAL_11>"),
                custom_chat_template=nemotron_h_reasoning_template,
                has_bos=False,
                has_system_role=True,
            )
        elif prompt_format == "nemotron-h-5p5-reasoning":
            self._prompt_config = PromptConfig(
                assistant_prefix_len=3,
                pad_token_id=tokenizer.convert_tokens_to_ids("<unk>"),
                custom_chat_template=nemotron_h_5p5_reasoning_template,
                has_bos=False,
                has_system_role=True,
                force_system_message=force_system_message,
            )
        elif prompt_format == "nemotron-h-5p5-reasoning-inference":
            self._prompt_config = PromptConfig(
                assistant_prefix_len=3,
                pad_token_id=tokenizer.convert_tokens_to_ids("<unk>"),
                custom_chat_template=nemotron_h_5p5_reasoning_inference_template,
                has_bos=False,
                has_system_role=True,
            )
        elif prompt_format == "llama-nemotron-super":
            self._prompt_config = PromptConfig(
                assistant_prefix_len=4,
                pad_token_id=tokenizer.convert_tokens_to_ids("<|finetune_right_pad_id|>"),
                custom_chat_template=llama_nemotron_super_template,
                has_bos=True,
                has_system_role=True,
            )
        elif prompt_format == "llama-nemotron-super-1p5":
            self._prompt_config = PromptConfig(
                assistant_prefix_len=4,
                pad_token_id=tokenizer.convert_tokens_to_ids("<|finetune_right_pad_id|>"),
                custom_chat_template=llama_nemotron_super_1p5_template,
                has_bos=True,
                has_system_role=True,
            )
        elif prompt_format == "nemotron6-moe":
            self._prompt_config = PromptConfig(
                assistant_prefix_len=None, # Not used.
                pad_token_id=tokenizer.convert_tokens_to_ids("<unk>"),
                custom_chat_template=None,
                has_bos=False,
                has_system_role=True,
            )
        else:
            raise NotImplementedError("unknown multimodal tokenizer type", prompt_format)

        self._prompt_format = prompt_format
        self._image_tag = IMAGE_TAGS[image_tag_type]
        self._keep_history_thinking = keep_history_thinking

    def _write_debug_snapshot(
        self,
        conversation: List[Dict],
        tokens: np.ndarray,
        target: np.ndarray,
        train_only_on_last_assistant_turn: bool,
        has_nonempty_thinking_trace: bool,
    ) -> None:
        """Write debug artifacts to disk when DEBUG=1."""
        if os.environ.get("DEBUG") != "1":
            return

        log_dir = os.environ.get("MM_TOKENIZER_DEBUG_DIR", "multimodal_tokenizer_logs")
        os.makedirs(log_dir, exist_ok=True)

        unique_key = f"{int(time.time())}_{os.getpid()}_{uuid.uuid4().hex[:8]}"
        base_path = os.path.join(log_dir, unique_key)

        # Prepare target for easier reading by collapsing consecutive IGNORE_INDEX.
        target_to_print = target.copy()
        is_ignore = target_to_print == IGNORE_INDEX
        prev_is_ignore = np.roll(is_ignore, 1)
        prev_is_ignore[0] = False  # First element has no previous.
        keep_mask = ~is_ignore | (is_ignore & ~prev_is_ignore)
        target_to_print = target_to_print[keep_mask]
        target_to_print[target_to_print == IGNORE_INDEX] = 0

        with open(f"{base_path}_conversation.json", "w", encoding="utf-8") as handle:
            json.dump(
                {
                    "train_only_on_last_assistant_turn": train_only_on_last_assistant_turn,
                    "conversation": conversation,
                    "has_nonempty_thinking_trace": has_nonempty_thinking_trace,
                },
                handle,
                ensure_ascii=True,
                indent=2,
            )
        with open(f"{base_path}_input.txt", "w", encoding="utf-8") as handle:
            handle.write(self.detokenize(tokens))
        with open(f"{base_path}_target.txt", "w", encoding="utf-8") as handle:
            handle.write(self.detokenize(target_to_print))

    def offsets(self, ids: list[int], text: str) -> list[int]:
        """
        Assume that the tokenizer is a HuggingFaceTokenizer.
        Copied from megatron.training.tokenizer.tokenizer.py:_HuggingFaceTokenizer.offsets
        """
        retok_ids: "transformers.BatchEncoding" = self._tokenizer(text)
        offsets, next_start_idx = [], 0
        for i in range(len(ids)):
            span = retok_ids.token_to_chars(i)
            if span is not None:
                offsets.append(span.start)
                next_start_idx = span.end
            else:
                offsets.append(next_start_idx)
        return offsets

    def _has_nonempty_thinking_trace(self, conversation: List[Dict]) -> bool:
        """Return True if any assistant message has a non-empty <think> trace."""
        for turn in conversation:
            if turn.get("role") != "assistant":
                continue
            content = turn.get("content") or ""
            if "<think>" not in content or "</think>" not in content:
                continue
            inner = content.split("<think>", 1)[1].split("</think>", 1)[0]
            if inner.strip():
                return True
        return False

    def _apply_image_tag(self, text: Union[str, List[Dict]]):
        """Surround <image> with image tags such as <img> and </img>."""
        if self._image_tag is None:
            return text

        replacement = f"{self._image_tag[0]}{IMAGE_TOKEN}{self._image_tag[1]}"

        if isinstance(text, list):
            for turn in text:
                turn["content"] = turn["content"].replace(IMAGE_TOKEN, replacement)
        else:
            text = text.replace(IMAGE_TOKEN, replacement)

        return text

    def tokenize(self, text: Union[str, List[Dict]], **kwargs):
        """Tokenize conversation or string input."""
        if isinstance(text, list):
            # This code path is used by the inference code currently.
            return self.tokenize_conversation(text, return_target=False, add_generation_prompt=True, **kwargs).tolist()

        return self._encode(text, **kwargs)

    def _encode(self, text: str):
        """Tokenize text input."""
        text = self._apply_image_tag(text)
        return self._tokenizer.encode(text)

    def tokenize_conversation(
        self, conversation: List[Dict], return_target: bool, add_generation_prompt: bool, train_only_on_last_assistant_turn: bool = False,
        **kwargs
    ):
        """Convert a conversation to tokens.

        Args:
            conversation (List[Dict]): Sequence of system/user/assistant messages.
                Must be in the following format:
                [
                    {"role": "user", "content": "something"},
                    {"role": "assistant", "content": "something2"},
                ]
            return_target (bool): Return target tokens with system and assistant masked.
            add_generation_prompt (bool): Add assistant prefix to the end.
            train_only_on_last_assistant_turn (bool): Train only on the last assistant turn.
        """
        if train_only_on_last_assistant_turn:
            assert self._prompt_format in ("nemotron6-moe"), "train_only_on_last_assistant_turn is only supported for nemotron6-moe"

        # Skip system message if the tokenizer doesn't have a system role.
        if not self._prompt_config.has_system_role and conversation[0]["role"] == "system":
            conversation = conversation[1:]

        if self._prompt_config.force_system_message:
            assert self._prompt_config.system_default is not None, "Trying to force system message with empty system default"
            if conversation[0]["role"] == "system":
                conversation[0] = self._prompt_config.system_default
            else:
                conversation = [self._prompt_config.system_default] + conversation

        if self._prompt_format == "nemotron5-aligned":
            for turn in conversation:
                tmp = turn['role']
                turn['role'] = tmp[:1].upper() + tmp[1:]

        # Apply possible image tag.
        conversation = self._apply_image_tag(conversation)

        has_nonempty_thinking_trace = self._has_nonempty_thinking_trace(conversation)

        if self._keep_history_thinking:
            kwargs["truncate_history_thinking"] = False

        tokens = self._tokenizer.apply_chat_template(
            conversation,
            tokenize=True,
            add_generation_prompt=add_generation_prompt,
            return_assistant_token_mask=False,
            return_tensors="np",
            chat_template=self._prompt_config.custom_chat_template,
            **kwargs,
        )[0]

        if not return_target:
            return tokens

        target = tokens.copy()

        # Temp hack for nemotron hybrid reasoning model.
        if self._prompt_format in ("nemotron-h-reasoning"):
            idx = np.where(tokens == 11)[0]
            assert tokens[-1] == 11, "last token should be <SPECIAL_11>"
            idx = idx[:-1]
            target[:idx[1]] = IGNORE_INDEX  # system prompt + initial user prompt

            for i in range(1, len(idx)):
                if i % 2 == 0:
                    # user message. Do not mask <SPECIAL_11> because it is also reused for termination the previous assistant.
                    target[idx[i]+1:idx[i+1]] = IGNORE_INDEX
                else:
                    # assistant message. Mask `<SPECIAL_11>Assistant\n`.
                    target[idx[i]:idx[i]+self._prompt_config.assistant_prefix_len] = IGNORE_INDEX

            return tokens, target
        elif self._prompt_format in ("nemotron-h-5p5-reasoning"):
            idx = np.where(tokens == 11)[0]
            target[:idx[1]] = IGNORE_INDEX  # system prompt + initial user prompt

            for i in range(1, len(idx)):
                if i % 2 == 0:
                    # user message. Mask the entire user message including <SPECIAL_11>.
                    target[idx[i]:idx[i+1]] = IGNORE_INDEX
                else:
                    # assistant message. Mask `<SPECIAL_11>Assistant\n`.
                    target[idx[i]:idx[i]+self._prompt_config.assistant_prefix_len] = IGNORE_INDEX

            # Also mask any <SPECIAL_10> (system) tokens that appear in the middle of the conversation
            idx_system = np.where(tokens == 10)[0]
            if len(idx_system) > 0:
                # Find all special tokens (10, 11, 12) to determine boundaries
                all_special_positions = np.sort(np.concatenate([
                    np.where(tokens == 10)[0],
                    np.where(tokens == 11)[0],
                    np.where(tokens == 12)[0]
                ]))

                # For each system token, mask until the next special token
                for sys_pos in idx_system[1:]:  # Skip the first system message (already masked above)
                    # Find the next special token position
                    next_special = all_special_positions[all_special_positions > sys_pos]
                    if len(next_special) > 0:
                        target[sys_pos:next_special[0]] = IGNORE_INDEX

            return tokens, target
        elif self._prompt_format in ("nemotron6-moe"):
            # Mask everything.

            target = np.full_like(tokens, IGNORE_INDEX)
            # Find positions where the pattern [10, 1503, 19464] occurs (true assistant message starts)
            # tokens[i] == 10, tokens[i+1] == 1503, tokens[i+2] == 19464
            pattern_matches = np.where(
                (tokens[:-2] == 10) & (tokens[1:-1] == 1503) & (tokens[2:] == 19464)
            )[0]
            assistant_idx = pattern_matches + 1  # +1 to get the position of token 1503
            end_idx = np.where(tokens == 11)[0]

            # If the conversation has thinking traces and train_only_on_last_assistant_turn is True,
            # only train on the assistant turns after the last non-tool user message.
            if train_only_on_last_assistant_turn and has_nonempty_thinking_trace:
                # User message starts with [10, 3263], tool-call user message starts with [10, 3263, 1010, 16].
                user_start_positions = np.where((tokens[:-1] == 10) & (tokens[1:] == 3263))[0]
                if len(user_start_positions) > 0:
                    # Filter out tool-call user messages.
                    is_tool_call = np.zeros_like(user_start_positions, dtype=bool)
                    # Need 2 extra tokens to compare [1010, 16].
                    valid_tool_check = user_start_positions + 3 < len(tokens)
                    tool_check_positions = user_start_positions[valid_tool_check]
                    is_tool_call[valid_tool_check] = (
                        (tokens[tool_check_positions + 2] == 1010)
                        & (tokens[tool_check_positions + 3] == 16)
                    )
                    non_tool_user_positions = user_start_positions[~is_tool_call]
                    if len(non_tool_user_positions) > 0:
                        last_user_pos = non_tool_user_positions[-1]
                        # Keep assistant turns that start after the last non-tool user message.
                        assistant_idx = assistant_idx[assistant_idx > last_user_pos]

            for i in assistant_idx:
                assert tokens[i+2] == 1010, "expected newline lb"

                lb = i

                valid = end_idx[end_idx > lb]

                ub = valid[0]

                assert tokens[ub+1] == 1010, "expected newline ub"

                target[lb+3:ub+1] = tokens[lb+3:ub+1]

            # import os
            # if os.environ.get("DEBUG") == "1":
            #     self._write_debug_snapshot(
            #         conversation=conversation,
            #         tokens=tokens,
            #         target=target,
            #         train_only_on_last_assistant_turn=train_only_on_last_assistant_turn,
            #         has_nonempty_thinking_trace=has_nonempty_thinking_trace,
            #     )

            return tokens, target

        # Mask system and user tokens in the target.
        idx = 0
        for turn_idx, turn in enumerate(conversation):
            if len(turn["content"]) == 0:
                raise ValueError(f"empty turn in conversation: {conversation}. Skipping.")

            turn_tokens = self._tokenizer.apply_chat_template(
                [turn], tokenize=True, chat_template=self._prompt_config.custom_chat_template,
                **kwargs,
            )

            # There should be only one BOS at the very beginning.
            # After the first turn, skip BOS token.
            if self._prompt_config.has_bos and turn_idx > 0:
                if self._prompt_config.custom_chat_template == llama_nemotron_template:
                    turn_tokens = turn_tokens[10:]
                elif self._prompt_config.custom_chat_template in (llama_nemotron_super_template, llama_nemotron_super_1p5_template):
                    # Skip BOS token (1) + empty system header (5) = 6 tokens total
                    turn_tokens = turn_tokens[6:]
                else:
                    turn_tokens = turn_tokens[1:]

            turn_len = len(turn_tokens)

            role = turn["role"].lower()
            if role in ("system", "user"):
                target[idx : idx + turn_len] = IGNORE_INDEX
            elif role == "assistant":
                if IMAGE_TOKEN in turn["content"]:
                    raise RuntimeError(f"{IMAGE_TOKEN} not allowed in assistant content!")

                if self._prompt_config.assistant_prefix_len > 0:
                    target[idx : idx + self._prompt_config.assistant_prefix_len] = IGNORE_INDEX

            assert np.allclose(
                tokens[idx : idx + turn_len], turn_tokens
            ), f"expected turn tokens to match tokens in conversation {conversation}"

            idx += turn_len

        assert idx == len(tokens), f"mismatch in target masking the conversation {conversation}"

        return tokens, target

    def convert_tokens_to_ids(self, tokens: List[str]):
        """Convert tokens to IDs."""
        return self._tokenizer.convert_tokens_to_ids(tokens)

    def detokenize(self, tokens: List[int]):
        """Detokenize tokens."""
        return self._tokenizer.decode(tokens)

    def get_special_tokens(self):
        """Get special tokens."""
        return self._tokenizer.get_added_vocab()

    @property
    def pad(self):
        """Pad token ID."""
        return self._prompt_config.pad_token_id

    @property
    def eod(self):
        """End of sentence token ID."""
        return self._tokenizer.eos_token_id

    @property
    def vocab(self):
        """Vocab."""
        return NotImplementedError("not used")

    @property
    def inv_vocab(self):
        """Inverse vocab."""
        return NotImplementedError("not used")

    @property
    def vocab_size(self):
        """Vocabulary size."""
        return self._vocab_size
