# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

import pytest

from megatron.core.inference.text_generation_server.dynamic_text_gen_server.endpoints.chat_completions import (
    _sanitize_chat_template_kwargs,
)


@pytest.mark.parametrize(
    "raw_kwargs,expected",
    [
        (
            {
                "chat_template": (
                    "{% for _ in range(100000) %}{% for _ in range(100000) %}"
                    "{% endfor %}{% endfor %}"
                )
            },
            {},
        ),
        (
            {"chat_template": "{{ 'a' * 100000000 }}", "enable_thinking": False},
            {"enable_thinking": False},
        ),
        (
            {"enable_thinking": True, "force_nonempty_content": True},
            {"enable_thinking": True, "force_nonempty_content": True},
        ),
        ({}, {}),
        (None, {}),
        ("chat_template", {}),
        ([{"chat_template": "x"}], {}),
    ],
)
def test_sanitize_chat_template_kwargs_strips_request_supplied_template(raw_kwargs, expected):
    assert _sanitize_chat_template_kwargs(raw_kwargs) == expected


def test_sanitize_chat_template_kwargs_does_not_mutate_caller_payload():
    raw_kwargs = {"chat_template": "{{ 'x' }}", "enable_thinking": False}

    sanitized = _sanitize_chat_template_kwargs(raw_kwargs)

    assert sanitized == {"enable_thinking": False}
    assert raw_kwargs["chat_template"] == "{{ 'x' }}"
