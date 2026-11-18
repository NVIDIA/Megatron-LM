# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

import copy
from collections import defaultdict
from types import SimpleNamespace

import pytest
import torch

from examples.multimodal.data_loading.conversation_base import conversation_convert_message
from examples.multimodal.data_loading.conversation_sample import ConversationSample, Message
from examples.multimodal.data_loading.cookers import conversation as conversation_cooker
from examples.multimodal.data_loading.cookers.conversation import (
    EXPLICIT_ASSISTANT_LOSS_COOK,
    EXPLICIT_ASSISTANT_LOSS_FIELD,
    EXPLICIT_ASSISTANT_LOSS_MODE,
    _validate_explicit_assistant_loss_contract,
    _validate_loss_mask_subflavors,
    _validate_openai_messages_have_no_explicit_loss,
    _validate_standard_jsonl_has_no_explicit_loss,
)
from examples.multimodal.data_loading.task_encoder import MultiModalTaskEncoder
from megatron.core.models.multimodal.llava_model import IGNORE_INDEX


def _valid_data() -> dict:
    return {
        "conversations": [
            {"from": "system", "value": "system"},
            {"from": "human", "value": "task"},
            {"from": "gpt", "value": "history", "loss": False},
            {"from": "human", "value": "observation"},
            {"from": "gpt", "value": "action one", "loss": True},
            {"from": "human", "value": "observation"},
            {"from": "assistant", "value": "action two", "loss": True},
        ]
    }


def _valid_subflavors() -> dict:
    return {
        "cook": EXPLICIT_ASSISTANT_LOSS_COOK,
        "loss_mask_mode": EXPLICIT_ASSISTANT_LOSS_MODE,
        "assistant_loss_mask_field": EXPLICIT_ASSISTANT_LOSS_FIELD,
    }


def test_explicit_assistant_loss_contract_accepts_multiple_targets() -> None:
    _validate_explicit_assistant_loss_contract(_valid_data(), _valid_subflavors())


def test_explicit_assistant_loss_contract_accepts_single_target() -> None:
    data = _valid_data()
    data["conversations"][4]["loss"] = False
    _validate_explicit_assistant_loss_contract(data, _valid_subflavors())


def test_loss_mask_subflavors_accept_legacy_and_explicit_contracts() -> None:
    assert not _validate_loss_mask_subflavors({"cook": "general_conversations_jsonl"})
    assert _validate_loss_mask_subflavors(_valid_subflavors())


def test_loss_mask_subflavors_reject_unknown_mode() -> None:
    subflavors = {
        "cook": "general_conversations_jsonl",
        "loss_mask_mode": "explicit_assistant_turns_v1",
    }
    with pytest.raises(ValueError, match="unsupported loss_mask_mode"):
        _validate_loss_mask_subflavors(subflavors)


@pytest.mark.parametrize("assistant_loss_mask_field", [EXPLICIT_ASSISTANT_LOSS_FIELD, None])
def test_loss_mask_subflavors_reject_stray_field(assistant_loss_mask_field: object) -> None:
    subflavors = {
        "cook": "general_conversations_jsonl",
        "assistant_loss_mask_field": assistant_loss_mask_field,
    }
    with pytest.raises(ValueError, match="assistant_loss_mask_field requires loss_mask_mode"):
        _validate_loss_mask_subflavors(subflavors)


@pytest.mark.parametrize(
    "cook",
    [
        "general_conversations_jsonl",
        "openai_messages_jsonl",
        "openai_messages_offline_packed_jsonl",
    ],
)
def test_loss_mask_subflavors_reject_explicit_mode_with_wrong_cooker(cook: str) -> None:
    subflavors = _valid_subflavors()
    subflavors["cook"] = cook
    with pytest.raises(ValueError, match="requires cook"):
        _validate_loss_mask_subflavors(subflavors)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("loss_mask_mode", "last_assistant_turn"),
        ("assistant_loss_mask_field", "loss"),
        ("train_only_on_last_assistant_turn", True),
        ("skip_chat_template", True),
        ("tool_response_as_turn_boundary", True),
        ("offline_packed_messages", True),
    ],
)
def test_explicit_assistant_loss_contract_rejects_bad_subflavors(field: str, value: object) -> None:
    subflavors = _valid_subflavors()
    subflavors[field] = value
    with pytest.raises(ValueError):
        _validate_explicit_assistant_loss_contract(_valid_data(), subflavors)


@pytest.mark.parametrize(
    "incompatible_option",
    [
        "train_only_on_last_assistant_turn",
        "skip_chat_template",
        "tool_response_as_turn_boundary",
        "offline_packed_messages",
    ],
)
def test_loss_mask_subflavors_reject_incompatible_options(incompatible_option: str) -> None:
    subflavors = _valid_subflavors()
    subflavors[incompatible_option] = True
    with pytest.raises(ValueError, match=f"incompatible with {incompatible_option}"):
        _validate_loss_mask_subflavors(subflavors)


@pytest.mark.parametrize("invalid_loss", [None, 0, 1, "true"])
def test_explicit_assistant_loss_contract_requires_boolean_assistant_loss(
    invalid_loss: object,
) -> None:
    data = _valid_data()
    data["conversations"][2]["loss"] = invalid_loss
    with pytest.raises(ValueError, match="must be a boolean"):
        _validate_explicit_assistant_loss_contract(data, _valid_subflavors())


def test_explicit_assistant_loss_contract_requires_every_assistant_flag() -> None:
    data = _valid_data()
    del data["conversations"][2]["loss"]
    with pytest.raises(ValueError, match="must be a boolean"):
        _validate_explicit_assistant_loss_contract(data, _valid_subflavors())


def test_explicit_assistant_loss_contract_rejects_non_assistant_flag() -> None:
    data = _valid_data()
    data["conversations"][1]["loss"] = False
    with pytest.raises(ValueError, match="only valid on assistant"):
        _validate_explicit_assistant_loss_contract(data, _valid_subflavors())


def test_explicit_assistant_loss_contract_requires_trainable_turn() -> None:
    data = _valid_data()
    for message in data["conversations"]:
        if "loss" in message:
            message["loss"] = False
    with pytest.raises(ValueError, match="at least one loss=true"):
        _validate_explicit_assistant_loss_contract(data, _valid_subflavors())


def test_explicit_assistant_loss_contract_rejects_unknown_sender() -> None:
    data = _valid_data()
    data["conversations"][1]["from"] = "computer"
    with pytest.raises(ValueError, match="unsupported sender"):
        _validate_explicit_assistant_loss_contract(data, _valid_subflavors())


def test_standard_jsonl_rejects_explicit_loss() -> None:
    with pytest.raises(ValueError, match="requires cook"):
        _validate_standard_jsonl_has_no_explicit_loss(_valid_data())

    data = copy.deepcopy(_valid_data())
    for message in data["conversations"]:
        message.pop("loss", None)
    _validate_standard_jsonl_has_no_explicit_loss(data)


def test_general_conversations_jsonl_forwards_named_media_sources(monkeypatch) -> None:
    captured_kwargs = {}
    expected_result = object()

    def record_post_processing(*args, **kwargs):
        captured_kwargs.update(kwargs)
        return expected_result

    monkeypatch.setattr(conversation_cooker, "conversation_post_processing", record_post_processing)
    babyvision_synth = object()
    babyvision_style_questions = object()

    result = conversation_cooker._cook_general_conversations_jsonl(
        {"json": {"conversations": []}},
        cache=object(),
        primary=object(),
        explicit_assistant_loss=False,
        babyvision_synth=babyvision_synth,
        babyvision_style_questions=babyvision_style_questions,
    )

    assert result is expected_result
    assert captured_kwargs["babyvision_synth"] is babyvision_synth
    assert captured_kwargs["babyvision_style_questions"] is babyvision_style_questions
    assert "media_sources" not in captured_kwargs


def test_openai_messages_reject_explicit_loss() -> None:
    messages = [{"role": "assistant", "content": "answer", "loss": False}]
    with pytest.raises(ValueError, match=r"messages\[0\]\.loss is unsupported"):
        _validate_openai_messages_have_no_explicit_loss(messages)

    _validate_openai_messages_have_no_explicit_loss([{"role": "assistant", "content": "answer"}])


@pytest.mark.parametrize("loss", [False, True])
def test_message_loss_round_trips_without_entering_fragments(loss: bool) -> None:
    message = conversation_convert_message(
        {"conversations": []},
        {"from": "gpt", "value": "answer"},
        defaultdict(int),
        check_if_media_file_exist=False,
        loss=loss,
    )
    assert message == Message(sender="assistant", fragments=["answer"], loss=loss)

    serialized = ConversationSample.to_json(SimpleNamespace(conversation=[message]))
    assert serialized == {
        "conversation": [{"sender": "assistant", "fragments": ["answer"], "loss": loss}]
    }
    round_tripped = ConversationSample.from_json(serialized, __key__="sample", __restore_key__=())
    assert round_tripped.conversation == [message]

    legacy_serialized = ConversationSample.to_json(
        SimpleNamespace(conversation=[Message(sender="assistant", fragments=["legacy"])])
    )
    assert legacy_serialized == {"conversation": [{"sender": "assistant", "fragments": ["legacy"]}]}


def test_offline_packed_conversations_reject_explicit_loss() -> None:
    sample = SimpleNamespace(
        __key__="offline",
        conversation=[Message(sender="assistant", fragments=["answer"], loss=True)],
    )
    with pytest.raises(ValueError, match="does not support explicit assistant loss"):
        MultiModalTaskEncoder._split_offline_packed_conversations(object(), sample)


def test_explicit_loss_span_validation_allows_partial_final_span() -> None:
    full = torch.tensor([IGNORE_INDEX, 1, 2, IGNORE_INDEX, 3, 4])
    partial_final = full[:-1]

    MultiModalTaskEncoder._validate_explicit_loss_target_spans(
        full, expected_span_count=2, sample_key="full", phase="truncation"
    )
    MultiModalTaskEncoder._validate_explicit_loss_target_spans(
        partial_final, expected_span_count=2, sample_key="partial", phase="truncation"
    )


def test_explicit_loss_span_validation_rejects_removed_target() -> None:
    removed_final = torch.tensor([IGNORE_INDEX, 1, 2, IGNORE_INDEX])
    with pytest.raises(ValueError, match="expected 2, found 1"):
        MultiModalTaskEncoder._validate_explicit_loss_target_spans(
            removed_final, expected_span_count=2, sample_key="removed", phase="truncation"
        )
