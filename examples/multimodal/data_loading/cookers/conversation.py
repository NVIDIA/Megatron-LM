# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
import dataclasses
import re
from collections import defaultdict

from megatron.energon import CachePool, FileStore, SourceInfo, basic_sample_keys, cooker, stateless

from ..conversation_base import (
    conversation_convert_message,
    conversation_post_processing,
    conversation_tags_mapping_sample_to_allowed,
)
from ..conversation_sample import (
    ConversationSample,
    ImageMedia,
    Message,
    VideoFrameMedia,
    VideoMedia,
)

warn_about_slow_media_loading = defaultdict(lambda: True)
_re_clean_path = re.compile(r"(?:^\./|/\.(?=/))")

NO_TOOL_SYSTEM_CONTENT = (
    "<|im_start|>system\n"
    "You are a helpful and harmless assistant.\n\n"
    "You are not allowed to use any tools.<|im_end|>\n"
)
LEGACY_SYSTEM_CONTENT = "<|im_start|>system\nYou are a helpful and harmless assistant.<|im_end|>\n"
EMPTY_SYSTEM_CONTENT = "<|im_start|>system\n<|im_end|>\n"

EXPLICIT_ASSISTANT_LOSS_COOK = "general_conversations_jsonl_explicit_loss_v1"
EXPLICIT_ASSISTANT_LOSS_MODE = "explicit_assistant_turns"
EXPLICIT_ASSISTANT_LOSS_FIELD = "conversations[*].loss"
_ASSISTANT_SENDERS = {"assistant", "gpt", "agent"}
_KNOWN_SENDERS = _ASSISTANT_SENDERS | {"system", "human", "user", "tool"}
_EXPLICIT_ASSISTANT_LOSS_INCOMPATIBLE_OPTIONS = (
    "train_only_on_last_assistant_turn",
    "skip_chat_template",
    "tool_response_as_turn_boundary",
    "offline_packed_messages",
)


def _validate_loss_mask_subflavors(subflavors: dict) -> bool:
    """Validate loss-mask configuration before selecting its masking path."""
    cook = subflavors.get("cook")
    loss_mask_mode = subflavors.get("loss_mask_mode")
    assistant_loss_mask_field = subflavors.get("assistant_loss_mask_field")

    if loss_mask_mode not in (None, "", EXPLICIT_ASSISTANT_LOSS_MODE):
        raise ValueError(f"unsupported loss_mask_mode={loss_mask_mode!r}")
    if "assistant_loss_mask_field" in subflavors and loss_mask_mode != EXPLICIT_ASSISTANT_LOSS_MODE:
        raise ValueError(
            "assistant_loss_mask_field requires " f"loss_mask_mode={EXPLICIT_ASSISTANT_LOSS_MODE}"
        )
    if loss_mask_mode == EXPLICIT_ASSISTANT_LOSS_MODE:
        if cook != EXPLICIT_ASSISTANT_LOSS_COOK:
            raise ValueError(
                f"loss_mask_mode={EXPLICIT_ASSISTANT_LOSS_MODE} requires "
                f"cook={EXPLICIT_ASSISTANT_LOSS_COOK}"
            )
        if assistant_loss_mask_field != EXPLICIT_ASSISTANT_LOSS_FIELD:
            raise ValueError(
                f"loss_mask_mode={EXPLICIT_ASSISTANT_LOSS_MODE} requires "
                f"assistant_loss_mask_field={EXPLICIT_ASSISTANT_LOSS_FIELD}"
            )
        for incompatible_option in _EXPLICIT_ASSISTANT_LOSS_INCOMPATIBLE_OPTIONS:
            if subflavors.get(incompatible_option, False):
                raise ValueError(
                    "explicit assistant loss is incompatible with " f"{incompatible_option}"
                )
        return True
    if cook == EXPLICIT_ASSISTANT_LOSS_COOK:
        raise ValueError(
            f"cook={EXPLICIT_ASSISTANT_LOSS_COOK} requires "
            f"loss_mask_mode={EXPLICIT_ASSISTANT_LOSS_MODE}"
        )
    return False


def _validate_standard_jsonl_has_no_explicit_loss(data: dict) -> None:
    """Reject explicit loss metadata on the legacy JSONL cooker."""
    for turn_index, message in enumerate(data.get("conversations", [])):
        if isinstance(message, dict) and "loss" in message:
            raise ValueError(
                "conversations[%d].loss requires cook=%s"
                % (turn_index, EXPLICIT_ASSISTANT_LOSS_COOK)
            )


def _validate_openai_messages_have_no_explicit_loss(messages: list) -> None:
    """Reject loss metadata on OpenAI schemas, where it is unsupported."""
    for turn_index, message in enumerate(messages):
        if isinstance(message, dict) and "loss" in message:
            raise ValueError(
                "messages[%d].loss is unsupported; use cook=%s with the "
                "conversations schema" % (turn_index, EXPLICIT_ASSISTANT_LOSS_COOK)
            )


def _validate_explicit_assistant_loss_contract(data: dict, subflavors: dict) -> None:
    """Validate the versioned explicit per-assistant-turn loss contract."""
    if subflavors.get("loss_mask_mode") != EXPLICIT_ASSISTANT_LOSS_MODE:
        raise ValueError(
            f"{EXPLICIT_ASSISTANT_LOSS_COOK} requires "
            f"loss_mask_mode={EXPLICIT_ASSISTANT_LOSS_MODE}"
        )
    if subflavors.get("assistant_loss_mask_field") != EXPLICIT_ASSISTANT_LOSS_FIELD:
        raise ValueError(
            f"{EXPLICIT_ASSISTANT_LOSS_COOK} requires "
            f"assistant_loss_mask_field={EXPLICIT_ASSISTANT_LOSS_FIELD}"
        )
    for incompatible_option in _EXPLICIT_ASSISTANT_LOSS_INCOMPATIBLE_OPTIONS:
        if subflavors.get(incompatible_option, False):
            raise ValueError(f"explicit assistant loss is incompatible with {incompatible_option}")

    conversations = data.get("conversations")
    if not isinstance(conversations, list) or not conversations:
        raise ValueError("conversations must be a non-empty list")

    has_trainable_assistant = False
    for turn_index, message in enumerate(conversations):
        if not isinstance(message, dict):
            raise ValueError(f"conversations[{turn_index}] must be an object")
        sender = message.get("from")
        if sender not in _KNOWN_SENDERS:
            raise ValueError(f"conversations[{turn_index}].from has unsupported sender {sender!r}")
        if sender in _ASSISTANT_SENDERS:
            if "loss" not in message or type(message["loss"]) is not bool:
                raise ValueError(
                    f"conversations[{turn_index}].loss must be a boolean for "
                    f"assistant sender {sender!r}"
                )
            has_trainable_assistant = has_trainable_assistant or message["loss"]
        elif "loss" in message:
            raise ValueError(f"conversations[{turn_index}].loss is only valid on assistant turns")

    if not has_trainable_assistant:
        raise ValueError("explicit assistant loss requires at least one loss=true turn")


def _openai_message_content_to_fragments(content) -> list[str]:
    """Convert OpenAI-style message content into text fragments."""
    if content is None:
        return [""]
    if isinstance(content, str):
        return [content]
    if isinstance(content, list):
        fragments: list[str] = []
        for part in content:
            if isinstance(part, str):
                fragments.append(part)
            elif isinstance(part, dict):
                part_type = part.get("type") or part.get("t")
                if part_type in (None, "text"):
                    fragments.append(
                        part.get("text") or part.get("content") or part.get("value") or ""
                    )
                else:
                    raise ValueError(
                        "openai_messages_jsonl only supports text content parts, "
                        f"got type={part_type!r}"
                    )
            else:
                raise ValueError(f"Unsupported OpenAI message content part: {type(part)}")
        return fragments
    raise ValueError(f"Unsupported OpenAI message content: {type(content)}")


def _openai_role_to_sender(role: str) -> str:
    if role in ("system", "user", "assistant", "tool"):
        return role
    if role == "function":
        return "tool"
    if role == "human":
        return "user"
    if role == "gpt":
        return "assistant"
    raise ValueError(f"Unsupported OpenAI message role: {role!r}")


def _normalize_nano_sft_text_messages(messages: list[dict]) -> list[Message]:
    """Match the text cleanup used by the Nano 3.5 offline SFT packer."""
    conversation = []
    for msg in messages:
        if not isinstance(msg, dict):
            raise ValueError(f"OpenAI messages entries must be objects, got {type(msg)}")
        sender = _openai_role_to_sender(msg["role"])
        content = "".join(_openai_message_content_to_fragments(msg.get("content")))
        conversation.append(Message(sender=sender, fragments=[content]))

    if conversation[0].sender != "system":
        first_content = conversation[0].fragments[0]
        if first_content.startswith(EMPTY_SYSTEM_CONTENT):
            conversation[0].fragments[0] = first_content.replace(EMPTY_SYSTEM_CONTENT, "")
        conversation = [Message(sender="system", fragments=[EMPTY_SYSTEM_CONTENT])] + conversation
    elif conversation[0].fragments[0] in (NO_TOOL_SYSTEM_CONTENT, LEGACY_SYSTEM_CONTENT):
        conversation[0].fragments[0] = EMPTY_SYSTEM_CONTENT

    for message in conversation:
        if message.sender == "tool":
            message.sender = "user"

        content = message.fragments[0]
        if (
            message.sender == "user"
            and "<|im_end|>\n<|im_start|>assistant\n<think></think>\n" in content
        ):
            message.fragments[0] = content.replace(
                "<|im_end|>\n<|im_start|>assistant\n<think></think>\n",
                "<|im_end|>\n<|im_start|>assistant\n<think></think>",
            )
        elif message.sender == "assistant":
            message.fragments[0] = content.rstrip() + "\n"

    for idx, message in enumerate(conversation):
        content = message.fragments[0]
        if message.sender == "user" and idx < len(conversation) - 1:
            next_message = conversation[idx + 1]
            if content.endswith(
                "<|im_end|>\n<|im_start|>assistant\n<think>\n"
            ) and next_message.fragments[0].startswith("\n</think>"):
                message.fragments[0] = content.replace(
                    "<|im_end|>\n<|im_start|>assistant\n<think>\n",
                    "<|im_end|>\n<|im_start|>assistant\n<think></think>",
                )
                next_message.fragments[0] = next_message.fragments[0][len("\n</think>") :].lstrip()
        elif (
            message.sender == "assistant"
            and idx > 0
            and content.startswith("\n")
            and conversation[idx - 1].fragments[0].endswith("\n")
        ):
            message.fragments[0] = content.lstrip()

    return conversation


def _split_openai_messages_at_system(messages: list[dict]) -> list[list[dict]]:
    """Split an offline-packed Nano SFT row into original conversations."""
    conversations: list[list[dict]] = []
    current: list[dict] = []
    for message in messages:
        if not isinstance(message, dict):
            raise ValueError(f"OpenAI messages entries must be objects, got {type(message)}")
        if message.get("role") == "system" and current:
            conversations.append(current)
            current = []
        current.append(message)
    if current:
        conversations.append(current)
    return conversations


@stateless
@cooker(need_cache=True)
def cook_openai_messages_jsonl(
    sample: dict, cache: CachePool, media_source: FileStore | None = None
) -> ConversationSample:
    """Load OpenAI-style JSONL rows with messages[*].role/content."""
    data = sample["json"]
    messages = data.get("messages")
    if not isinstance(messages, list) or not messages:
        raise ValueError("openai_messages_jsonl requires a non-empty messages list")
    _validate_openai_messages_have_no_explicit_loss(messages)

    conversation = _normalize_nano_sft_text_messages(messages)
    for idx, msg in enumerate(conversation):
        sender = msg.sender
        if sender == "system" and idx > 0:
            raise ValueError(
                "openai_messages_jsonl only supports a leading system message. "
                "Split rows with repeated system prompts before using this cooker."
            )

    return ConversationSample(conversation=conversation, **basic_sample_keys(sample))


@stateless
@cooker(need_cache=True)
def cook_openai_messages_offline_packed_jsonl(
    sample: dict, cache: CachePool, media_source: FileStore | None = None
) -> ConversationSample:
    """Load Nano-style offline-packed JSONL rows with merged ``messages``.

    Each row is one already-packed training item. The row may contain multiple
    conversations concatenated together, separated by repeated ``system`` turns.
    Unlike ``openai_messages_jsonl``, this cooker preserves those repeated
    systems so the task encoder can tokenize each original conversation
    separately and emit packed ``cu_lengths`` without running online packing.
    """
    data = sample["json"]
    messages = data.get("messages")
    if not isinstance(messages, list) or not messages:
        raise ValueError("openai_messages_offline_packed_jsonl requires a non-empty messages list")
    _validate_openai_messages_have_no_explicit_loss(messages)

    conversation: list[Message] = []
    for split_messages in _split_openai_messages_at_system(messages):
        conversation.extend(_normalize_nano_sft_text_messages(split_messages))

    if not conversation:
        raise ValueError("openai_messages_offline_packed_jsonl produced an empty conversation")

    sample_keys = basic_sample_keys(sample)
    subflavors = dict(sample_keys.get("__subflavors__", {}) or {})
    subflavors["offline_packed_messages"] = True
    sample_keys["__subflavors__"] = subflavors
    return ConversationSample(conversation=conversation, **sample_keys)


def _basic_sample_keys_with_json_dataset(sample: dict, data: dict) -> dict:
    """Preserve optional raw JSONL dataset metadata for task-encoder filters."""
    sample_keys = basic_sample_keys(sample)
    dataset_name = data.get("dataset")
    if dataset_name is not None:
        subflavors = dict(sample_keys.get("__subflavors__", {}) or {})
        subflavors["dataset"] = dataset_name
        sample_keys["__subflavors__"] = subflavors
    return sample_keys


@stateless
@cooker(need_cache=True)
def cook_conversation(
    sample: dict,
    cache: CachePool,
    media_source: FileStore | None = None,
    **media_sources: FileStore,
) -> ConversationSample:
    global warn_about_slow_media_loading

    data = sample["json"]
    cs = ConversationSample.from_json(data, **_basic_sample_keys_with_json_dataset(sample, data))

    for msg in cs.conversation:
        for frag in msg.fragments:
            if isinstance(frag, (ImageMedia, VideoMedia, VideoFrameMedia)):
                current_media_source = media_source
                media_path = frag.value
                if (
                    current_media_source is None
                    and media_sources
                    and "aux_data_prefixes" in cs.__subflavors__
                ):
                    path = _re_clean_path.sub("", media_path)
                    for prefix, aux_key in cs.__subflavors__["aux_data_prefixes"].items():
                        if path.startswith(prefix):
                            if aux_key not in media_sources:
                                raise ValueError(f"Unknown auxiliary media source {aux_key!r}")
                            current_media_source = media_sources[aux_key]
                            media_path = path[len(prefix) :]
                            break
                    else:
                        raise ValueError(
                            f"No prefix for {path!r} in {cs.__subflavors__['aux_data_prefixes']} "
                            f"for {cs.__sources__}"
                        )
                if current_media_source is None:
                    raise ValueError(
                        "cook_conversation requires media_source for samples with media fragments"
                    )
                if frag.metadata is None:
                    try:
                        frag.metadata = dataclasses.asdict(
                            current_media_source.get_media_metadata(media_path)
                        )
                    except Exception as e:
                        if warn_about_slow_media_loading[current_media_source.get_path()]:
                            print(
                                f"WARNING: Dataset {current_media_source.get_path()} not prepared with media "
                                f"metadata, slow metadata for {media_path}: {e!r}"
                            )
                            warn_about_slow_media_loading[current_media_source.get_path()] = False
                cs.__sources__ = (
                    *cs.__sources__,
                    SourceInfo(
                        dataset_path=current_media_source.get_path(),
                        index=media_path,
                        shard_name=None,
                        file_names=(media_path,),
                    ),
                )
                frag.value = cache.get_lazy(current_media_source, media_path)
            elif isinstance(frag, str):
                # No source
                pass
            else:
                raise ValueError(f"Unknown fragment type: {type(frag)}")

    return cs


@stateless
@cooker(need_cache=True, need_primary=True)
def cook_general_conversations_webdataset(
    sample: dict,
    cache: CachePool,
    primary: FileStore,
    media_source: FileStore | None = None,
    **media_sources: FileStore,
) -> ConversationSample:
    """Loads general conversation-based datasets of webdataset format (monolithic or polylithic).
    Each sample can be single/multi-turn converstaions with multiple modalities.
    Each modality can have one or more number of media objects.
    Media tags such as ``<image>`` and ``<video>`` may appear in any conversation turn.
    """

    data = sample["json"].copy()
    _validate_standard_jsonl_has_no_explicit_loss(data)
    for tag in conversation_tags_mapping_sample_to_allowed:
        if tag in data and tag != conversation_tags_mapping_sample_to_allowed[tag]:
            sample['json'][conversation_tags_mapping_sample_to_allowed[tag]] = data[tag]
            del sample['json'][tag]
    # update the data
    data = sample["json"].copy()

    media_index = defaultdict(int)
    tried_default_extensions = set()

    # Build the conversation
    conversation = []
    for msg in data["conversations"]:
        conversation.append(
            conversation_convert_message(
                data,
                msg,
                media_index,
                raw=sample,
                check_if_media_file_exist=False,
                tried_default_extensions=tried_default_extensions,
                tags_mapping_sample_to_allowed=conversation_tags_mapping_sample_to_allowed,
            )
        )

    # Check if all media files are retrieved
    for media in media_index:
        medias = data[media]
        if not isinstance(medias, list):
            medias = [medias]
        if media_index[media] != len(medias):
            raise ValueError(
                f"Retrieved {media_index[media]}/{len(medias)} {media} files from {sample}"
            )

    return conversation_post_processing(
        conversation,
        sample,
        cache,
        primary=primary,
        media_source=media_source,
        **media_sources,
    )


def _cook_general_conversations_jsonl(
    sample: dict,
    cache: CachePool,
    primary: FileStore,
    media_source: FileStore | None = None,
    *,
    explicit_assistant_loss: bool,
    **media_sources: FileStore,
) -> ConversationSample:
    """Loads general conversation datasets that have the json (manifest) files and media files in separate files (jsonl datasets).
    The json(l) file structure is the same as the cook_general_conversations_webdataset
    """
    data = sample["json"].copy()

    if explicit_assistant_loss:
        _validate_explicit_assistant_loss_contract(data, sample.get("__subflavors__", {}))
    else:
        _validate_standard_jsonl_has_no_explicit_loss(data)

    for tag in conversation_tags_mapping_sample_to_allowed:
        if tag in data and tag != conversation_tags_mapping_sample_to_allowed[tag]:
            sample["json"][conversation_tags_mapping_sample_to_allowed[tag]] = data[tag]
            del sample["json"][tag]

    data = sample["json"].copy()

    media_index = defaultdict(int)
    tried_default_extensions = set()

    # Build the conversation
    conversation = []
    for msg in data["conversations"]:
        conversation.append(
            conversation_convert_message(
                data,
                msg,
                media_index,
                check_if_media_file_exist=False,
                tried_default_extensions=tried_default_extensions,
                tags_mapping_sample_to_allowed=conversation_tags_mapping_sample_to_allowed,
                loss=msg.get("loss") if explicit_assistant_loss else None,
            )
        )

    # Check if all media files are retrieved
    for media in media_index:
        medias = data[media]
        if not isinstance(medias, list):
            medias = [medias]
        if media_index[media] != len(medias):
            raise ValueError(
                f"Retrieved {media_index[media]}/{len(medias)} {media} files from {sample}"
            )

    return conversation_post_processing(
        conversation,
        sample,
        cache,
        primary=primary,
        media_source=media_source,
        **media_sources,
    )


@stateless
@cooker(need_primary=True, need_cache=True)
def cook_general_conversations_jsonl(
    sample: dict,
    cache: CachePool,
    primary: FileStore,
    media_source: FileStore | None = None,
    **media_sources: FileStore,
) -> ConversationSample:
    """Load legacy general-conversation JSONL without explicit loss metadata."""
    return _cook_general_conversations_jsonl(
        sample, cache, primary, media_source, explicit_assistant_loss=False, **media_sources
    )


@stateless
@cooker(need_primary=True, need_cache=True)
def cook_general_conversations_jsonl_explicit_loss_v1(
    sample: dict,
    cache: CachePool,
    primary: FileStore,
    media_source: FileStore | None = None,
    **media_sources: FileStore,
) -> ConversationSample:
    """Load versioned JSONL with explicit per-assistant-turn loss flags."""
    return _cook_general_conversations_jsonl(
        sample, cache, primary, media_source, explicit_assistant_loss=True, **media_sources
    )
