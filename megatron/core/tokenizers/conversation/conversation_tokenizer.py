# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

from typing import Callable, Dict, List, Optional, Tuple, Union

import numpy as np

from megatron.core.tokenizers.conversation.prompt_config import PromptConfig

IGNORE_INDEX = -100


def tokenize_conversation(
    tokenizer,
    conversation: List[Dict],
    prompt_config: PromptConfig,
    return_target: bool,
    add_generation_prompt: bool,
    apply_image_tag_fn: Optional[Callable] = None,
) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """Tokenize a conversation with optional target masking for training.

    This is the shared implementation used by both SFT and multimodal tokenizers.

    Args:
        tokenizer: A tokenizer instance with ``apply_chat_template`` support.
        conversation: Sequence of system/user/assistant messages in the format:
            [{"role": "system", "content": "..."}, {"role": "user", "content": "..."}, ...]
        prompt_config: Configuration controlling tokenization and masking behavior.
        return_target: If True, return target tokens with system/user tokens masked.
        add_generation_prompt: If True, add assistant prefix to the end.
        apply_image_tag_fn: Optional callback to apply image tags to conversation
            (used by multimodal tokenizer).

    Returns:
        tokens (np.ndarray): Token IDs for the conversation.
        target (np.ndarray): Target token IDs with masked positions set to IGNORE_INDEX.
            Only returned if return_target is True.
    """
    # 1. Skip system message if the tokenizer doesn't have a system role.
    if not prompt_config.has_system_role and conversation[0]["role"] == "system":
        conversation = conversation[1:]

    # 2. Force system message if configured.
    if prompt_config.force_system_message:
        assert (
            prompt_config.system_default is not None
        ), "Trying to force system message with empty system default"
        if conversation[0]["role"] == "system":
            conversation[0] = prompt_config.system_default
        else:
            conversation = [prompt_config.system_default] + conversation

    # 3. Capitalize roles if needed (e.g. nemotron5-aligned format).
    if prompt_config.capitalize_roles:
        for turn in conversation:
            role = turn['role']
            turn['role'] = role[:1].upper() + role[1:]

    # 4. Apply image tags if callback provided (multimodal).
    if apply_image_tag_fn is not None:
        conversation = apply_image_tag_fn(conversation)

    # 5. Tokenize with chat template.
    result = tokenizer.apply_chat_template(
        conversation,
        tokenize=True,
        add_generation_prompt=add_generation_prompt,
        chat_template=prompt_config.custom_chat_template,
    )
    # Normalize to 1D numpy array regardless of backend.
    if isinstance(result, np.ndarray):
        tokens = result.flatten()
    else:
        tokens = np.array(result, dtype=np.int64)

    if not return_target:
        return tokens

    target = tokens.copy()

    # 6. Skip masking if configured (e.g. SFT "default" format).
    if prompt_config.skip_masking:
        return tokens, target

    # 7. Mask system and user tokens in the target.
    masked_roles = {"system", "user"}
    if prompt_config.allow_tool_role:
        masked_roles.add("tool")

    idx = 0
    for turn_idx, turn in enumerate(conversation):
        role = turn["role"].lower()

        # Validate empty turns.
        if prompt_config.allow_tool_role:
            # SFT behavior: only check assistant turns for empty content.
            if role == "assistant" and len(turn["content"]) == 0:
                raise ValueError(f"empty assistant turn in conversation: {conversation}.")
            if role == "assistant":
                assert conversation[turn_idx - 1]["role"].lower() in ("user", "tool")
        else:
            # Multimodal behavior: check all turns for empty content.
            if len(turn["content"]) == 0:
                raise ValueError(f"empty turn in conversation: {conversation}. Skipping.")

        # Validate no image token in assistant content.
        if prompt_config.validate_no_image_in_assistant and role == "assistant":
            try:
                from megatron.core.models.multimodal.llava_model import IMAGE_TOKEN

                if IMAGE_TOKEN in turn["content"]:
                    raise RuntimeError(f"{IMAGE_TOKEN} not allowed in assistant content!")
            except ImportError:
                pass

        turn_result = tokenizer.apply_chat_template(
            [turn], tokenize=True, chat_template=prompt_config.custom_chat_template
        )
        turn_tokens = list(turn_result) if not isinstance(turn_result, list) else turn_result

        # There should be only one BOS at the very beginning.
        # After the first turn, skip BOS token.
        if prompt_config.has_bos and turn_idx > 0:
            turn_tokens = turn_tokens[1:]
        turn_len = len(turn_tokens)

        if role in masked_roles:
            target[idx : idx + turn_len] = IGNORE_INDEX
        elif role == "assistant":
            if prompt_config.assistant_prefix_len > 0:
                target[idx : idx + prompt_config.assistant_prefix_len] = IGNORE_INDEX
        else:
            raise ValueError(f"Wrong role value: {role}")

        assert np.allclose(
            tokens[idx : idx + turn_len], turn_tokens
        ), f"expected turn tokens to match tokens in conversation {conversation}"

        idx += turn_len

    assert idx == len(tokens), f"mismatch in target masking the conversation {conversation}"

    return tokens, target
