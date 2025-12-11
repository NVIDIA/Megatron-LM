# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

from typing import Dict, List, Optional, Union

try:
    from transformers.utils.chat_template_utils import _compile_jinja_template

    HAVE_TRANSFORMERS = True
except ImportError:
    HAVE_TRANSFORMERS = False


class MegatronTokenizerChatTemplate:
    """Chat template class for Megatron text tokenizers."""

    def apply_chat_template(
        self,
        conversation: List[Dict[str, str]],
        chat_template: str,
        tokenize: Optional[bool] = True,
        truncation: Optional[bool] = False,
        max_length: Optional[int] = None,
        add_generation_prompt: Optional[bool] = False,
    ) -> Union[str, List[int]]:
        """
        Applies tokenizer's chat template to the conversation.

        Args:
            conversation (List[Dict[str, str]]): a list of dicts with "role" and "content" keys,
                representing the chat history so far. Conversation example:
                [
                    {"role": "system", "content": "You are a witty and helpful assistant."},
                    {"role": "user", "content": "Hey, what's a fun fact about octopuses?"},
                    {"role": "assistant", "content": "Octopuses blood is blue!"},
                    {"role": "user", "content": "Whoa, why is their blood blue?"},
                ]
            tokenize (bool): whether to tokenize the output. If `False`,
                the output will be a string.
            truncation (bool): whether to truncate sequences at the maximum length.
                Has no effect if tokenize is `False`.
            max_length (int): maximum length to use fro truncation.
                Has no effect if tokenize is `False`.
            add_generation_prompt (bool): If this is set, a prompt with the token(s) that indicate
                the start of an assistant message will be appended to the formatted output.
                This is useful when you want to generate a response from the model.
                Note that this argument will be passed to the chat template,
                and so it must be supported in the template for this argument to have any effect.
        """

        assert chat_template, (
            "Chat template is not defined. "
            "Please, specify tokenizer chat template in the metadata file."
        )
        if truncation:
            assert max_length, "max_length must be specified if truncation is used."

        if HAVE_TRANSFORMERS:
            compiled_template = _compile_jinja_template(chat_template)
            chat_text = compiled_template.render(
                messages=conversation, add_generation_prompt=add_generation_prompt
            )

            if tokenize:
                chat_ids = self.text_to_ids(chat_text)
                if truncation:
                    chat_ids = chat_ids[:max_length]
                return chat_ids

            return chat_text
        else:
            raise ModuleNotFoundError("Please, install transformers library.")
