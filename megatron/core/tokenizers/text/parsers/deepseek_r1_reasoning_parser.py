from megatron.core.tokenizers.text.parsers.base_parser import BaseParser


class DeepSeekR1ReasoningParser(BaseParser):
    @staticmethod
    def parse(text: str, **kwargs) -> tuple[str, dict[str, str]]:
        """
        Extracts the reasoning content from the text using <think>...</think> tags.
        Only extracts the first set of think tags.
        If an initial <think> tag is not present but a </think> tag is, it will infer a <think> tag at the beginning of the text.

        Args:
            text (str): The text to parse.

        Returns:
            tuple[str, dict[str, str]]: A tuple containing the unprocessed text
            and a dictionary with the extracted reasoning content.
        """

        if "</think>" in text:
            if "<think>" in text:
                # Strip the <think> prefix (it might not be present if it was part of the prompt)
                pre_text, text = text.split("<think>", maxsplit=1)
            else:
                pre_text = ""
            reasoning_content, remaining_text = text.split("</think>", maxsplit=1)
            return pre_text + remaining_text, {'reasoning': reasoning_content}
        else:
            return text, {}
