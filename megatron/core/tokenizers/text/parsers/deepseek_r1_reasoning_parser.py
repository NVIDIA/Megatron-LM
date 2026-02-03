from megatron.core.tokenizers.text.parsers.base_parser import BaseParser

class DeepSeekR1ReasoningParser(BaseParser):
    @staticmethod
    def parse(text: str, **kwargs) -> tuple[str, dict[str, str]]:
        """
        Extracts the reasoning content from the text using <think>...</think> tags.

        Args:
            text (str): The text to parse.

        Returns:
            tuple[str, dict[str, str]]: A tuple containing the unprocessed text
            and a dictionary with the extracted reasoning content.
        """

        if "</think>" in text:
            if "<think>" in text:
                # Strip the <think> prefix (it might not be present if it was part of the prompt)
                text = text.split("<think>")[1]
            reasoning_content = text.split("</think>")[0]
            return text.split("<think>")[0]+text.split("</think>")[-1], {'reasoning': reasoning_content}
        else:
            return text, {}