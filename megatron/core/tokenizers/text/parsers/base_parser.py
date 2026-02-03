class BaseParser:
    @staticmethod
    def parse(text: str, **kwargs) -> tuple[str, dict[str, str]]:
        """
        Parses the text into a tuple containing extracted content
        and a dictionary of additional information.

        Args:
            text (str): The text to parse.

        Returns:
            tuple[str, dict[str, str]]: A tuple containing the unprocessed text
            and a dictionary with the extracted information.
        """
        return text, {}