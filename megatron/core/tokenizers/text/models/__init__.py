# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

# The individual model wrapper classes (GPTTokenizer, BertTokenizer, etc.) have been removed
# as they were empty subclasses that added no functionality. These aliases are kept for
# backward compatibility with any code that imports them by name.
from megatron.core.tokenizers.text.text_tokenizer import MegatronTokenizerText as BertTokenizer
from megatron.core.tokenizers.text.text_tokenizer import (
    MegatronTokenizerText as DefaultTokenizerText,
)
from megatron.core.tokenizers.text.text_tokenizer import MegatronTokenizerText as GPTTokenizer
from megatron.core.tokenizers.text.text_tokenizer import MegatronTokenizerText as MambaTokenizer
from megatron.core.tokenizers.text.text_tokenizer import MegatronTokenizerText as T5Tokenizer
