# Compatibility stub: this module was moved to megatron.core.tokenizers.megatron_tokenizer
# Re-export for backwards compatibility with Megatron-Bridge
from megatron.core.tokenizers.megatron_tokenizer import MegatronTokenizer

# MegatronLegacyTokenizer was removed; alias to MegatronTokenizer for Bridge compat
MegatronLegacyTokenizer = MegatronTokenizer
