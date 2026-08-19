"""Batch-invariant DeepSeek-V4 training with vLLM-visible kernels."""

from importlib import import_module

_EXPORTS = {
    "DeepseekV4Model": ".model",
    "ImplConfig": ".protocol",
    "DS4SparseAttentionMetadataBuilderAdapter": ".runtime_metadata",
    "DS4SparseIndexerCompressorMetadataAdapter": ".runtime_metadata",
}


def __getattr__(name):
    module = _EXPORTS.get(name)
    if module is None:
        raise AttributeError(name)
    return getattr(import_module(module, __name__), name)


__all__ = list(_EXPORTS)
