"""DeepSeek-V4 checkpoint quantization policy shared by load and resync."""


def is_release_unquantized_weight(name: str) -> bool:
    """Return whether the official V4 release stores ``name`` unscaled."""

    if name in {"embed.weight", "head.weight", "norm.weight"}:
        return True
    if name.endswith("norm.weight") or name.endswith(".ffn.gate.weight"):
        return True
    if ".attn.compressor." in name:
        return True
    return ".attn.indexer." in name and not name.endswith(
        ".attn.indexer.wq_b.weight"
    )


__all__ = ["is_release_unquantized_weight"]
