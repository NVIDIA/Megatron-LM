"""Adapter to expose MegaBlocks package, if available."""
try:
    import megablocks
except ImportError:
    megablocks = None

def megablocks_is_available():
    return megablocks is not None

def assert_megablocks_is_available():
    assert megablocks_is_available(), (
        'MegaBlocks not available. Please run `pip install megablocks`.')

moe = megablocks.layers.moe if megablocks_is_available() else None
dmoe = megablocks.layers.dmoe if megablocks_is_available() else None
arguments = megablocks.layers.arguments if megablocks_is_available() else None
