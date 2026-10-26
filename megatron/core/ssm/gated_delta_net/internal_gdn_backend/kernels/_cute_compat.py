# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""CuTe DSL compatibility helpers shared by the fused GDR kernels.

cutlass-dsl 4.4.2 removed ``cpasync.TmaInfo``; ``make_tiled_tma_atom_*`` now returns a plain
``(CopyAtom, Tensor)`` tuple. The kernels keep passing one object per TMA operand through the
``@cute.jit`` / ``@cute.kernel`` boundary, so this module provides the MLIR-value shim and the
``_wrap_tma`` helper that both the forward and the backward kernel import.
"""

try:
    from cutlass.cute.nvgpu.cpasync import TmaInfo
except ImportError:
    from cutlass.base_dsl import extract_mlir_attributes as _ema
    from cutlass.base_dsl import extract_mlir_values as _emv
    from cutlass.base_dsl import get_mlir_types as _gmt
    from cutlass.base_dsl import new_from_mlir_values as _nfmv

    class TmaInfo:  # type: ignore[no-redef]
        """Compatibility shim replacing cpasync.TmaInfo for cutlass-dsl >= 4.4.2."""

        def __init__(self, atom, tma_tensor):
            self._atom = atom
            self._tma_tensor = tma_tensor

        @property
        def atom(self):
            return self._atom

        @property
        def tma_tensor(self):
            return self._tma_tensor

        def __extract_mlir_values__(self):
            return _emv(self._atom) + _emv(self._tma_tensor)

        def __extract_mlir_attributes__(self):
            return _ema(self._atom) + _ema(self._tma_tensor)

        def __new_from_mlir_values__(self, values):
            n = len(_gmt(self._atom))
            return TmaInfo(_nfmv(self._atom, values[:n]), _nfmv(self._tma_tensor, values[n:]))

        def __iter__(self):
            yield self._atom
            yield self._tma_tensor

        def __getitem__(self, i):
            return (self._atom, self._tma_tensor)[i]

        def __len__(self):
            return 2


def _wrap_tma(ret):
    """Wrap a make_tiled_tma_atom_* return value in TmaInfo if it is not one already."""
    if isinstance(ret, TmaInfo):
        return ret
    # cutlass-dsl >= 4.4.2 returns a (CopyAtom, Tensor) tuple.
    return TmaInfo(ret[0], ret[1])


__all__ = ["TmaInfo", "_wrap_tma"]
