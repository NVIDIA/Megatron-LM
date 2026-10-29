# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Bind a BinIdx ownership companion to one immutable prepared-token prefix."""

import json

from megatron.energon.decoders.sample_decoder import SampleDecoder
from megatron.energon.epathlib import EPath
from megatron.energon.flavors.binidx.binidx_reader import BinIdxReader
from megatron.energon.flavors.binidx.token_mask import TokenMaskFileStore
from megatron.energon.recipe.recipe import AuxFilesystemReference

_KEY_PREFIX = "token-loss-mask:"


def token_mask_key(primary, index: int) -> str:
    return _KEY_PREFIX + json.dumps([str(primary), index]) + ".mask"


class BinIdxTokenMaskDecoder(SampleDecoder):
    """Bypass decoding only for reserved ownership keys.

    Delegate everything else to ordinary ``SampleDecoder``.
    """

    def decode(self, fname, raw):
        if fname.startswith(_KEY_PREFIX) and fname.endswith(".mask"):
            return raw
        return super().decode(fname, raw)


class PreparedTokenMaskStore(TokenMaskFileStore):
    """A companion must share its intended primary's prefix and document count."""

    def __init__(self, path):
        path = EPath(path)
        if not path.name.endswith(".mask"):
            raise ValueError("indexed-token-mask requires a same-prefix .mask companion")
        self.primary_path = path.parent / (path.name.removesuffix(".mask") + ".bin")
        super().__init__(path)
        if self.document_count != BinIdxReader.count_samples(self.primary_path):
            raise ValueError(f"token mask document count disagrees with {self.primary_path}")

    def __getitem__(self, key):
        primary, index = json.loads(key.removeprefix(_KEY_PREFIX).removesuffix(".mask"))
        if EPath(primary) != self.primary_path:
            raise ValueError(f"token mask {self.path} is bound to primary {self.primary_path}")
        if type(index) is not int:
            raise ValueError("token mask requires an original integer document index")
        return super().__getitem__(f"{index}.mask")

    def get_path(self):
        return json.dumps(
            {
                "mask": str(self.path),
                "primary": str(self.primary_path),
                "format": 2,
                "ownership": "target-token-v1",
            },
            sort_keys=True,
        )


class IndexedTokenMaskReference(AuxFilesystemReference):
    """Recipe protocol ``indexed-token-mask://<primary-prefix>.mask``."""

    def get_file_store(self):
        if not isinstance(self.fs_path, EPath):
            raise ValueError("indexed token mask requires a resolved recipe path")
        return PreparedTokenMaskStore(self.fs_path)
