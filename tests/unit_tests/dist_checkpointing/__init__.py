import os
import weakref
from pathlib import Path
from shutil import rmtree
from tempfile import TemporaryDirectory
from typing import Union, Optional

from tests.unit_tests.test_utilities import Utils


def empty_dir(path: Path):
    if Utils.rank > 0:
        return
    for p in path.iterdir():
        if p.is_dir():
            rmtree(p)
        else:
            p.unlink()



class TempNamedDir(TemporaryDirectory):
    """ TemporaryDirectory with a fully named directory. Empties the dir if not empty. """
    def __init__(self, name: Union[str, Path], sync=True,
                 ignore_cleanup_errors=False) -> None:
        self.name = str(name)
        if Utils.rank == 0:
            os.makedirs(name, exist_ok=True)
            empty_dir(Path(name))

        self._ignore_cleanup_errors = ignore_cleanup_errors
        self._finalizer = weakref.finalize(
            self, self._cleanup, self.name,
            warn_message="Implicitly cleaning up {!r}".format(self))
        self.sync = sync

    def cleanup(self, override_sync: Optional[bool] = None) -> None:
        sync = self.sync if override_sync is None else override_sync
        if sync :
            import torch
            torch.distributed.barrier()

        if Utils.rank == 0:
            super().cleanup()

    def __enter__(self):
        return Path(super().__enter__())

    def __exit__(self, exc_type, exc_val, exc_tb):
        raised = exc_type is not None
        self.cleanup(False if raised else None)

