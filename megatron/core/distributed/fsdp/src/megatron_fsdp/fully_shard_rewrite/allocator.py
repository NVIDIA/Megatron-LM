import dataclasses

import torch

from .utils import ParamGroupIdx


@dataclasses.dataclass
class Bucket:
    data: torch.Tensor


class TemporaryBucketAllocator:
    """Manages temporary flat buffers keyed by param_group_id.

    Used by DataParallelBuffer for unshard (all-gather) and gradient
    reduction (reduce-scatter) operations.
    """

    def __init__(self):
        self.buckets = {}

    def allocate(
        self, param_group_id: ParamGroupIdx, size: int, dtype: torch.dtype, device: torch.device
    ) -> Bucket:
        if param_group_id not in self.buckets:
            self.buckets[param_group_id] = Bucket(
                data=torch.empty(size, dtype=dtype, device=device)
            )
        return self.buckets[param_group_id]

    def free(self, param_group_id: ParamGroupIdx) -> None:
        if param_group_id in self.buckets:
            _free_storage(self.buckets[param_group_id].data)
            del self.buckets[param_group_id]


def _free_storage(tensor: torch.Tensor):
    """
    Frees the underlying storage of ``tensor``.

    Returns:
        bool: ``True`` if the method freed the storage and ``False`` if the
        storage was already freed.
    """
    with torch.no_grad():
        if not torch.distributed._functional_collectives.is_torchdynamo_compiling():
            already_freed = tensor._typed_storage()._size() == 0
            if not already_freed:
                assert tensor.storage_offset() == 0, (
                    "Freeing a tensor's storage is unsafe when it is not the sole occupant\n"
                    f"storage offset: {tensor.storage_offset()}\n"
                    f"storage size: {tensor._typed_storage()._size()}\n"
                    f"tensor shape: {tensor.shape}"
                )
                tensor._typed_storage()._resize_(0)
