import dataclasses

import torch


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
        self, param_group_id: int, size: int, dtype: torch.dtype, device: torch.device
    ) -> Bucket:
        if param_group_id not in self.buckets:
            self.buckets[param_group_id] = Bucket(
                data=torch.empty(size, dtype=dtype, device=device)
            )
        self.buckets[param_group_id].data._typed_storage()._resize_(size)
        return self.buckets[param_group_id]

    def free(self, param_group_id: int) -> None:
        if param_group_id in self.buckets:
            with torch.no_grad():
                storage = self.buckets[param_group_id].data._typed_storage()
                if storage._size() != 0:
                    storage._resize_(0)
            # del self.buckets[param_group_id]
