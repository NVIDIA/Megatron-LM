from typing import Callable

import torch.nn as nn
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor.placement_types import Shard

from .mixed_precision import MixedPrecisionPolicy

def fully_shard(
    module,
    *,
    mesh: DeviceMesh | None = None,
    reshard_after_forward: bool | int | None = None,
    shard_placement_fn: Callable[[nn.Parameter], Shard | None] | None = None,
    mp_policy: MixedPrecisionPolicy = MixedPrecisionPolicy(),
    offload_policy: "OffloadPolicy" = None,
    ignored_params: set[nn.Parameter] | None = None,
):
    # FIXME: implement this function.
    pass
