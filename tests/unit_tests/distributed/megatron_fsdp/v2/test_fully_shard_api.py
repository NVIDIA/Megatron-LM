import pytest
import torch

from megatron.core.distributed.fsdp.src.megatron_fsdp.v2 import fully_shard


@pytest.mark.parametrize(
    ("arg_name", "arg_value"),
    [
        ("reshard_after_forward", False),
        ("shard_placement_fn", lambda param: None),
        ("offload_policy", object()),
    ],
)
def test_fully_shard_rejects_unsupported_pytorch_api_args(arg_name, arg_value):
    module = torch.nn.Linear(1, 1)

    with pytest.raises(
        NotImplementedError,
        match=f"Megatron FSDP v2 does not support `{arg_name}` yet.",
    ):
        fully_shard(module, **{arg_name: arg_value})

    assert type(module) is torch.nn.Linear
